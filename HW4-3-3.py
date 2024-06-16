import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from sklearn.model_selection import StratifiedKFold
from PIL import Image
import glob
from tempfile import TemporaryDirectory
import sklearn

# Define the DenseBlock
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, 4*growth_rate, kernel_size=(1,1), stride=(1,1), bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        
        self.shortcut = nn.Sequential()
        if in_channels != growth_rate:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, growth_rate, kernel_size=(1,1), stride=(1,1), bias=False),
                nn.BatchNorm2d(growth_rate)
            )

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        
        out += self.shortcut(identity)
        
        return torch.cat((x, out), 1)

# Define the TransitionLayer
class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(TransitionLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1=nn.ReLU(inplace=True)
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=(1,1),bias=False)
        self.pool=nn.AvgPool2d(kernel_size=2,stride=2,padding=0)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.pool(out)
        return out

# Define Densenet
class Densenet(nn.Module):
    def __init__(self, block, layers, growth_rate=12, num_class=11, theta=0.5):
        super(Densenet, self).__init__()
        self.growth_rate = growth_rate

        self.conv1 = nn.Conv2d(3, 2*growth_rate, kernel_size=7, stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(2*growth_rate)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        in_channel = 2*growth_rate
        self.layer1 = self._make_denselayer(block, in_channel, layers[0])
        in_channel += layers[0]*growth_rate
        self.transition1 = self._make_tranlayer(TransitionLayer, in_channel, int(in_channel*theta))
        in_channel = int(in_channel*theta)

        self.layer2 = self._make_denselayer(block, in_channel, layers[1])
        in_channel += layers[1]*growth_rate
        self.transition2 = self._make_tranlayer(TransitionLayer, in_channel, int(in_channel*theta))
        in_channel = int(in_channel*theta)

        self.layer3 = self._make_denselayer(block, in_channel, layers[2])
        in_channel += layers[2]*growth_rate
        self.transition3 = self._make_tranlayer(TransitionLayer, in_channel, int(in_channel*theta))
        in_channel = int(in_channel*theta)

        self.layer4 = self._make_denselayer(block, in_channel, layers[3])
        in_channel += layers[3]*growth_rate

        self.bn_final = nn.BatchNorm2d(in_channel)

        # Calculate the size of the input to the fully connected layer
        self.fc_input_features = self._get_fc_input_features()

        self.fc = nn.Linear(self.fc_input_features, num_class, bias=True)

    def _make_denselayer(self, block, in_channel, blocks):
        layers = []
        for i in range(blocks):
            layers.append(block(in_channel + i*self.growth_rate, self.growth_rate))
        return nn.Sequential(*layers)

    def _make_tranlayer(self, block, in_channel, out_channel):
        return block(in_channel, out_channel)

    def _get_fc_input_features(self):
        # Create a dummy input tensor to calculate the size of the fully connected input features
        x = torch.zeros(1, 3, 224, 224)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.layer1(x)
        x = self.transition1(x)
        x = self.layer2(x)
        x = self.transition2(x)
        x = self.layer3(x)
        x = self.transition3(x)
        x = self.layer4(x)
        x = self.bn_final(x)
        return x.numel()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.layer1(x)
        x = self.transition1(x)
        x = self.layer2(x)
        x = self.transition2(x)
        x = self.layer3(x)
        x = self.transition3(x)
        x = self.layer4(x)
        x = self.bn_final(x)

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x

def build_densenet(class_number,trainable=True):
  model=Densenet(DenseBlock,[6,12,32,32],growth_rate=32,num_class=class_number,theta=0.5)
  return model

# Load CCSN dataset images and labels
def CCSN_images(dataset_dir):
    classfolder = glob.glob(os.path.join(dataset_dir, "*"), recursive=True)
    class_name = [f.split(os.path.sep)[-1] for f in classfolder]
    img_labels = []
    img_list = []

    for class_id, f in enumerate(classfolder):
        files = glob.glob(os.path.join(f, "*.jpg"), recursive=True)
        img_labels.extend([class_id] * len(files))
        img_list.extend(files)

    img_labels = np.array(img_labels)
    img_list = np.array(img_list, dtype=object)

    return img_list.reshape((-1, 1)), img_labels, class_name

# Define Densenet dataset for CCSN
class CCSNImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, img_labels, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.img_list = img_list
        self.img_labels = img_labels

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_list[idx, 0]
        image = Image.open(img_path)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# Paths and data loading
CCSNDataset_Path = "D:\大學\大三\機器視覺\HW4\database\Kaggle\CCSN\CCSN_v2"
img_list, img_labels, label_names = CCSN_images(CCSNDataset_Path)
CCSNDataset = CCSNImageDataset(img_list, img_labels)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print('GPU is available')
else:
    print('CPU only')

# Data augmentation and normalization for training
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

batch_size = 48
epochs = 100
patience = epochs // 5

test_accuracy = []
test_loss = []
histories = []

criterion = nn.CrossEntropyLoss()

# Training function
def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=25, patience=0, scheduler=None):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    since = time.time()

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_loss = None
        best_acc = 0
        history = {'loss':[], 'accuracy':[], 'val_loss':[], 'val_accuracy':[]}
        patience_c = 0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch:3d}/{num_epochs - 1}', end=' ')

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}', end=' ' if phase == 'train' else '\n')

                if phase == 'train':
                    history['loss'].append(epoch_loss)
                    history['accuracy'].append(epoch_acc.cpu().numpy())
                else:
                    history['val_loss'].append(epoch_loss)
                    history['val_accuracy'].append(epoch_acc.cpu().numpy())

                if phase == 'val':
                    if scheduler:
                        scheduler.step(epoch_loss)
                    if best_loss is None or epoch_loss < best_loss:
                        best_loss = epoch_loss
                        best_acc = epoch_acc.cpu().numpy()
                        torch.save(model.state_dict(), best_model_params_path)
                        patience_c = 0
                    else:
                        patience_c += 1
            if patience_c >= patience:
                break

        time_elapsed = time.time() - since
        print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val acc: {best_acc:4f}')

        model.load_state_dict(torch.load(best_model_params_path))
    return model, history

# Evaluation function
def evaluate(model, test_loader):
    model.eval()
    model.to(device)

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    test_loss = running_loss / len(test_loader.dataset)
    test_accuracy = running_corrects.double() / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f} Acc: {test_accuracy:.4f}')
    return test_loss, test_accuracy.cpu().numpy()

# Training and validation loop
for i, (train_idx, test_idx) in enumerate(skf.split(img_list, img_labels)):

    print('-'*50)

    X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(img_list[train_idx], img_labels[train_idx], test_size=0.2, random_state=i)

    testloader = torch.utils.data.DataLoader(CCSNImageDataset(img_list[test_idx], img_labels[test_idx], data_transforms['val'], lambda x: torch.tensor(x, dtype=torch.long)), batch_size=1, shuffle=True)
    trainloader = torch.utils.data.DataLoader(CCSNImageDataset(X_train, y_train, data_transforms['train'], lambda x: torch.tensor(x, dtype=torch.long)), batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(CCSNImageDataset(X_valid, y_valid, data_transforms['val'], lambda x: torch.tensor(x, dtype=torch.long)), batch_size=batch_size, shuffle=True)

    model = build_densenet(len(label_names),True)

    optimizer_ft = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=5,verbose=True)

    trained_model, history = train_model(model, criterion, optimizer_ft,
                                         dataloaders={'train': trainloader, 'val': validloader},
                                         dataset_sizes={'train': X_train.shape[0], 'val': X_valid.shape[0]},
                                         patience=patience,
                                         num_epochs=epochs,
                                         scheduler=scheduler)

    histories.append(history)
    print(f'Fold {i+1:2d}', end=' ')
    loss, acc = evaluate(trained_model, testloader)

    test_loss.append(loss)
    test_accuracy.append(acc)

# Plotting training and validation loss/accuracy
plt.figure(figsize=(8, 3*len(histories)))

max_loss = 0
max_acc = 0
for i, history in enumerate(histories):
    max_loss = max(max_loss, np.max(history['loss']), np.max(history['val_loss']))

max_loss *= 1.05
for i, history in enumerate(histories):
    plt.subplot(len(histories), 2, i*2+1)
    plt.title(f'fold:{i+1}')
    plt.plot(history['loss'], label='train')
    plt.plot(history['val_loss'], label='valid')
    plt.xlabel('epoch')
    step = int(np.ceil(len(history['loss']) / 5))
    plt.xticks(np.arange(0, len(history['loss']), step), [str(u+1) for u in np.arange(0, len(history['loss']), step)])
    plt.ylabel('loss')
    plt.ylim([0, max_loss])
    plt.grid(True)
    plt.legend()

    plt.subplot(len(histories), 2, i*2+2)
    plt.title(f'fold:{i+1}')
    plt.plot(history['accuracy'], label='train')
    plt.plot(history['val_accuracy'], label='valid')
    plt.xticks(np.arange(0, len(history['accuracy']), step), [str(u+1) for u in np.arange(0, len(history['accuracy']), step)])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim([0, 1.0])
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.show()

# Plotting average accuracy
plt.figure()
plt.bar(x=np.arange(len(test_accuracy)), height=np.array(test_accuracy))
plt.xlabel('fold')
plt.xticks(np.arange(len(test_accuracy)), [str(i+1) for i in np.arange(len(test_accuracy))])
plt.ylabel('accuracy')
plt.title(f'average accuracy rate: {np.mean(np.array(test_accuracy)):.3f} +/- {np.std(np.array(test_accuracy)):.3f}')
plt.grid(True)
plt.ylim([0, 1.0])
plt.show()

print(f'average accuracy rate:{np.mean(np.array(test_accuracy)):.3f}+/-{np.std(np.array(test_accuracy)):.3f}')