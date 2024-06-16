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

class inceptionv1_block(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2_step1, out_channels2_step2, out_channels3_step1, out_channels3_step2, out_channels4):
        super(inceptionv1_block, self).__init__()
        self.branch1_conv = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels1, kernel_size=1),
                          nn.ReLU(inplace=True))
        
        self.branch2_conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels2_step1, kernel_size=1),
                          nn.ReLU(inplace=True))
        self.branch2_conv2 = nn.Sequential(nn.Conv2d(in_channels=out_channels2_step1, out_channels=out_channels2_step2, kernel_size=3, padding=1),
                          nn.ReLU(inplace=True))
        
        self.branch3_conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels3_step1, kernel_size=1),
                          nn.ReLU(inplace=True))
        self.branch3_conv2 = nn.Sequential(nn.Conv2d(in_channels=out_channels3_step1, out_channels=out_channels3_step2, kernel_size=5, padding=2),
                          nn.ReLU(inplace=True))
        
        self.branch4_maxpooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels4, kernel_size=1),
                          nn.ReLU(inplace=True))
     
    def forward(self, x):
        out1 = self.branch1_conv(x)
        out2 = self.branch2_conv2(self.branch2_conv1(x))
        out3 = self.branch3_conv2(self.branch3_conv1(x))
        out4 = self.branch4_conv1(self.branch4_maxpooling(x))
        out = torch.cat([out1, out2, out3, out4], dim=1)

        return out
      
class auxiliary_classifiers(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(auxiliary_classifiers, self).__init__()
        self.avgpooling = nn.AvgPool2d(kernel_size=5, stride=3)
        
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1)
        
        self.fc1 = nn.Linear(in_features=3200, out_features=1024)

        self.fc2 = nn.Linear(in_features=1024, out_features=out_channels)
     
    def forward(self, x):
        x = self.avgpooling(x)
        x = nn.functional.relu(self.conv(x))
        x = torch.flatten(x, start_dim=1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.dropout(x, p=0.5)
        x = self.fc2(x)

        return x
      
class InceptionV1(nn.Module):
    def __init__(self, num_classes, training=True):
        super(InceptionV1, self).__init__()
        self.training = training
        self.conv = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
                      nn.ReLU(inplace=True),
                      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1),
                      nn.ReLU(inplace=True),
                      nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
                      nn.ReLU(inplace=True),
                      nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.inception1 = inceptionv1_block(in_channels=192, out_channels1=64, out_channels2_step1=96, out_channels2_step2=128, out_channels3_step1=16, out_channels3_step2=32, out_channels4=32)
        self.inception2 = inceptionv1_block(in_channels=256, out_channels1=128, out_channels2_step1=128, out_channels2_step2=192, out_channels3_step1=32, out_channels3_step2=96, out_channels4=64)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception3 = inceptionv1_block(in_channels=480, out_channels1=192, out_channels2_step1=96, out_channels2_step2=208, out_channels3_step1=16, out_channels3_step2=48, out_channels4=64)

        if self.training == True:
            self.auxiliary1 = auxiliary_classifiers(in_channels=512,out_channels=num_classes)

        self.inception4 = inceptionv1_block(in_channels=512 ,out_channels1=160, out_channels2_step1=112, out_channels2_step2=224, out_channels3_step1=24, out_channels3_step2=64, out_channels4=64)
        self.inception5 = inceptionv1_block(in_channels=512, out_channels1=128, out_channels2_step1=128, out_channels2_step2=256, out_channels3_step1=24, out_channels3_step2=64, out_channels4=64)
        self.inception6 = inceptionv1_block(in_channels=512, out_channels1=112, out_channels2_step1=144, out_channels2_step2=288, out_channels3_step1=32, out_channels3_step2=64, out_channels4=64)

        if self.training == True:
            self.auxiliary2 = auxiliary_classifiers(in_channels=528,out_channels=num_classes)

        self.inception7 = inceptionv1_block(in_channels=528, out_channels1=256, out_channels2_step1=160, out_channels2_step2=320, out_channels3_step1=32, out_channels3_step2=128, out_channels4=128)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception8 = inceptionv1_block(in_channels=832, out_channels1=256, out_channels2_step1=160, out_channels2_step2=320, out_channels3_step1=32, out_channels3_step2=128, out_channels4=128)
        self.inception9 = inceptionv1_block(in_channels=832, out_channels1=384, out_channels2_step1=192, out_channels2_step2=384, out_channels3_step1=48, out_channels3_step2=128, out_channels4=128)

        self.avgpooling = nn.AvgPool2d(kernel_size=7,stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(in_features=16384,out_features=num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.maxpooling1(x)
        x = self.inception3(x)
        aux1 = self.auxiliary1(x)
        x = self.inception4(x)
        x = self.inception5(x)
        x = self.inception6(x)
        aux2 = self.auxiliary2(x)
        x = self.inception7(x)
        x = self.maxpooling2(x)
        x = self.inception8(x)
        x = self.inception9(x)
        x = self.avgpooling(x)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        out = self.fc(x)

        if self.training == True:
            return aux1, aux2, out

        else:
            return out

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

# Define custom dataset for CCSN
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
        transforms.RandomResizedCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

batch_size = 64
epochs = 100
patience = epochs // 5

test_accuracy = []
test_loss = []
histories = []

criterion = nn.CrossEntropyLoss()

# Training function
def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, scheduler,num_epochs=25, patience=0):

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
                        if phase == 'train':
                            aux1, aux2, outputs = model(inputs)
                            loss1 = criterion(aux1, labels)
                            loss2 = criterion(aux2, labels)
                            loss_main = criterion(outputs, labels)
                            loss = loss_main + 0.3 * (loss1 + loss2)
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                        
                        _, preds = torch.max(outputs, 1)

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
            if patience_c > patience:
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

    model = InceptionV1(num_classes=len(label_names)).to(device)

    optimizer_ft = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=5)

    trained_model, history = train_model(model, criterion, optimizer_ft,
                                         dataloaders={'train': trainloader, 'val': validloader},
                                         dataset_sizes={'train': X_train.shape[0], 'val': X_valid.shape[0]},
                                         scheduler=scheduler,
                                         patience=patience,
                                         num_epochs=epochs)

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