import os
import glob
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
import time
from PIL import Image
import timm

def CCSN_images(dataset_dir):
    classfolder = glob.glob(os.path.join(dataset_dir,"*"),recursive=True)
    class_name  = [f.split(os.path.sep)[-1] for f in classfolder]
    img_labels  = []
    img_list    = []

    for class_id, f in enumerate(classfolder):
        files = glob.glob(os.path.join(f,"*.jpg"),recursive=True)
        img_labels.extend([class_id]*len(files))
        img_list.extend(files)

    img_labels = np.array(img_labels)
    img_list   = np.array(img_list,dtype=object)
    
    return img_list.reshape((-1,1)), img_labels, class_name

class CCSNImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, img_labels, transform=None, target_transform=None):
        self.transform        = transform
        self.target_transform = target_transform
        self.img_list         = img_list
        self.img_labels       = img_labels

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_list[idx,0]
        image    = Image.open(img_path)  
        label    = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
CCSNDataset_Path = "D:\大學\大三\機器視覺\HW4\database\Kaggle\CCSN\CCSN_v2"
img_list, img_labels, label_names = CCSN_images(CCSNDataset_Path) 
CCSNDataset = CCSNImageDataset(img_list,img_labels)

from tempfile import TemporaryDirectory

def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, scheduler,num_epochs=25, patience=0):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    since = time.time()

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_loss = None
        best_acc  = 0
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

                    if scheduler:
                        scheduler.step(epoch_loss)

                if phase == 'val':
                    history['val_loss'].append(epoch_loss)
                    history['val_accuracy'].append(epoch_acc.cpu().numpy())
                    if best_acc < epoch_acc.cpu().numpy():
                        best_acc = epoch_acc.cpu().numpy()
                    if epoch == 0 or epoch_loss < best_loss:
                        best_loss = epoch_loss
                        patience_c = 0
                        torch.save(model.state_dict(), best_model_params_path)
                    else:
                        patience_c += 1

            if patience_c > patience:
                break

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        model.load_state_dict(torch.load(best_model_params_path))
    return model, history

def evaluate(model, dataloader):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()

    running_loss = 0.0
    running_corrects = 0
    dataset_size = 0

    for inputs, labels in dataloader:

        dataset_size += inputs.size(0)

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects.double() / dataset_size

    print(f'test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return epoch_loss, epoch_acc.cpu().numpy()

def build_tnt(class_number, trainable=True):
    tnt = timm.create_model('tnt_s_patch16_224', pretrained=True)
    for param in tnt.parameters():
        param.requires_grad = trainable

    num_ftrs = tnt.head.in_features
    tnt.head = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_ftrs, class_number)
    )

    return tnt

from sklearn.model_selection import StratifiedKFold

device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print('gpu is available')
else:
    print('cpu only')

data_transforms = {
    'train': torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

skf        = StratifiedKFold(n_splits=5, shuffle=True, random_state=0) # 80 % for training and validation sets, 20 % for the test set

batch_size = 64
epochs     = 100
patience   = epochs//5

test_accuracy = []
test_loss     = []
histories     = []

criterion = nn.CrossEntropyLoss()

for i, (train_idx, test_idx) in enumerate(skf.split(img_list, img_labels)):

    print('-'*50)

    X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(img_list[train_idx], img_labels[train_idx], test_size=0.2, random_state=i)

    testloader = torch.utils.data.DataLoader(CCSNImageDataset(img_list[test_idx], img_labels[test_idx], data_transforms['val'], lambda x: torch.tensor(x, dtype=torch.long)), batch_size=1, shuffle=True)
    trainloader = torch.utils.data.DataLoader(CCSNImageDataset(X_train, y_train, data_transforms['train'], lambda x: torch.tensor(x, dtype=torch.long)), batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(CCSNImageDataset(X_valid, y_valid, data_transforms['val'], lambda x: torch.tensor(x, dtype=torch.long)), batch_size=batch_size, shuffle=True)

    tnt = build_tnt(len(label_names), True)
    tnt = tnt.to(device)

    optimizer_ft = optim.AdamW(tnt.parameters(), lr=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=10, verbose=True)

    trained_model, history = train_model(tnt, criterion, optimizer_ft,
                                         dataloaders={'train': trainloader, 'val': validloader},
                                         dataset_sizes={'train': X_train.shape[0], 'val': X_valid.shape[0]},
                                         patience=patience,
                                         scheduler=scheduler,
                                         num_epochs=epochs
                                         )

    histories.append(history)
    print(f'Fold {i+1:2d}', end=' ')
    loss, acc = evaluate(trained_model, testloader)

    test_loss.append(loss)
    test_accuracy.append(acc)

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