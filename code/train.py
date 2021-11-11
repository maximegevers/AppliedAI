import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder

from Model import ConvModel
from trainer import Trainer


def get_data(images_path, val_split=0.25):
    """
    Loads the data from the given path and splits it into training and validation sets.
    :param images_path: The path to the images.
    :param val_split: The percentage of the data to be used for validation.
    :return: A tuple containing the training and validation sets.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.RandomHorizontalFlip(0.2),
        transforms.Resize((224,224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
    dataset = ImageFolder(images_path, transform=transform) 
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    train = Subset(dataset, train_idx)
    valid = Subset(dataset, val_idx)
    return train, valid

if __name__ ==  "__main__":
    # configurations
    CFG = {
        'data_path': '/Users/shubhampatel/Documents/Comp 6721/Project/updated_data',
        'train_BS': 64,
        'valid_BS': 64,
        'lr': 0.01,
        'grad_clip': 0.1,
        'weight_decay': 0.01,
        'epochs': 30,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_model_at': '/Users/shubhampatel/Documents/Comp 6721/Project/model/',
        'early_stopping': 3
    }

    # get data for training and validation
    train_data, valid_data = get_data(CFG['data_path'])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=CFG['train_BS'], shuffle=True, pin_memory=False)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=CFG['valid_BS'], shuffle=False, pin_memory=False)

    # create model
    model = ConvModel(4)
    model = model.to(CFG['device'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CFG['lr'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, CFG['lr'], epochs=CFG['epochs'], steps_per_epoch=len(train_loader))

    # setup model trainer
    trainer = Trainer(model=model, device=CFG['device'], optimizer=optimizer, criterion=criterion, scheduler=scheduler)

    # start training
    History = trainer.fit(ES=CFG['early_stopping'], model_path=CFG['save_model_at'], train_loader=train_loader, val_loader=valid_loader, epochs=CFG['epochs'], start_epoch=0, fold=0, train_BS=CFG['train_BS'], valid_BS=CFG['train_BS'], grad_clip=CFG['grad_clip'])

    # plot training history
    plt.plot(History['train_loss'], label='train')
    plt.plot(History['val_loss'], label='valid')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    plt.plot(History['train_acc'], label='train')
    plt.plot(History['val_acc'], label='valid')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(loc='lower right')
    plt.show()


