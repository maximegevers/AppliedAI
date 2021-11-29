import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, ConcatDataset
from torchvision.datasets import ImageFolder

from Model import Model
from trainer import Trainer
from predictions import predict_batch
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, f1_score


def get_data(images_path, val_split=0.15, test_split=0.1):
    """
    Loads the data from the given path and splits it into training and validation sets.
    :param images_path: The path to the images.
    :param val_split: The percentage of the data to be used for validation.
    :return: A tuple containing the training validation and testing sets.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.RandomHorizontalFlip(0.2),
        transforms.Resize((224,224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
    dataset = ImageFolder(images_path, transform=transform) 
    train_test_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    train_test = Subset(dataset, train_test_idx)
    valid = Subset(dataset, val_idx)


    train_idx, test_idx = train_test_split(list(range(len(train_test))), test_size=test_split)
    train = Subset(train_test, train_idx)
    test = Subset(train_test, test_idx)

    return train, valid, test

if __name__ ==  "__main__":
    # configurations
    CFG = {
        'data_path': '/content/updated_data',
        'train_BS': 64,
        'valid_BS': 64,
        'lr': 0.01,
        'grad_clip': 0.1,
        'weight_decay': 0.01,
        'epochs': 1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_model_at': '/content/',
        'early_stopping': 3,
        'folds': 10
    }

    # get data for training and validation
    train_data, valid_data, test_data = get_data(CFG['data_path'])
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=CFG['valid_BS'], shuffle=False, pin_memory=False)
    data = ConcatDataset([train_data, valid_data])

    kfold = KFold(n_splits=CFG['folds'], shuffle=True)

    accuracy = []
    recall = []
    f1 = []
    histories = []
    for fold, (train, valid) in enumerate(kfold.split(data)):

        train_sampler = torch.utils.data.SubsetRandomSampler(train)
        valid_sampler = torch.utils.data.SubsetRandomSampler(valid)

        train_loader = torch.utils.data.DataLoader(data, sampler=train_sampler, batch_size=CFG['train_BS'], pin_memory=False)
        valid_loader = torch.utils.data.DataLoader(data, sampler=valid_sampler, batch_size=CFG['valid_BS'], pin_memory=False)

        model = Model(4)
        model = model.to(CFG['device'])

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=CFG['lr'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, CFG['lr'], epochs=CFG['epochs'], steps_per_epoch=len(train_loader))

        # setup model trainer
        trainer = Trainer(model=model, device=CFG['device'], optimizer=optimizer, criterion=criterion, scheduler=scheduler)

        print("\n\nFOLD: {}".format(fold))
        # start training
        History = trainer.fit(ES=CFG['early_stopping'], model_path=CFG['save_model_at'], train_loader=train_loader, val_loader=valid_loader, epochs=CFG['epochs'], start_epoch=0, fold=fold, train_BS=CFG['train_BS'], valid_BS=CFG['train_BS'], grad_clip=CFG['grad_clip'])

        histories.append(History)
        plt.plot(History['train_loss'], label='train')
        plt.plot(History['val_loss'], label='valid')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig('loss.png')
        plt.show()

        plt.plot(History['train_acc'], label='train')
        plt.plot(History['val_acc'], label='valid')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend(loc='lower right')
        plt.savefig('accuracy.png')
        plt.show()

        ypred = predict_batch(model, test_loader, CFG['device'])
        true = []
        labels=['cloth', 'ffp2', 'surgical', 'without']
        for x, y in test_loader:
            true.extend(y.cpu().numpy())
        true = [labels[x] for x in true]

        print(classification_report(true, ypred))
        accuracy.append(accuracy_score(true, ypred))
        recall.append(recall_score(true, ypred, average="weighted"))
        f1.append(f1_score(true, ypred, average="weighted"))
        cm = metrics.confusion_matrix(true, ypred)
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = labels)
        disp.plot()

        np.save('history.npy', histories)
        np.save('accuracy.npy', accuracy)
        np.save('recall.npy', recall)
        np.save('f1.npy', f1)