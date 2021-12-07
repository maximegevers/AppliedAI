import matplotlib.pyplot as plt
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, ConcatDataset
from torchvision.datasets import ImageFolder

from Model import Model
from trainer import Trainer
from predictions import predict_batch, predict_image
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, f1_score
import glob
import tqdm
from PIL import Image
import os

def age_bias(model, data_path, device):
    transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    for age in ['child', 'mid', "age"]:
        preds = []
        true = []
        print("*"*50)
        print("Predictions for {} images.".format(age))
        age_path = os.path.join(data_path, age)
        for mask in ['cloth', 'ffp2', 'surgical', 'without']:
            preds_counts = {
                'cloth': 0,
                'ffp2':  0,
                'surgical': 0,
                'without': 0
            }
            masked_image_path = os.path.join(age_path, mask, "*.jpg")
            files =  glob.glob(masked_image_path)
            for image_path in tqdm.tqdm(files, total=len(files)):
                image = Image.open(image_path)
                image = transform(image)
                pred = predict_image(model, image, device=device)
                preds_counts[pred] += 1
            for x in preds_counts:
                preds.extend([x]*preds_counts[x])
            true.extend([mask]*len(files))

        print(classification_report(true, preds))
        cm = confusion_matrix(true, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['cloth', 'ffp2', 'surgical', 'without'])
        disp.plot()

def gender_bias(model, data_path, device):
    transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    for gender in ['male', 'female']:
        preds = []
        true = []
        print("*"*50)
        print("Predictions for {} images.".format(gender))
        gender_path = os.path.join(data_path, gender)
        for mask in ['cloth', 'ffp2', 'surgical', 'without']:
            preds_counts = {
                'cloth': 0,
                'ffp2':  0,
                'surgical': 0,
                'without': 0
            }
            masked_image_path = os.path.join(gender_path, mask, "*.jpg")
            files =  glob.glob(masked_image_path)
            for image_path in tqdm.tqdm(files, total=len(files)):
                image = Image.open(image_path)
                image = transform(image)
                pred = predict_image(model, image, device=device)
                preds_counts[pred] += 1
            for x in preds_counts:
                preds.extend([x]*preds_counts[x])
            true.extend([mask]*len(files))

        print(classification_report(true, preds))
        cm = confusion_matrix(true, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['cloth', 'ffp2', 'surgical', 'without'])
        disp.plot()

if __name__ == "__main__":
    CFG = {
        'age_data_path': '/Users/shubhampatel/Documents/Comp 6721/age',
        'gender_data_path': '/Users/shubhampatel/Downloads/gender_updated (1)',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'model_checkpoints': '/Users/shubhampatel/Documents/Comp 6721/Project/model/model.pt',
    }

    model = Model(4)
    model = model.to(CFG['device'])
    ckpts = torch.load(CFG['model_checkpoints'], map_location=CFG['device'])
    model.load_state_dict(ckpts['model'])
    model.to(CFG['device'])

    # gender_bias(model, CFG['gender_data_path'], CFG['device'])
    age_bias(model, CFG['age_data_path'], CFG['device'])
