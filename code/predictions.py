import cv2
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from Model import Model

def predict_batch(model, batch, labels=['cloth', 'ffp2', 'surgical', 'without'], device="cpu"):
    """
    Given a batch of images, predict the class of each image.
    """
    model.eval()
    with torch.no_grad():
        batch = batch.to(device)
        output = model(batch)    
    preds = output.argmax(dim=1).cpu().numpy()
    return [labels[pred] for pred in preds]

def predict_image(model, image, labels=['cloth', 'ffp2', 'surgical', 'without'], device="cpu"):
    """
    Given an image, predict the class of the image.
    """
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image.unsqueeze(0))
    pred = output.argmax(dim=1).cpu().numpy()
    return labels[pred[0]]

if __name__ == "__main__":
    import os

    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    CFG = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'model_checkpoints': '/Users/shubhampatel/Documents/Comp 6721/Project/model/model.pt'
    }

    image = cv2.imread("/Users/shubhampatel/Downloads/IMG_5440.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    image = transform(image)
    model = Model(4)
    model = model.to(CFG['device'])
    ckpts = torch.load(CFG['model_checkpoints'], map_location=CFG['device'])
    model.load_state_dict(ckpts['model'])
    print(predict_image(model, image))

