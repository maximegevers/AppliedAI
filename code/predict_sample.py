from os import listdir
from os.path import isfile, join
from PIL import Image
import argparse
import torch
import torchvision.transforms as transforms
from Model import Model

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='Path to image files')
    parser.add_argument('--model', help='Path to model checkpoints')
    args = vars(parser.parse_args())
    files = [f for f in listdir(args['data']) if isfile(join(args['data'], f))]
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    preds = []
    labels = ['cloth', 'fffp2', 'surgical', 'without']
    model = Model(4)
    model = model.to(device)
    ckpts = torch.load(args['model'], map_location=device)['model']
    model.load_state_dict(ckpts)
    opt_dict = {}
    for path in [join(args['data'], f) for f in files if '.DS_Store' not in f]:
        image = Image.open(path)
        image = transform(image)
        predict = model(image.to(device).unsqueeze(0))
        opt_dict[path] = labels[predict.argmax(1)] 
        # preds.append(labels[predict.argmax(1)])
    
    print(opt_dict)