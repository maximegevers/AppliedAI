Files:

-> Model.py: Contains CNN model
-> trainer.py: Script for model training
-> train.py: Main file containing all the code to start training
-> predict_sample.py: To make predictions on a sample of 100 images.
-> prediction.py: To make prediction on a dataloader and single image.

How to start training:
-> train.py contains CFG dictionary where the necessary changes can be made such as data location, learning rate, where to save model etc.
-> After all the changes are made run: python train.py
-> After the training is completed evaluation will be shown, loss and acccuracy graph would be plotted, also the classification report and confusion matrix would be displayed.

To run the model on sample of 100 images:
python predict_sample --data [data path] --model [model checkpoints path]
e.g: python predict_sample --data ../data/ --model model/model.pth