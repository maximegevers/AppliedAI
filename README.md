# AppliedAI

Our Applied AI project. 

Files:

-> Model.py: Contains CNN model<br/>
-> trainer.py: Script for model training<br/>
-> train.py: Main file containing all the code to start training<br/>
-> predict_sample.py: To make predictions on a sample of 100 images.<br/>
-> prediction.py: To make prediction on a dataloader and single image.<br/>

Data Scrapping: https://drive.google.com/drive/folders/1jqD6_SCZnZppG6UzILw_RjsPt1LMPJF8?usp=sharing

Training Data: https://drive.google.com/file/d/1dKLFJj4HKse7MqEtYVU0YLE5qSiB6orz/view?usp=sharing  <br/>

Trained Model: https://drive.google.com/file/d/1kVutbrtw-dxn3TZ8wYUJHFHSoi7A1KTu/view?usp=sharing  <br/>


How to start training:<br/>
-> train.py contains CFG dictionary where the necessary changes can be made such as data location, learning rate, where to save model etc.<br/>
-> After all the changes are made run: python train.py<br/>
-> After the training is completed evaluation will be shown, loss and acccuracy graph would be plotted, also the classification report and confusion matrix would be displayed.<br/>

To run the model on sample of 100 images:<br/>
python predict_sample.py --data [data path] --model [model checkpoints path]<br/>
e.g: python predict_sample.py --data ../data/ --model model/model.pth

