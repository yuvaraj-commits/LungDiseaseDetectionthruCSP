This project main intention is to train a model on CXR image dataset to predict x-ray images covid, normal or pneumonia images.
To do that we have used CSPDenseNet architecture with standard mode.
We built binary and multiclassification algorithms using CSPDenseNet architecture.
we collected CXR data from kaggle: https://www.kaggle.com/pranavraikokte/covid19-image-dataset/download. and created kaggle_dataset folder.
EDA.jpnb for exploratory data analysis on datasets.
later we do datapreparation (preprocess and augmentation) step and code can be seen at datapreparation.py 
Here, we read images from kaggle_dataset and preprocessed to 224x224x3. 
we split the dataset and stored in dataset folder.
For split dataset, we created two types of folders 1. binray and 2. multi 
In each folder we have train, val and test folders.
For train dataset we applied augmentation strategy.
Once dataset is avaialble we created cspdensenet architecture which is in csp_densenet.py 
THen we created a wrapper for covid classification in covid_classification.py.
For this model we need to pass directory paths and tuning parameters. It will train the model and return the results.
1. training method is to train the model
2. test method is to evaluate the model
We prepared  binary_classification.jpnb and multi classification.jpnb files.
From this code we can call all the scripts we created like datacollection, datapreparation and model training, prediction.
We deployed best predicted model for classification on heroku using flask framework (https://x-ray-pred.herokuapp.com/). THat code can be seen deploy folder. We can upload x-ray images and can see prediction.
figures folder contain eda and confusion matrix images.
results folder contain train results for each epoch in csv/txt file 
weights folder contain best weights obtained in the model training. https://github.com/veerendrapv/covid_classification_cspdensenet/tree/master/weights
