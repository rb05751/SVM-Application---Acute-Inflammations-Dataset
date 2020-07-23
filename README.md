# SVM-Application---Acute-Inflammations-Dataset

This project is an application of a Support Vector Machine for Binary Classification on the Acute Inflammations Dataset available on the UCI Machine Learning Repository website (link: https://archive.ics.uci.edu/ml/datasets/Acute+Inflammations), created by Jacek Czerniak, Ph.D., Assistant Professor at Systems Research Institute. In order to preprocess the data there are a few steps needed if one is not familiar with sourcing data from UCI:

1) On the page of the ACI data, click on 'Data Folder', then click on the 'diagnosis.data' folder. This will either download directly into Notepad as a .txt file or you will have to go to your downloads folder, right click on the file, select open with notepad and go from there. 
2) Once you have it downloaded in NotePad, save it as a .txt file.
3) Open up Excel, go your Data tab and then click on the upload data button, make sure to upload from TXT to CSV. Here is a link to help: https://support.geekseller.com/knowledgebase/convert-txt-file-csv/
4) The uploaded=files.upload() line in the beginning will prompt you upload file from your desktop.

The learning experience here was preprocessing the dataset so that the commas were extracted from the Temperature column and the no's and yes's were turned into 0's and 1's respectively in order to be fed to the model. The Train/Test split was an 80/20 and the SVM classifier achieved 100% accuracy on the test set and achieved 100% accuracy even when doing K-fold cross validation. Note that this was achieved on both target variables.

The user can refer to the UCI website for a description of what the features mean and the targets as well. 
