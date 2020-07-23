import csv
import numpy as np
import tensorflow as tf
from google.colab import files
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import itertools
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score


#UPLOAD DATASET FROM STORED CSV FILE. I had to download the file from the UCI ML website in my Notepad and then upload to excel. 
uploaded=files.upload()

#Visualize the Dataframe
data = pd.read_csv('Acute Inflammations Data Set.csv')
data.head()

#As you can see above, the Temperature columns values have commas instead of periods. This can easily be fixed by using the .str.replace method from Pandas.
#Removing the commas from the 'Temp' column
data['Temp'] = data.Temp.str.replace(',', '.')
#Applying the strip function to get rid of any ending whitespace characters that may have appeared
data['Temp'] = data['Temp'].apply(lambda x: x.strip())
data.head()


#Change No's and Yes's in all the columns (except temperature) to 1's and 0's:

for i in range(data.shape[0]):

  a = data.iloc[i, 1]
  b = data.iloc[i, 2]
  c = data.iloc[i, 3]
  d = data.iloc[i, 4]
  e = data.iloc[i, 5]
  f = data.iloc[i, 6]
  g = data.iloc[i, 7]

  if a == 'no':
    data.iloc[i, 1] = 0
  else:
    data.iloc[i, 1] = 1
  if b == 'no':
    data.iloc[i, 2] = 0
  else:
    data.iloc[i, 2] = 1
  if c == 'no':
    data.iloc[i, 3] = 0
  else:
    data.iloc[i, 3] = 1
  if d == 'no':
    data.iloc[i, 4] = 0
  else:
    data.iloc[i, 4] = 1
  if e == 'no':
    data.iloc[i, 5] = 0
  else:
    data.iloc[i, 5] = 1
  if f == 'no':
    data.iloc[i, 6] = 0
  else:
    data.iloc[i, 6] = 1
  if g == 'no':
    data.iloc[i, 7] = 0
   else:
    data.iloc[i, 7] = 1

#Visualize to make sure changes occurred correctly
data.head()

#Lets do one set of training on the data without doing K-fold in order to do visualization of the precision and recall. 
X = data.drop(['Nephritis of Renal Pelvis'], axis = 1).values
X = StandardScaler().fit(X).transform(X.astype(float))
X = np.asanyarray(X)
y = data.loc[:,['Nephritis of Renal Pelvis']].values
y = np.asanyarray(y)

#Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
y_train = y_train.squeeze()
y_test = y_test.squeeze()
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

#Instantiate the SVM, with kernel rbf (There are other kernel options but you will see that this one works just fine)
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train) 

#Test model on the test set
yhat = clf.predict(X_test)


#Plot a confusion matrix to visualize accuracy which is 100% in this case.
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)

print(classification_report(y_test, yhat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Urinary Inflammation(1)','No Inflammation(0)'],normalize= False,  title='Confusion matrix')

#Print Accuracy Scores
print('Jaccard Similarity: ', jaccard_score(y_test, yhat))
print('F1_score: ', f1_score(y_test, yhat, average='weighted')) 

#Visualize the correct predictions in a dataframe
test = pd.DataFrame(data = yhat , columns = ['Test Predictions'])
test['Ground Truth Labels'] = y_test
test.head()

#NOTE: For the sake of space, this same test can be ran for the Nephritis column. All that needs to be done, is that on line 80, just drop the Inflammation column instead.

#Lets first test on Inflammation prediction using K-fold cross validation
X = data.drop(['Inflammation Decision'], axis = 1).values
X = StandardScaler().fit(X).transform(X.astype(float))
X = np.asanyarray(X)
y = data.loc[:,['Inflammation Decision']].values
y = np.asanyarray(y)


cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
clf = svm.SVC(kernel='rbf') #Instantiation is redundant in this case but it helps me keep things organized.
cross_val_score(clf, X, y.squeeze(), cv=cv)
#EXPECTED OUTPUT: array([1., 1., 1., 1., 1.]) indicating that accuracy was 100% for all splits of the data


#Next lets test on the Nephritis using K-fold:
X = data.drop(['Nephritis of Renal Pelvis'], axis = 1).values
X = StandardScaler().fit(X).transform(X.astype(float))
X = np.asanyarray(X)
y = data.loc[:,['Nephritis of Renal Pelvis']].values
y = np.asanyarray(y)


cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
clf = svm.SVC(kernel='rbf') #Instantiation is redundant in this case but it helps me keep things organized.
cross_val_score(clf, X, y.squeeze(), cv=cv)
#EXPECTED OUTPUT: array([1., 1., 1., 1., 1.]) indicating that accuracy was 100% for all splits of the data



