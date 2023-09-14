import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("creditcard.csv")
df.head()

df.info()

df.shape

df.columns

df.isnull().sum()

# there are no missing values in the dataset
# now splitting the dataset into input and target variables
x = df.drop('Class',axis=1)
y = df['Class']

x.shape

y.shape

# splitting data into train and test data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

# i am using logistic regression to predict the fraud detection
model = LogisticRegression()

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

y_pred

calulating accuracy of our model and confusion matrix

score = accuracy_score(y_test,y_pred)*100
print(" accuracy of the model : ",score)

pscore = precision_score(y_test,y_pred)*100
rscore = recall_score(y_test,y_pred)*100
fscore = f1_score(y_test,y_pred)*100
print(" precision score : ",pscore)
print(" recall score : ",rscore)
print(" f1 score : ",fscore)

cm = confusion_matrix(y_test,y_pred)
cm

CONCLUSION : 

- in above i built a model which used logistic regression which predi
- the accuracy score of the model is 99%
- the precision score of the model is 64.5%
- the recall score of the model is 67.3%
- the f1 score of the model is 65.9%
- it has predicted very accuracte that the transaction is fraud or not 
