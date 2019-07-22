
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

df = "" #load data into dataframe

## extracting the target variable from df and assigning it to variable 'y'
y = df['label']

#split the dataset into train and test with test size of 33%
X_train,X_test,Y_train,Y_test = train_test_split(df['plot'],y,
												test_size = 0.33,
												random_state = 53)

#extracting features from test data by using bag of words method and excluding stopwords 
count_vectorizer = CountVectorizer(stop_words='english')

count_train = count_vectorizer.fit_transform(X_train.values)

count_test = count_vectorizer.transform(X_test.values)

# Import naivebayes classifier and metrics from sklearn
from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics

nb_classifier = MultinomialNB()

#fit the nb_classifier model on train data
nb_classifier.fit(count_train,Y_train)

#Use test data to predict the values using the model built in previous step
pred = nb_classifier.predict(count_test)

#check the accuracy of the model
metrics.accuracy(Y_test,pred)

metrics.confusion_matrix(Y_test,pred,labels=[0,1])

