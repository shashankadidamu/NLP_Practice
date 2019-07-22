
#steps: 
#1) extracting features using both bagofwords and tfidf 
#2) train naivebayes classifier on the extracted features
#3) test the accuracy of both the approaches

# Import the necessary modules
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

#read dataset
df = pd.read_csv("/data/fake_or_real_news.csv")

# Print the head of df
print(df.head())

# Create a series to store the labels: y
y = df.label

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)

# Initialize a CountVectorizer object: count_vectorizer
count_vectorizer = CountVectorizer(stop_words='english')

# Transform the training data using only the 'text' column values: count_train 
count_train = count_vectorizer.fit_transform(X_train)

# Transform the test data using only the 'text' column values: count_test 
count_test = count_vectorizer.transform(X_test)

# Print the first 10 features of the count_vectorizer
print(count_vectorizer.get_feature_names()[:10])


###Using Tfidf vectorizer
# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize a TfidfVectorizer object: tfidf_vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Transform the training data: tfidf_train 
tfidf_train = tfidf_vectorizer.fit_transform(X_train)

# Transform the test data: tfidf_test 
tfidf_test = tfidf_vectorizer.transform(X_test)

# Print the first 10 features
print(tfidf_vectorizer.get_feature_names()[:10])

# Print the first 5 vectors of the tfidf training data
print(tfidf_train.A[:5])

###Inspecting the vectors

# Create the CountVectorizer DataFrame: count_df
count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())

# Create the TfidfVectorizer DataFrame: tfidf_df
tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())

# Print the head of count_df
print(count_df.head())

# Print the head of tfidf_df
print(tfidf_df.head())

# Calculate the difference in columns: difference
difference = set(count_df.columns) - set(tfidf_df.columns)
print(difference)

# Check whether the DataFrames are equal
print(count_df.equals(tfidf_df))

#import naivesbayes and metrics modules from sklearn
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB

nb_classifier = MultinomialNB()

#train the naive bayes model on features present in count_train
nb_classifier.train(count_train, y_train)

#predict 
pred = nb_classifier.predict(count_test)

#check accuracy
score = metrics.accuracy_score(y_test,pred)
print(score)

#confusion matrix
cm = metrics.confusionmatrix(y_test,pred,labels=['FAKE','REAL'])
print(cm)


##naive bayes classifier using features created by tdidf vectorizer
nb_classifier2 = MultinomialNB()

nb_classifier2.train(tfidf_train,y_train)

pred2 = nb_classifier2.predict(tfidf_test)

#check accuracy
score2 = metrics.accuracy_score(y_test,pred2)
print(score2)

#confusion matrix
cm = metrics.confusionmatrix(y_test,pred2,labels=['FAKE','REAL'])
print(cm)


##test a few different alpha levels using the Tfidf vectors to determine if there is a better performing combination
alphas = np.arange(0, 1, 0.1)

#define train_and_predict that takes alpha value as an argument and returns accuracy score
def train_and_predict(alpha):
	nb_classifier = MultinomialNB(alpha=alpha)

	#fit
	nb_classifier.fit(tfidf_train,y_train)

	#predict
	nb_classifier.predict(tfidf_test)

	score = metrics.accuracy_score(y_test,pred)
	return score

#itereate over alphas
for alpha in alphas:
	print('Alpha:', alpha)
	print('Score:', train_and_predict(alpha))
	print()

##inspecting model
# Get the class labels: class_labels
class_labels = nb_classifier.classes_

# Extract the features: feature_names
feature_names = tfidf_vectorizer.get_feature_names()

# Zip the feature names together with the coefficient array and sort by weights: feat_with_weights
feat_with_weights = sorted(zip(nb_classifier.coef_[0], feature_names))

# Print the first class label and the top 20 feat_with_weights entries
print(class_labels[0], feat_with_weights[:20])

# Print the second class label and the bottom 20 feat_with_weights entries
print(class_labels[1], feat_with_weights[-20:])







