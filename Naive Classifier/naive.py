import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Read the dataset
msg = pd.read_csv('naivetext1.csv', names=['message', 'label'])
print('The dimensions of the dataset:', msg.shape)

# Map labels to numerical values
msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})
X = msg.message
y = msg.labelnum

# Handle missing values in y
y = y.fillna(0)  # Replace NaN values with 0 or any other appropriate value

# Split the dataset into train and test data
xtrain, xtest, ytrain, ytest = train_test_split(X, y)
print("ytest shape:", ytest.shape)
print("xtrain shape:", xtrain.shape)
print("xtest shape:", xtest.shape)
print("ytrain shape:", ytrain.shape)

# Feature Extraction to convert text to document term matrix
count_vect = CountVectorizer()
xtrain_dtm = count_vect.fit_transform(xtrain)
xtest_dtm = count_vect.transform(xtest)
print("Feature names:", count_vect.get_feature_names_out())

df = pd.DataFrame(xtrain_dtm.toarray(), columns=count_vect.get_feature_names_out())
print("Dataframe representation:")
print(df)
print("Sparse matrix representation:")
print(xtrain_dtm)

# Training Naive Bayes (NB) classifier on training data
clf = MultinomialNB().fit(xtrain_dtm, ytrain)
predicted = clf.predict(xtest_dtm)

# Printing accuracy metrics
print('Accuracy metrics')
print('Accuracy of the classifier is', metrics.accuracy_score(ytest, predicted))
print('Confusion matrix')
print(metrics.confusion_matrix(ytest, predicted))
print('Recall and Precision')
print(metrics.recall_score(ytest, predicted, zero_division=1))
print(metrics.precision_score(ytest, predicted, zero_division=1))
