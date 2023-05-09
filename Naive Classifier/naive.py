import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
# Read data from CSV file
msg = pd.read_csv('naivetext1.csv')
print('The dimensions of the dataset:', msg.shape)

# Create labelnum column from label
msg['labelnum'] = msg['label'].map({'pos': 1, 'neg': 0})
X=msg.message
y=msg.labelnum
print(X)
print(y)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(msg['message'], msg['labelnum'], random_state=1)
print('The shape of ytest:', y_test.shape)
print('The shape of xtrain:', X_train.shape)
print('The shape of xtest:', X_test.shape)
print('The shape of ytrain:', y_train.shape)

# Create count vectorizer and transform training data
vect = CountVectorizer()
X_train_dtm = vect.fit_transform(X_train)
print('Number of features extracted from messages:', len(vect.get_feature_names()))

# Create DataFrame showing the frequency of each feature in the training dataset
df = pd.DataFrame(X_train_dtm.toarray(), columns=vect.get_feature_names())
print('Frequency of each feature in the training dataset:')
print(df)

# Train Naive Bayes classifier
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)

# Predict on testing data and evaluate performance
X_test_dtm = vect.transform(X_test)
y_pred_class = nb.predict(X_test_dtm)
print('Accuracy:', metrics.accuracy_score(y_test, y_pred_class))
print('Confusion matrix:')
print(metrics.confusion_matrix(y_test, y_pred_class))
print('Recall score:', metrics.recall_score(y_test, y_pred_class))
print('Precision score:', metrics.precision_score(y_test, y_pred_class))
