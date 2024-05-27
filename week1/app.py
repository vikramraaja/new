import pandas as pd
import streamlit as st
import requests
import zipfile
import io
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Title and introduction
st.title("Naive Bayes Classifier for IMDb Review Classification")
st.write("This app uses a Naive Bayes classifier to predict whether an IMDb review is positive or negative.")

# URL for the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"

# Function to download and extract data
@st.cache_data
def download_and_extract_data(url):
    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall()
        with open(z.namelist()[0], 'r') as file:
            df = pd.read_csv(file, sep='\t', names=["sentiment", "text"])
    return df

# Download and extract the dataset
df = download_and_extract_data(url)

# Display the first five rows of the dataset
st.subheader("First Five Rows of the Dataset")
st.write(df.head())

# Dataset information
st.subheader("Dataset Information")
buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

# Check for missing values
st.subheader("Missing Values")
st.write(df.isna().sum())

# Map sentiment to numerical values
df['snum'] = df.sentiment.map({'ham': 1, 'spam': 0})

# Split the dataset into features and target variable
x = df['text']
y = df['snum']

# Split the data into training and test sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)

# Initialize the CountVectorizer
cv = CountVectorizer()
xtrain_dtm = cv.fit_transform(xtrain)
xtest_dtm = cv.transform(xtest)

# Initialize the MultinomialNB classifier
clf = MultinomialNB().fit(xtrain_dtm, ytrain)

# Predict the test set results
predicted = clf.predict(xtest_dtm)

# Display the confusion matrix
st.subheader("Confusion Matrix")
cm = metrics.confusion_matrix(ytest, predicted)
st.write(cm)

# Display the accuracy score
st.subheader("Accuracy Score")
accuracy = metrics.accuracy_score(ytest, predicted)
st.write(f"Accuracy: {accuracy:.2f}")

# Display precision and recall
st.subheader("Precision and Recall")
precision = metrics.precision_score(ytest, predicted, average='micro')
recall = metrics.recall_score(ytest, predicted, average='micro')
st.write(f"Precision: {precision:.2f}")
st.write(f"Recall: {recall:.2f}")
