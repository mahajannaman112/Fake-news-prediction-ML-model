# id = unique id for a news article
# title = the title of the news article
# text = the text of the news article
# label = the label of the news article (FAKE or REAL)
# author = the author of the news article

# 1 : Fake news
# 0 : Real news

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# import nltk
# nltk.download('stopwords')
# print(stopwords.words('english'))


#Data Pre processing

#Load dataset to pandas dataframe
news_dataset = pd.read_csv( r"c:\Users\acer.DESKTOP-SAIO2RF\Desktop\Machine Learning\projects\Fake news pridiction\fake_news.csv")
# print(news_dataset.shape)

# print the first 5 rows of the dataframe
# print(news_dataset.head())

# counting the number of missing values in the dataset
news_dataset.isnull().sum()

#replacing the missing values with nul string
news_dataset = news_dataset.fillna('')

#merging the author name and news title
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']

print(news_dataset['content'])


# seprating the data and label
X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']

print (X)
print(Y)

#Stemming : stemming is the process of reducing a word to its root word
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_dataset['content'] = news_dataset['content'].apply(stemming)
print(news_dataset['content'])

#seprating the data and label
X = news_dataset['content'].values
Y = news_dataset['label'].values

print(X)
print(Y)

#converting the textual data to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)
print(X)

#splitting the dataset into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2) 
print(X.shape, X_train.shape, X_test.shape)

#training model : Logistic regression model
model = LogisticRegression()
model.fit(X_train, Y_train)
print(model.fit(X_train, Y_train))

#evaluation accuracy score for training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of training data : ', training_data_accuracy)

#evaluation accuracy score for testing data
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)  
print('Accuracy score of testing data : ', testing_data_accuracy)

#Making a predictive system
X_new = X_test[0]

pridiction = model.predict(X_new)
print(pridiction)

if (pridiction[0]==0):
    print('The news is Real')   
else:
    print('The news is Fake')