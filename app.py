import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import emoji
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold,cross_val_score
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score,accuracy_score,classification_report,confusion_matrix,log_loss
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer,LancasterStemmer,WordNetLemmatizer
from wordcloud import WordCloud
import pickle

data=pd.read_csv('fakenews.csv')
fv=data.iloc[:,0]
cv=data.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(fv,cv,test_size=0.2,random_state=1,stratify=cv)

with st.sidebar:
    radio_button=st.radio('MLDLC steps',['Prediction','Problem statement','Simple EDA','Preprocessing','EDA','Model selection'])


if(radio_button=='Problem statement'):
    st.subheader('Problem statement: To classify whether the news article is real or Fake')
    st.write('With the rise of social media and digital news platforms, the dissemination of fake news has become a significant problem. Fake news can spread rapidly, leading to misinformation, social unrest, and erosion of trust in the media. Detecting fake news is crucial for maintaining the integrity of information sources and preventing its harmful effects.The objective of this project is to develop a machine learning model that can accurately classify news articles as either real or fake based on their content and other relevant features.')
    st.write('The dataset is collected from Kaggle website.https://www.kaggle.com/datasets/iamrahulthorat/fakenews-csv?resource=download')
    st.write('The dataset consists of a collection of news articles labeled as either real(0) or fake(1).The "fakenews.csv" dataset contains 4986 unique values that have been labeled as either real or fake news. Each entry in the dataset likely represents a news article or piece of information, along with its classification as either genuine or false. Out of 4986 news articles, 2014 articles belong to Fake news and 2972 articles belong to Genuine news')
    st.write('Based on the problem statement, I have implemented Supervised Machine Learning techniques in Classification.')
    st.write('KNN(Bag of words, Binary Bag of words, TFIDF vectorizer)')
    st.write('Bernoulli Naive bayes using Binary Bag of  words')
    st.write('Multinomial Naive Bayes(Bag of words and TFIDF)')


if(radio_button=='Simple EDA'):
    def eda3(data,column):
        lower=' '.join(data[column]).islower()
        html=data[column].apply(lambda x: True if re.search('<.*?>',x) else False).sum()
        urls=data[column].apply(lambda x: True if re.search('http[s]?://.+?\S+',x) else False).sum()
        hasht=data[column].apply(lambda x: True if re.search('#\S+',x) else False).sum()
        mentions=data[column].apply(lambda x: True if re.search('@\S+',x) else False).sum()
        un_c=data[column].apply(lambda x: True if re.search("[]\.\*'\-#@$%^?~`!&,(0-9)]",x) else False).sum()
        emojiss=data[column].apply(lambda x: True if emoji.emoji_count(x) else False).sum()
        if(lower==False):
            st.write('your data contains lower and upper case')
        if(html>0):
            st.write("Your data contains html tags")
        if(urls>0):
            st.write("Your data contains urls")
        if(hasht>0):
            st.write("Your data contains hashtags")
        if(mentions>0):
            st.write("Your data contains mentions")
        if(un_c):
            st.write("Your data contains unwanted chars")
        if(emojiss):
            st.write("Your data contains emojis")

    eda3(data,'text')


if(radio_button=='Preprocessing'):
    st.image('https://as1.ftcdn.net/v2/jpg/03/06/40/28/1000_F_306402837_UgcBrYZ89K8GXRtA1KjhbNEHwtTdttCD.jpg')
    st.write('The dataset "fakenews.csv" contained mixed case, HTML tags, URLs, hashtags, mentions, unwanted characters, and emojis. To standardize the text for analysis, preprocessing steps included lowercasing, HTML tag, URL, mention, and hashtag removal, elimination of unwanted characters, and conversion of emojis to text. These steps ensured data consistency and prepared the text for analysis tasks like fake news detection and natural language processing.')
    st.write('The dataset underwent additional preprocessing steps where stop words were removed, and each word was lemmatized to its root form. These enhancements further refined the text data, improving the quality of analysis tasks.')

def basic_pp(x,emoj="F"):
    if(emoj=="T"):
        x=emoji.demojize(x)
    x=x.lower()
    x=re.sub('<.*?>',' ',x)
    x=re.sub('http[s]?://.+?\S+',' ',x)
    x=re.sub('#\S+',' ',x)
    x=re.sub('@\S+',' ',x)
    x=re.sub("[]\.\*'’‘_—,:{}\-#@$%^?~`!&(0-9)]",' ',x)
    
    return x

x_train_p=x_train.apply(basic_pp,args=("T"))
x_test_p=x_test.apply(basic_pp,args=('T'))

## Stop words removal

stp=stopwords.words('english')
stp.remove('not')



def stop_words(x):
    sent=[]
    for word in word_tokenize(x):
        if word in stp:
            pass
        else:
            sent.append(word)
    return ' '.join(sent)

x_train_p=x_train_p.apply(stop_words)
x_test_p=x_test_p.apply(stop_words)

def lemmat(x):
    sent=[]
    ls=LancasterStemmer()
    for word in word_tokenize(x):
        sent.append(ls.stem(word))
    return " ".join(sent)

x_train_p=x_train_p.apply(lemmat)
x_test_p=x_test_p.apply(lemmat)


if(radio_button == 'EDA'):
    data1 = pd.DataFrame(x_train_p)
    data1['label'] = y_train
    data2 = data1.loc[data1['label'] == 1, 'text']
    wc = WordCloud(background_color='black', width=1600, height=800).generate(' '.join(data2))
    wc_image_path = 'wordcloud.png'
    wc.to_file(wc_image_path)
    st.image(wc_image_path, caption='Word Cloud for Fake news')

    data2 = data1.loc[data1['label'] == 0, 'text']
    wc = WordCloud(background_color='black', width=1600, height=800).generate(' '.join(data2))
    wc_image_path = 'wordcloud.png'
    wc.to_file(wc_image_path)
    st.image(wc_image_path, caption='Word Cloud for Real News')


if(radio_button=='Model selection'):
    st.write('I employed supervised machine learning algorithms, including KNN and Naive Bayes, to classify news articles as either fake or real. By leveraging these algorithms, I created six models, each utilizing a different vectorizer. This approach allowed for comprehensive exploration of feature representation techniques and their impact on classification performance.')
    ## KNN with Bag of words
    st.subheader('KNN with Bag of words')
    st.write('After preprocessing, I utilized the Bag of Words technique to transform the text data into numerical vectors.')
    st.write('I employed the Stratified K-Fold method with 3 splits and visualized the training F1 score versus the cross-validation F1 score for various values of K. Notably, both F1 scores peaked at K=1.')
    st.write('The chosen final model is the 1-Nearest Neighbors (1NN) classifier, achieving a Generalized F1 score of 0.58.')
    
    ## KNN with BBOW
    st.subheader('KNN with Binary Bag of words')
    st.write('Later, I employed the Binary Bag of Words method to encode the preprocessed text data into numerical vectors.')
    st.write('Applying the Stratified K-Fold technique with 3 splits, I visualized the training F1 score against the cross-validation F1 score for various values of K. Notably, both F1 scores peaked when K=1.')
    st.write('The selected final model is the 1-Nearest Neighbors (1NN) classifier, achieving a Generalized F1 score of 0.52.')

    ## KNN with TFIDF
    st.subheader('KNN with TFIDF')
    st.write('Subsequently, I applied the TF-IDF vectorizer to transform the text data into numerical vectors.')
    st.write('Employing the Stratified K-Fold technique with 3 splits, I visualized the training F1 score compared to the cross-validation F1 score across different values of K. Notably, both F1 scores peaked when K=1.')
    st.write('The chosen final model remains the 1-Nearest Neighbors (1NN) classifier, achieving a Generalized F1 score of 0.671.')

    ## Bernoulli Naive Bayes
    st.subheader('Bernoulli Naive Bayes')
    st.write('I transitioned to Bernoulli Naive Bayes as the algorithm of choice, utilizing the Binary Bag of Words vectorizer.')
    st.write('Using the cross-validation score method with 5 folds, I explored various values of alpha and plotted the cross-validation F1 scores. The optimal alpha value, resulting in the highest F1 score, was identified as 1')
    st.write('Ultimately, the final model, configured with alpha=1, achieved a generalized F1 score of 0.512 on the test data')

    ## Multinomial Naive Bayes using Bag of words
    st.subheader('Multinomial Naive Bayes with Bag of words')
    st.write('I transitioned to Multinomial Naive Bayes as the chosen algorithm, employing the Bag of Words vectorizer.')
    st.write('Utilizing the cross-validation score method with 5 folds, I explored various values of alpha and plotted the cross-validation F1 scores. The optimal alpha value, resulting in the highest F1 score, was identified as 1.')
    st.write('Ultimately, the final model, configured with alpha=1, achieved a generalized F1 score of 0.670 on the test data.')

    ## Multinomial Naive Bayes with TFIDF Vectorizer
    st.subheader('Multinomial using TFIDF Vectorizer')
    st.write('I transitioned to using the TF-IDF vectorizer for feature extraction.')
    st.write('Employing the cross-validation score method with 5 folds, I explored various values of alpha and plotted the cross-validation F1 scores. The optimal alpha value, resulting in the highest F1 score, was identified as 1.')
    st.write('Ultimately, the final model, configured with alpha=1, achieved a generalized F1 score of 0.444 on the test data.')

    st.subheader("Selecting the best model")
    st.write("Among the six models developed using different algorithms and vectorizers, KNN with TF-IDF vectorizer stands out as the most effective in classifying news articles as fake or real.")

model=pickle.load(open('KNNtfidf.pkl','rb'))
vect=pickle.load(open('vectorizer.pkl",'rb'))


if(radio_button=='Prediction'):
     text=st.text_input('Enter the news') 
     submit_button=st.button('Predict')

     if(text and submit_button):
         def predict_news(x,vect,model):
            preprocessed_data=basic_pp(x,emoj='T')
            preprocessed_data=stop_words(preprocessed_data)
            preprocessed_data=lemmat(preprocessed_data)
            preprocessed_data=[preprocessed_data]
            preprocessed_data=vect.transform(preprocessed_data)
            prediction=model.predict(preprocessed_data)[0]
            return prediction
         
         output=predict_news(text,vect,model)
         if(output==1):
            st.write('The news is fake')
         elif(output==0):
            st.write("The news is real")
         
