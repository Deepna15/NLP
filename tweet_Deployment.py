# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 14:04:19 2023

@author: Chandrika
"""

import pandas as pd
import streamlit as st
import re
import pickle


st.header('Tweet Semantic Analysis')

raw_text=st.text_area("Enter text here")


def clean(tweets):
    tweets=tweets.lower()
    tweets=re.sub(r'[0-9]',"",str(tweets)) 
    tweets=re.sub(r'#',"",str(tweets))
    tweets=re.sub(r"@[\w]*","",tweets) 
    tweets=re.sub(r"http\S+","",tweets) 
    tweets=re.sub(r'[^\w\s]',"",tweets)  
    return tweets

input=[clean(raw_text)]



tfidf=pickle.load(open('tfidf_vectors.pkl','rb'))
model=pickle.load(open('tweet_analysis.pkl','rb'))

if st.button('Predict'):
    x=tfidf.transform((input))
    pred=model.predict(x)
    if (int(pred)==0):
        st.write("The tweet is classified as FIGURATIVE class")
    if (int(pred)==1):
        st.write("The tweet is classified as IRONY class")
    if (int(pred)==2):
        st.write("The tweet is classified as REGULAR class")
    if (int(pred)==3):
        st.write("The tweet is classified as SARCASM class")