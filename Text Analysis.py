#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 14:50:24 2021

@author: abhishekmishra
"""


# importing required libraries

import numpy as np 
import pandas as pd 
import re
import requests
from bs4 import BeautifulSoup 
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize

import matplotlib.pyplot as plt
import seaborn as sns



import warnings
warnings.filterwarnings("ignore")


# reading the cik_list file
cik_list = pd.read_excel("cik_list.xlsx")
cik_list

# adding url prefix to the SECFNAME
url_prefix = "https://www.sec.gov/Archives/"
cik_list.SECFNAME = url_prefix + cik_list.SECFNAME
cik_list


# reading the Master Dictionary file
mast_dict = pd.read_excel("LoughranMcDonald_MasterDictionary_2018.xlsx")
mast_dict

# positive and negative words
positive_word=[word for word in mast_dict[mast_dict['Positive']!=0]['Word']]
negative_word=[word for word in mast_dict[mast_dict['Negative']!=0]['Word']]



# stop words
stop = open("StopWords_Generic.txt","r")
stop_words = stop.read().lower()
stop_words = stop_words.split("\n")


# uncertainity words
uncertainity_word = pd.read_excel("uncertainty_dictionary.xlsx")
uncertainity_word = list(uncertainity_word['Word'])



# constrainting words
constrainting_word = pd.read_excel("constraining_dictionary.xlsx")
constrainting_word = list(constrainting_word['Word'])


# now let's fetch the reports from the links from the cik file
links = [i for i in cik_list.SECFNAME]

#extracting the financial reports using beautiful soup
financial_reports=[]
for i in links:
    page=requests.get(i)
    data=BeautifulSoup(page.text, "html.parser")
    financial_reports.append(data.text)

# intialising lemmatizer for text cleaning
wordnet=WordNetLemmatizer()

# defining a function to clean the text and fetching the text data from the files
def clean_text(text):
    # tokenizing into sentences
    sent = sent_tokenize(text)
    num_sent = len(sent)
    
    # tokeninsing to words and fetching text using regular expression
    word = re.sub(r"[^A-Za-z]"," ",text.lower())
    words = word_tokenize(word)
    
    # cleaning the words by lemmatising and removing the stop words
    cleaned_words = [wordnet.lemmatize(i) for i in words if i not in stop_words]
    num_words = len(cleaned_words)
    cleaned_text = ' '.join(cleaned_words)
    return num_words,num_sent,cleaned_words,cleaned_text


# function to count the words from master dictionary and extracted data
def count_word(exst_word,new_word):
    exst_word = [i.lower() for i in exst_word]
    count = 0
    for i in new_word:
        if(i in exst_word):
            count+=1
    return count

# function to determine polarity score
def polarity_score(positive, negative):
    polarity=(positive - negative)/((positive + negative)+ 0.000001)
    return polarity

#function to determine subjectivity score
def subjectivity_score(positive,negative,total_words):
    subjectivity= (positive + negative)/ ((total_words) + 0.000001)
    return subjectivity



# function for sentiment score categorization of the report
def sentiment_score_categorization(polarity):
    if(polarity<-0.5):
        return "most negative"
    elif(polarity>=-0.5 and polarity<0):
        return "negative"
    elif(polarity==0):
        return "neutral"
    elif(polarity>0 and polarity<0.5):
        return "positive"
    else:
        return "very positive"
    
    
    
# function to count number of syllables to get differentiate between complex and non complex words
def num_syllables(words):
    num=0
    for word in words:
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count += 1
        if count>2:
            num+=1
    return num



# creating a new dataframe to store the variables 
df = pd.DataFrame()

# creating some empty lists for the variables required
pos_sc =[]
neg_sc =[]
pol_sc =[]
avg_sent_len=[]
perc_of_comp_words=[]
fog_ind=[]
com_word_ct =[]
word_ct=[]
unce_sc=[]
const_sc=[]
pos_word_prop=[]
neg_word_prop=[]
unce_word_prop=[]
const_word_prop=[]
const_words_whole_report =[]

# running a loop on financial reports
for i in range(len(financial_reports)):
    
    # getting cleaned words
    words = clean_text(financial_reports[i])[2]
    
    # calculating number of sentences
    num_sentences = clean_text(financial_reports[i])[1]
    
    # calculating number of words
    num_words = clean_text(financial_reports[i])[0]
    word_ct.append(num_words)
    
    # calculating positive scores
    positive_score = count_word(positive_word,words)
    pos_sc.append(positive_score)
    
    # calculating negative scores
    negative_score = count_word(negative_word,words)
    neg_sc.append(negative_score)
    
    # calculating polarity score
    polarity = polarity_score(positive_score,negative_score)
    pol_sc.append(polarity)
    
    #calculating subjectivity score
    subjectivity = subjectivity_score(positive_score,negative_score,num_words)
    
    #calculating average sentence length
    average_sentence_length = (num_words/num_sentences)
    avg_sent_len.append(average_sentence_length)
    
    # getting count of complex words
    num_complex_words = num_syllables(words)
    com_word_ct.append(num_complex_words)
    
    # getting percentage of complex words
    percentage_complex_words = (num_complex_words/num_sentences)
    perc_of_comp_words.append(percentage_complex_words)
    
    # calculating fog index
    fog_index = 0.4*(average_sentence_length + percentage_complex_words)
    fog_ind.append(fog_index)
    
    # calculating uncertainity score
    uncertainty_score = count_word(uncertainity_word,words)
    unce_sc.append(uncertainty_score)
    
    # calculating contraining score
    constraining_score = count_word(constrainting_word,words)
    const_sc.append(constraining_score)
    
    # checking positive word proportion
    positive_word_proportion = (positive_score/num_words)
    pos_word_prop.append(positive_word_proportion)
    
    # checking negative word proportion
    negative_word_proportion = (positive_score/num_words)
    neg_word_prop.append(negative_word_proportion)
    
    # checking uncertainty word proportion
    uncertainty_word_proportion = (uncertainty_score/num_words)
    unce_word_prop.append(uncertainty_word_proportion)
    
    # checking constraining word proportion
    constraining_word_proportion = (constraining_score/num_words)
    const_word_prop.append(constraining_word_proportion)
    
    # storing the variales in datframe
df['positive_score'] = pos_sc
df['negative_score'] = neg_sc
df['polarity_score'] = pol_sc
df['average_sentence_length'] = avg_sent_len
df['percentage_of_complex_words'] = perc_of_comp_words
df['fog_index'] = fog_ind
df['complex_word_count'] = com_word_ct
df['word_count'] = word_ct
df['uncertainty_score'] = unce_sc
df['constraining_score'] = const_sc
df['positive_word_proportion'] = pos_word_prop
df['negative_word_proportion'] = neg_word_prop
df['uncertainty_word_proportion'] = unce_word_prop
df['constraining_word_proportion'] = const_word_prop


# concatinating the dataframes to get output data with all the variables
output_df = pd.concat([cik_list,df],axis=1)
output_df.to_excel('output.xlsx')



## Visualisation and Analysis


# Distribution of the variables
cols = ['positive_score',
       'negative_score', 'polarity_score', 'average_sentence_length',
       'percentage_of_complex_words', 'fog_index', 'complex_word_count',
       'word_count', 'uncertainty_score', 'constraining_score',
       'positive_word_proportion', 'negative_word_proportion',
       'uncertainty_word_proportion', 'constraining_word_proportion']
plt.figure(figsize=(25,25))
plot = 1
for var in cols:
    plt.subplot(5,3,plot)
    sns.distplot(df[var],color='lime')
    plot+=1
plt.show()

# Visualisation of FORMS with different derived components

plt.figure(figsize=(25,25))
plot = 1
for i in df.columns:
    plt.subplot(5,3,plot)
    sns.barplot(x='FORM',y=i,data=output_df,palette='prism_r')
    plot+=1
plt.show()

# creating a pivot report of company name and form with the variable

detailed_report = pd.pivot_table(output_df, index = ['CONAME','FORM'], values = ['positive_score',
       'negative_score', 'polarity_score', 'average_sentence_length',
       'percentage_of_complex_words', 'fog_index', 'complex_word_count',
       'word_count', 'uncertainty_score', 'constraining_score',
       'positive_word_proportion', 'negative_word_proportion',
       'uncertainty_word_proportion', 'constraining_word_proportion'])










