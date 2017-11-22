# encoding=utf8
import sys
import codecs
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import seaborn as sns
from nltk.corpus import stopwords 
from nltk import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

def pre_process_data(data, remove_stop_words=True):

    list_personality = []
    list_posts = []
    len_data = len(data)
    i=0
    
    for row in data.iterrows():
        i+=1
        if i % 500 == 0:
            print("%s | %s rows" % (i, len_data))

        ##### Remove and clean comments
        posts = row[1].posts
        temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'link', posts)
        temp = re.sub("[^a-zA-Z]", " ", temp)
        temp = re.sub(' +', ' ', temp).lower()
        if remove_stop_words:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in cachedStopWords])
        else:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])

        type_labelized = lab_encoder.transform([row[1].type])[0]
        list_personality.append(type_labelized)
        list_posts.append(temp)

    #del data
    list_posts = np.array(list_posts)
    list_personality = np.array(list_personality)
    return list_posts, list_personality


def extraverted(mbti_type):
    if mbti_type[0]=="E": return "extraverted"
    else: return "introverted"


df=pd.read_csv("mbti_1.csv",sep=",")
print df.head()


###unbalanced distribution

df['extraverted']=df['type'].map(lambda x: extraverted(x))
print df['extraverted'].head()

# Lemmatizer | Stemmatizer
stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()


###preprocessing
#Replace urls with a dummy word: "link"
#Keep only words and put everything lowercase
#Lemmatize each word

unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
       'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
lab_encoder = LabelEncoder().fit(unique_type_list)

cachedStopWords = stopwords.words("english")

list_posts, list_personality = pre_process_data(df, remove_stop_words=True)


###which words are the most common in all posts?
all_posts="".join(list_posts)
words=all_posts.split()

import collections
counter = collections.Counter(words)
most_common_20 = counter.most_common(40)
most_common_20_words=[elem[0] for elem in most_common_20]

list_posts_words=[]
for i in range(0,len(list_posts)):
    word_counts=[list_posts[i].count(word) for word in most_common_20_words]
    list_posts_words += [ [lab_encoder.inverse_transform(list_personality[i])] + word_counts]
    
df_new=pd.DataFrame(list_posts_words, columns=['nbti']+most_common_20_words)

df_new['extraverted']=df_new['nbti'].map(lambda x: extraverted(x))
print df_new.head()

df_new.to_csv("nbti_most_common_40_words.csv",sep=";")
df_new.to_csv("nbti_most_common_40_words_orange.csv",sep=",")


'''
-reducing number intreverted people (balanced)-how to make it balanced depending on different type distributions
- emotional words count
- negative/positive words
- negative words (negations)
- different most common words between E/I
- NLTK most common words in english
- excluding verbs
- zero usage/absence of specific words
- splitting into training and testing 
- URL distributions
- numbers in text
- smilies, emotions (hah)
- topics (beauty, tech ...)
- posting about own personality type
- classification T/F type

EVERYONE works on at least on 2 things
follow up meetup 13th of dec
TO DOs: github share original data and transformed data set

'''