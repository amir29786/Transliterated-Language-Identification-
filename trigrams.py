#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 11:13:34 2017

@author: anas
"""

from random import shuffle
from nltk import ngrams
import nltk
def split_line(text):
    words=text.split()
    return words

def generate_features(unionlist,word):
    feature_vector = numpy.zeros(shape=(1,len(unionlist)))
    i=0
    wordlen=len(word)
    j=0
    count=3
    individual=""
    while j<=(wordlen-count):
        k=0
        individual=""
        while k<count:
            individual+=word[j+k]
            k+=1
        t=0
        union_len=len(unionlist)
        while t<union_len:
            if unionlist[t]==individual:
                feature_vector[0][t]+=1
            t+=1
        j+=1
    return feature_vector

from nltk.corpus import words

eng= nltk.corpus.words.words()

i=0
for x in eng:
    eng[i]=eng[i].lower()
    i+=1
f=open("Hindi Word Transliteration Pairs.txt","r+")
text=f.read()
words=split_line(text)
english=[]
hindi=[]
i=0
number=len(words)
while i<number:
    if(i%2==0):
        english.append(words[i])
    else:
       hindi.append(words[i])
    i+=1
i=0

bigrams_hindi=[]
number_english=len(english)

training_english=4*number_english/5
while i<training_english:
    word=english[i]
    wordlen=len(word)
    j=0
    count=3
    individual=""
    while j<=(wordlen-count):
        k=0
        individual=""
        while k<count:
            individual+=word[j+k]
            k+=1
        bigrams_hindi.append(individual)
        j+=1
        
    i+=1
no_hindi= len(bigrams_hindi)
hindi_bigrams= list(set(bigrams_hindi))

hindi_len=len(hindi_bigrams)








import numpy 
freq_hindi = numpy.zeros(shape=(26*26*26,1))
i=0
while i<no_hindi:
    freq_hindi[((abs(ord(bigrams_hindi[i][0])-97))*26*26+abs(ord(bigrams_hindi[i][1])-97)*26+abs(ord(bigrams_hindi[i][2])-97))%(26*26*26)]+=1
    i=i+1
    

i=0
j=0
f.close()
i=0
length_english=len(eng)
english_words=[]
while i<length_english:
    english_words.append(eng[i])
    i+=7

length_english=len(english_words)
shuffle(english_words)
bigrams_eng=[]

i=0
length_english_train=4*len(english_words)/5
while i<length_english_train:
    word=english_words[i]
    wordlen=len(word)
    j=0
    count=3
    individual=""
    while j<=(wordlen-count):
        k=0
        individual=""
        while k<count:
            individual+=word[j+k]
            k+=1
        bigrams_eng.append(individual)
        j+=1
        
    i+=1
no_eng= len(bigrams_eng)

eng_bigrams= list(set(bigrams_eng))

eng_len=len(eng_bigrams)

print (eng_bigrams)

freq_eng = numpy.zeros(shape=(26*26,1))

i=0

#freq_eng[((abs(ord(bigrams_eng[i][0])-97))*26*26+abs(ord(bigrams_eng[i][1])-97)*26+abs(ord(bigrams_eng[i][2])-97))%(26*26*26)]+=1
    
freq_eng = numpy.zeros(shape=(26*26*26,1))
i=0

while i<no_eng:
    freq_eng[((abs(ord(bigrams_eng[i][0])-97))*26*26+abs(ord(bigrams_eng[i][1])-97)*26+abs(ord(bigrams_eng[i][2])-97))%(26*26*26)]+=1
    i=i+1

i=0
j=0
i=0
while i<26*26*26:
    print (freq_eng[i],freq_hindi[i])
    i=i+1

print (no_eng,no_hindi)

bigrams_english_final=[]
bigrams_hindi_final=[]
first='a'
second='a'
third='a'
#first=chr(ord(first)+1)
temp=first+second+third
#print (temp)
#list to store bigrams according to frequency
i=0

first='a'
second='a'
third='a'
i=0
tempp=first+second+third
while i<26*26*26:
    temp=first+second+third
    bigrams_english_final.append(temp)
    bigrams_hindi_final.append(temp)
    third=chr(ord(third)+1)
    if third==chr(ord('z')+1):
        second=chr(ord(second)+1)
        third='a'
        if second==chr(ord('z')+1):
            first=chr(ord(first)+1)
            second='a'
    i+=1
englishlist=[]
hindilist=[]
i=0
while i<26*26*26:
    englishlist.append(freq_eng[i][0])
    hindilist.append(freq_hindi[i][0])
    i+=1
i=0
k=0

#now removing those whose frequency is less than 5
print ('done')
removed_english=[]
removed_english_freq=[]
while i<26*26*26:
    if englishlist[i]>=20:
        removed_english_freq.append(englishlist[i])
        removed_english.append(bigrams_english_final[i])
    i+=1

i=0
removed_hindi=[]
removed_hindi_freq=[]
while i<26*26*26:
    if hindilist[i]>=20:
        removed_hindi_freq.append(hindilist[i])
        removed_hindi.append(bigrams_hindi_final[i])
    i+=1
unionlist=[]
i=0
while i<len(removed_hindi):
    unionlist.append(removed_hindi[i])
    i+=1
i=0
while i<len(removed_english):
    unionlist.append(removed_english[i])
    i+=1
    
unionlist= list(set(unionlist))
"""
p=generate_features(unionlist,'kutta')
dataset=numpy.zeros(shape=(int(4*(len(english_words)+len(english))/5),len(unionlist)))
x_train=numpy.zeros(shape=(int(4*int(4*(len(english_words)+len(english))/5)/5+5),len(unionlist)))
y_train=numpy.zeros(shape=(int(4*int(4*(len(english_words)+len(english))/5)/5+5)))
x_test=numpy.zeros(shape=(int(1*int(4*(len(english_words)+len(english))/5)/5+5),len(unionlist)))
y_test=numpy.zeros(shape=(int(1*int(4*(len(english_words)+len(english))/5)/5+5)))
i=0
#this is for english
length=int(4*(length_english)/5)
k=0
m=0
while i<length:
    dataset[i]=generate_features(unionlist,english_words[i])
    if i<int((4*length)/5):
        x_train[k]=dataset[i]
        y_train[k]=1
        k=k+1
    else:
        x_test[m]=dataset[i]
        y_test[m]=1
        m=m+1
        
    i+=1
j=0
#this is for transliterated
length=int(4*(number_english)/5)
while j<length:
    dataset[i]=generate_features(unionlist,english[j])
    if j<int((4*length)/5):
        x_train[k]=dataset[i]
        y_train[k]=0
        k=k+1
    else:
        x_test[m]=dataset[i]
        y_test[m]=0
        m=m+1    
    j+=1
    i+=1
size=int(4*(len(english_words)+len(english))/5)
result=numpy.zeros(shape=(size,1))
i=0
#1 is for english
#0 is for hindi
while i<int(4*(length_english)/5):
    result[i]=1
    i+=1
j=0
while j<int(4*(number_english)/5):
    result[i]=0
    i+=1
    j+=1

#from sklearn.preprocessing import train_test_split
#x_train,x_test,y_train,y_test=train_test_split(dataset,result,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
"""
"""
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
"""

"""
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
clf = SVC()
clf.fit(x_train, y_train) 
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
print accuracy_score(y_test,clf.predict(x_test))
"""
