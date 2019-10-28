# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 19:46:20 2019
@author: willywanghw
"""
#7108029211 王皓威
import pandas as pd 
from sklearn import preprocessing
#import multiprocessing
#import numpy as np

#資料預處理
bank = pd.read_csv('bank-full.csv',delimiter = ';')
le = preprocessing.LabelEncoder()
#bank['job'] = le.fit_transform(bank['job'])
bank['housing'] = le.fit_transform(bank['housing'])
bank['default'] = le.fit_transform(bank['default'])
bank['loan'] = le.fit_transform(bank['loan'])
Label = bank['y']
Label = le.fit_transform(Label)
bank = bank.drop(['y'],axis=1)
#df = pd.DataFrame(arr, columns = ["name", "num"]) # 指定欄標籤名稱  

#用一般知識進行編碼

#不同職業進行不同的編碼
bank['job'].replace({'entrepreneur':10,'management':7,'technician':5,'blue-collar':4,
    'retired':3,'admin.':5,'services':4,'self-employed':3,'unemployed':2
    ,'housemaid':4,'student':2,'unknown':1},inplace=True)

#結婚設為3  單身與離婚設為1
bank['marital'].replace({'married':5,'single':1,'divorced':1},inplace=True)
#大學設為3 中學2 小學1  未知0
bank['education'].replace({'tertiary':3,'secondary':2,'primary':1,'unknown':0},inplace=True)

bank['poutcome'].replace({'success':5,'other':2,'failure':1,'unknown':0},inplace=True)

#查看缺失值
#print(bank.isnull().any())

#X為要訓練的資料
X = pd.DataFrame(bank,columns = ['age','housing','balance','marital','education'
                                 ,'loan','duration','default','poutcome','pdays','job'])
#X = pd.DataFrame(bank)

#資料標準化
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X = ss.fit_transform(X)


#特徵選取
from sklearn.feature_selection import SelectKBest
selek =  SelectKBest(k=5)
X_selectk = selek.fit_transform(X, Label)
print('age','housing','balance','marital','education','loan','duration','default','poutcome','pdays','job')
print(selek.scores_)
#print(X_selectk)

#資料切割訓練集測試集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_selectk,Label, test_size=0.3, random_state=20)


#模型建立進行預測
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression(solver='liblinear')
logistic.fit(X_train,y_train)
logResult = logistic.score(X_test,y_test)
PredictResult = logistic.predict_proba(X_test)
print('LogisticRegression score : ',logistic.score(X_test,y_test))


#print(multiprocessing.cpu_count())



