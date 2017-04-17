#!/usr/bin/python
# -*- coding:utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score


mpl.rcParams['font.sans-serif'] = [u'SimHei']  # FangSong/黑体 FangSong/KaiTi
mpl.rcParams['axes.unicode_minus'] = False




def preprocessData(path,is_train):
    df = pd.read_csv(path)
    df['Sex'] = df['Sex'].map({'female':0,'male':1}).astype(int)

    # 起始城市做one-hot编码
    df.loc[(df.Embarked.isnull()),'Embarked']='C'
    embarked_data = pd.get_dummies(df.Embarked)
    # print embarked_data
    embarked_data = embarked_data.rename(columns=lambda x: 'Embarked_' + str(x))
    df = pd.concat([df, embarked_data], axis=1)

    # 增加两列，标志年龄，船票是否被人工填补
    age_autofilled = np.zeros(len(df['PassengerId']))
    fare_autofilled = np.zeros(len(df['PassengerId']))
    for i,r in enumerate(df['Age']):
        if np.isnan(np.float(r)):
            age_autofilled[i]=1
    age_autofilled = pd.DataFrame(age_autofilled)
    age_autofilled.columns = ['Age_autofilled']
    for i,r in enumerate(df['Fare']):
        if np.isnan(np.float(r)):
            fare_autofilled[i]=1
    fare_autofilled = pd.DataFrame(fare_autofilled)
    fare_autofilled.columns = ['Fare_autofilled']
    df = pd.concat([df, fare_autofilled,fare_autofilled], axis=1)

    # 對title進行歸類，分為Matser,Miss,Mr,Mrs和Rare，Rare為其他稀有title
    title = []
    for i,r in enumerate(df['Name']):
        if 'Mr.' in r:
            title.append('Mr')
        elif 'Mrs.' in r:
            title.append('Mrs')
        elif 'Miss.' in r:
            title.append('Miss')
        elif 'Master.' in r:
            title.append('Master')
        else:
            title.append('Rare')

    title = pd.DataFrame(title)
    title.columns = ['Title']
    title_one_hot = pd.get_dummies(title)
    title_one_hot.columns=['Is_Master','Is_Miss','Is_Mr','Is_Mrs','Is_Rare']
    # print title_one_hot
    df = pd.concat([df, title_one_hot], axis=1)

    # 补齐船票价格缺失值
    if len(df.Fare[df.Fare.isnull()]) > 0:
        fare = np.zeros(3)
        for f in range(0, 3):
            fare[f] = df[df.Pclass == f + 1]['Fare'].dropna().median()
        for f in range(0, 3):  # loop 0 to 2
            df.loc[(df.Fare.isnull()) & (df.Pclass == f + 1), 'Fare'] = fare[f]

    # 年龄：使用均值代替缺失值
    # print '年龄使用均值填充'
    # mean_age = df['Age'].dropna().mean()
    # df.loc[(df.Age.isnull()), 'Age'] = mean_age

    # print '年龄使用中位数填充'
    # mean_age = df['Age'].dropna().median()
    # df.loc[(df.Age.isnull()), 'Age'] = mean_age


    print '随机森林预测缺失年龄2：--start--'
    data_for_age = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    age_exist = data_for_age.loc[(df.Age.notnull())]  # 年龄不缺失的数据
    age_null = data_for_age.loc[(df.Age.isnull())]
    # print age_exist
    x = age_exist.values[:, 1:]
    y = age_exist.values[:, 0]
    rfr = RandomForestRegressor(n_estimators=800)
    rfr.fit(x, y)
    age_hat = rfr.predict(age_null.values[:, 1:])
    # print age_hat
    df.loc[(df.Age.isnull()), 'Age'] = age_hat
    print '随机森林预测缺失年龄2：--over--'


    # print df.describe()
    df.to_csv('New_Data.csv')

    # x = df[['Pclass','Sex','Age','SibSp','Parch','Fare']]
    # x = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
    x = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Is_Master',
            'Is_Miss', 'Is_Mr', 'Is_Mrs', 'Is_Rare']]

    y = None
    if 'Survived' in df:
        y = df['Survived']

    x = np.array(x)
    y = np.array(y)

    # 提高重采样
    x = np.tile(x, (100, 1))
    y = np.tile(y, (100, ))

    if is_train:
        return x, y
    return x, df['PassengerId']

def analyzeAge(x,y):
    # 数据分析
    x_show = np.linspace(0,99,100)
    y_show = np.zeros(100)
    for i,d1 in enumerate(x['Age']):
        if not (pd.isnull(d1)):
            y_show[int(round(d1))] += 1
    plt.subplot('224')
    plt.plot(x_show,y_show,'r-',label=u'All')
    # plt.bar(x_show, y_show, width=0.2, align="center", yerr=0.000001)
    plt.title(u'全部')
    Pclass = set(x['Pclass'])
    for n,pc in enumerate(Pclass):
        yt_show = np.zeros(100)
        for i,d2 in enumerate(x['Age']):
            if not (pd.isnull(d2)):
                if x['Pclass'][i]==pc:
                    yt_show[int(round(d2))] += 1
        subplotStr = '22%d'%(n+1)
        plt.subplot(subplotStr)
        plt.plot(x_show, yt_show, 'r-', label=u'All')
        plt.title(u'Pclass=%d'%(pc))
    plt.show()

def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    acc_rate = 100 * float(acc.sum()) / a.size
    print '%s正确率：%.3f%%' % (tip, acc_rate)
    print '正确率：\t', accuracy_score(a, b)
    print ' 精度 ：\t', precision_score(a, b, pos_label=1)
    print '召回率：\t', recall_score(a, b, pos_label=1)
    print 'F1-score：\t', f1_score(a, b, pos_label=1)
    return acc_rate


def gen_results(path, c, c_type):
    print '读取测试集...'
    x, passenger_id = preprocessData(path,False)
    y = c.predict(x)
    predictions_file = open("Prediction_%d.csv" % c_type, "wb")
    print '生成结果文件...'
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["PassengerId", "Survived"])
    open_file_object.writerows(zip(passenger_id, y))
    predictions_file.close()
    print '完成。'


if __name__ == "__main__":
    path = '../datasets/Titanic/train.csv'
    x,y = preprocessData(path,True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=1)
    rfc = RandomForestClassifier(n_estimators=100,)
    rfc.fit(x_train,y_train.ravel())
    y_hat = rfc.predict(x_test)
    y_train_hat = rfc.predict(x_train)
    rfc_rate = show_accuracy(y_train_hat,y_train,'随机森林训练集')
    rfc_rate = show_accuracy(y_hat,y_test,'随机森林测试集')

    # 从测试数据生成结果
    path = '../datasets/Titanic/test.csv'
    gen_results(path,rfc,1)




