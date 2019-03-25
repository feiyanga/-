# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:45:54 2019

@author: Feiyang
"""
#https://www.cnblogs.com/star-zhao/p/9801196.html
import pandas as pd
pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000)
df_train,df_test=pd.read_csv('C:/Users/Feiyang/Desktop/pycase/taitannike/data/train.csv'),pd.read_csv('C:/Users/Feiyang/Desktop/pycase/taitannike/data/test.csv')
#查看前5行数据
df_train.head()
#查看后5行数据
df_train.tail()
#查看数据信息, 其中包含数据维度, 数据类型, 所占空间等信息
df_train.info()
df_train.describe()
#同样是使用describe()方法
df_train[['Name','Sex','Ticket','Cabin','Embarked']].describe()
#1. PassengerId
#2. Pclass
import numpy as np
import matplotlib.pyplot as plt
#生成Pclass_Survived的列联表
Pclass_Survived = pd.crosstab(df_train['Pclass'], df_train['Survived'])
#绘制堆积柱形图
Pclass_Survived.plot(kind = 'bar', stacked = True)
Survived_len = len(Pclass_Survived.count())
Pclass_index = np.arange(len(Pclass_Survived.index))
Sum1 = 0
for i in range(Survived_len):
    SurvivedName = Pclass_Survived.columns[i]
    PclassCount = Pclass_Survived[SurvivedName]
    Sum1, Sum2= Sum1 + PclassCount, Sum1
    Zsum = Sum2 + (Sum1 - Sum2)/2
    for x, y, z in zip(Pclass_index, PclassCount, Zsum):
        #添加数据标签
        plt.text(x,z, '%.0f'%y, ha = 'center',va='center' )
#修改x轴标签
plt.xticks(Pclass_Survived.index-1, Pclass_Survived.index, rotation=360)
plt.title('Survived status by pclass')

#生成Survived为0时, 每个Pclass的总计数
Pclass_Survived_0 = df_train.Pclass[df_train['Survived'] == 0].value_counts()
#生成Survived为1时, 每个Pclass的总计数
Pclass_Survived_1 = df_train.Pclass[df_train['Survived'] == 1].value_counts()
#将两个状况合并为一个DateFrame 
Pclass_Survived = pd.DataFrame({ 0: Pclass_Survived_0, 1: Pclass_Survived_1})

#3Name
import re
df_train['Appellation']=df_train.Name.apply(lambda x: re.search('\w+\.',x).group()).str.replace('.','')
df_train.Appellation.unique()
Appellation_Sex = pd.crosstab(df_train.Appellation, df_train.Sex)
Appellation_Sex.T
df_train['Appellation'] = df_train['Appellation'].replace(['Capt','Col','Countess','Don','Dr','Jonkheer','Lady','Major','Rev','Sir'], 'Rare')
df_train['Appellation'] = df_train['Appellation'].replace(['Mlle','Ms'], 'Miss')
df_train['Appellation'] = df_train['Appellation'].replace('Mme', 'Mrs')
df_train.Appellation.unique()
Appellation_Survived = pd.crosstab(df_train['Appellation'], df_train['Survived'])
Appellation_Survived.plot(kind = 'bar')
plt.xticks(np.arange(len(Appellation_Survived.index)), Appellation_Survived.index, rotation = 360)

#4. Sex
Sex_Survived=pd.crosstab(df_train.Sex,df_train.Survived)
Survived_len = len(Sex_Survived.count())
Sex_index = np.arange(len(Sex_Survived.index))
single_width = 0.35
for i in range(Survived_len):
    SurvivedName = Sex_Survived.columns[i]
    SexCount = Sex_Survived[SurvivedName]
    SexLocation = Sex_index * 1.05 + (i - 1/2)*single_width
   #绘制柱形图
    plt.bar(SexLocation, SexCount, width = single_width)
    for x, y in zip(SexLocation, SexCount):
        #添加数据标签
        plt.text(x, y, '%.0f'%y, ha='center', va='bottom')
index = Sex_index * 1.05 
plt.xticks(index, Sex_Survived.index, rotation=360)
plt.title('Survived status by sex')

#6. SibSp
#生成列联表
SibSp_Survived = pd.crosstab(df_train['SibSp'], df_train['Survived'])
SibSp_Survived.plot(kind = 'bar')
plt.title('Survived status by SibSp')

#7. Parch
Parch_Survived = pd.crosstab(df_train['Parch'], df_train['Survived'])
Parch_Survived.plot(kind = 'bar')
plt.title('Survived status by Parch')

#Ticket
#计算每张船票使用的人数
Ticket_count=df_train.groupby('Ticket',as_index=False)['PassengerId'].count()
#获取使用人数为1的船票
Ticket_count_0=Ticket_count[Ticket_count.PassengerId==1]['Ticket']
#当船票在已经筛选出使用人数为1的船票中时, 将0赋值给GroupTicket, 否则将1赋值给GroupTicket
df_train['GroupTicket']=np.where(df_train.Ticket.isin(Ticket_count_0),0,1)
#绘制柱形图
GroupTicket_Survived = pd.crosstab(df_train['GroupTicket'], df_train['Survived'])
GroupTicket_Survived.plot(kind = 'bar', stacked = True)
Survived_len = len(GroupTicket_Survived.count())
GroupTicket_index = np.arange(len(GroupTicket_Survived.index))
Sum1 = 0
for i in range(Survived_len):
    SurvivedName = GroupTicket_Survived.columns[i]
    GroupTicketCount = GroupTicket_Survived[SurvivedName]
    Sum1, Sum2= Sum1 + GroupTicketCount, Sum1
    Zsum = Sum2 + (Sum1 - Sum2)/2
    for x, y, z in zip(GroupTicket_index, GroupTicketCount, Zsum):
        #添加数据标签
        plt.text(x,z, '%.0f'%y, ha = 'center',va='center' )
#修改x轴标签
plt.xticks(GroupTicket_index, GroupTicket_Survived.index, rotation=360)
plt.title('Survived status by GroupTicket')

#9. Fare
#对Fare进行分组: 2**10>891分成10组, 组距为(最大值512.3292-最小值0)/10取值60
bins=[0,60,120,180,240,300,360,420,480,540,600]
df_train['GroupFare']=pd.cut(df_train.Fare,bins,right=False)
GroupFare_Survived = pd.crosstab(df_train['GroupFare'], df_train['Survived'])
GroupFare_Survived.plot(kind = 'bar')
plt.title('Survived status by GroupFare')
GroupFare_Survived.iloc[2:].plot(kind = 'bar')
plt.title('Survived status by GroupFare(Fare>=120)')

#10. Cabin 由于含有大量缺失值, 处理完缺失值再对其进行分析.

#11.Embarked 同样也含有缺失值, 处理完缺失值在对其进行分析.

#4. 特征工程（在缺失值处理之前, 应当将数据拷贝一份, 以保证原始数据的完整性）
train = df_train.copy()
#众数进行填充
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
train['Cabin']=train['Cabin'].fillna('No')#不妨用'NO'来填充.
#求出每个头衔对应的年龄中位数
Age_Appellation_median = train.groupby('Appellation')['Age'].median()
#在当前表设置Appellation为索引 
train.set_index('Appellation', inplace = True)
#在当前表填充缺失值
train.Age.fillna(Age_Appellation_median, inplace = True)
#重置索引
train.reset_index(inplace = True)
#第一种: 返回0即表示没有缺失值
train.Age.isnull().sum()
#第二种: 返回False即表示没有缺失值
train.Age.isnull().any()
#第三种: 描述性统计
train.Age.describe()

#4.2 缺失特征分析
#4.2.1 Embarked
Embarked_Survived=pd.crosstab(train['Embarked'],train['Survived'])
#图1
Embarked_Survived.plot(kind='bar', stacked = True)
Survived_len=len(Embarked_Survived.count())
Survived_index=np.arange(len(Embarked_Survived.index))
sum1=0
for i in range(Survived_len):
    EmbarkedName=Embarked_Survived.columns[i]
    Embarkedcount=Embarked_Survived[EmbarkedName]
    sum1,sum2=sum1+Embarkedcount,sum1
    Zsum = sum2 + (sum1 - sum2)/2
    for x,y,z in zip(Survived_index,Embarkedcount,Zsum):
        plt.text(x,z,y, ha = 'center',va='center')
        
plt.xticks(Survived_index,Embarked_Survived.index,rotation=360)
plt.title('Survived status by Embarked')
#图2
#Survived_len=len(Embarked_Survived.count())
#Survived_index=np.arange(len(Embarked_Survived.index))
#s1=0.3
#for i in range(Survived_len):
    #EmbarkedName=Embarked_Survived.columns[i]
    #Embarkedcount=Embarked_Survived[EmbarkedName]
    #pl=Survived_index+(i-0.5)*s1
    #plt.bar(pl,Embarkedcount,width=s1)
    #for x,y in zip(pl,Embarkedcount):
        #plt.text(x,y,y,ha = 'center',va='bottom')       
#plt.xticks(Survived_index,Embarked_Survived.index,rotation=360)
#plt.title('Survived status by Embarked')

#4.2.2 Cabin
#将没有舱位的归为0, 有舱位归为1.
train['GroupCabin'] = np.where(train.Cabin == 'No',0, 1)
#绘制柱形图
GroupCabin_Survived = pd.crosstab(train['GroupCabin'], train['Survived'])
Survived_len=len(GroupCabin_Survived.count())
Survived_index=np.arange(len(GroupCabin_Survived.index))
s1=0.3
for i in range(Survived_len):
    GroupCabinName=GroupCabin_Survived.columns[i]
    GroupCabincount=GroupCabin_Survived[GroupCabinName]
    pl=Survived_index+(i-0.5)*s1
    plt.bar(pl,GroupCabincount,width=s1)
    for x,y in zip(pl,GroupCabincount):
        plt.text(x,y,y,ha = 'center',va='bottom')       
plt.xticks(Survived_index,GroupCabin_Survived.index,rotation=360)
plt.title('Survived status by GroupCabin')

#4.2.3 Age
#对Age进行分组: 2**10>891分成10组, 组距为(最大值80-最小值0)/10 =8取9
bins=[0, 9, 18, 27, 36, 45, 54, 63, 72, 81, 90]
train['GroupAge'] = pd.cut(train.Age, bins)
GroupAge_Survived = pd.crosstab(train['GroupAge'], train['Survived'])
Survived_len=len(GroupAge_Survived.count())
Survived_index=np.arange(len(GroupAge_Survived.index))
s1=0.3
for i in range(Survived_len):
    GroupAgeName=GroupAge_Survived.columns[i]
    GroupAgecount=GroupAge_Survived[GroupAgeName]
    pl=Survived_index+(i-0.5)*s1
    plt.bar(pl,GroupAgecount,width=s1)
    for x,y in zip(pl,GroupAgecount):
        plt.text(x,y,y,ha = 'center',va='bottom')       
plt.xticks(Survived_index,GroupAge_Survived.index,rotation=360)
plt.title('Survived status by GroupAge')

#4.3 新特征的提取
#1. Pclass中没有更多信息可供提取, 且为定量变量, 这里不作处理. 

#2、3 Appellation,Sex是定性变量, 将其转化为定量变量: 
train['Appellation'] = train.Appellation.map({'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master': 3, 'Rare': 4})
train.Appellation.unique()
train['Sex'] = train['Sex'].map({'female': 0, 'male': 1})

#4. 按照GroupAge特征的范围将Age分为10组.
train.loc[train['Age'] < 9, 'Age'] = 0
train.loc[(train['Age'] >= 9) & (train['Age'] < 18), 'Age'] = 1
train.loc[(train['Age'] >= 18) & (train['Age'] < 27), 'Age'] = 2
train.loc[(train['Age'] >= 27) & (train['Age'] < 36), 'Age'] = 3
train.loc[(train['Age'] >= 36) & (train['Age'] < 45), 'Age'] = 4
train.loc[(train['Age'] >= 45) & (train['Age'] < 54), 'Age'] = 5
train.loc[(train['Age'] >= 54) & (train['Age'] < 63), 'Age'] = 6
train.loc[(train['Age'] >= 63) & (train['Age'] < 72), 'Age'] = 7
train.loc[(train['Age'] >= 72) & (train['Age'] < 81), 'Age'] = 8
train.loc[(train['Age'] >= 81) & (train['Age'] < 90), 'Age'] = 9
train.Age.unique()

#5. 将SibSp和Parch这两个特征组合成FamilySize特征,当SibSp和Parch都为0时, 则孤身一人.
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
train.FamilySize.unique()

#6. GroupTicket是定量变量, 不作处理
#7按照GroupFare特征的范围将Fare分成10组:
train.loc[train['Fare'] < 60, 'Fare'] = 0
train.loc[(train['Fare'] >= 60) & (train['Fare'] < 120), 'Fare'] = 1
train.loc[(train['Fare'] >= 120) & (train['Fare'] < 180), 'Fare'] = 2
train.loc[(train['Fare'] >= 180) & (train['Fare'] < 240), 'Fare'] = 3
train.loc[(train['Fare'] >= 240) & (train['Fare'] < 300), 'Fare'] = 4
train.loc[(train['Fare'] >= 300) & (train['Fare'] < 360), 'Fare'] = 5
train.loc[(train['Fare'] >= 360) & (train['Fare'] < 420), 'Fare'] = 6
train.loc[(train['Fare'] >= 420) & (train['Fare'] < 480), 'Fare'] = 7
train.loc[(train['Fare'] >= 480) & (train['Fare'] < 540), 'Fare'] = 8
train.loc[(train['Fare'] >= 540) & (train['Fare'] < 600), 'Fare'] = 9
train.Fare.unique()
# 8. GroupCabin是定量变量, 不作处理
# 9. Embarked是定类变量, 转化为定量变量.
train['Embarked'] = train.Embarked.map({'S': 0, 'C': 1, 'Q': 2})

# 现有特征:
#PassengerId, Survived, Pclass, Name, Appellation,Sex, Age, GroupAge, SibSp, 
#Parch, FamilySize, Ticket, GroupTicket, Fare, GroupFare, Cabin, GroupCabin, Embarked
#删除重复多余的以及与Survived不相关的:
train.drop(['PassengerId', 'Name', 'GroupAge', 'SibSp', 
            'Parch', 'Ticket', 'GroupFare', 'Cabin'], axis = 1, inplace =True)
#删除后, 现有特征:
#Survived, Pclass, Appellation, Sex, Age, FamilySize, GroupTicket, Fare, GroupCabin, Embarked.

#5. 构建模型 用sklearn库实现机器学习算法 考虑用到的算法有: 逻辑回归, 决策树
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X=train[['Pclass', 'Appellation', 'Sex', 'Age', 'FamilySize', 'GroupTicket', 'Fare', 'GroupCabin', 'Embarked']]
y=train['Survived']
#随机划分训练集和测试集
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
#逻辑回归模型初始化
lg=LogisticRegression()
#训练逻辑回归模型
lg.fit(X_train,y_train)
#用测试数据检验模型好坏
lg.score(X_test, y_test)

from sklearn.tree import DecisionTreeClassifier
#树的最大深度为15, 内部节点再划分所需最小样本数为2, 叶节点最小样本数1, 最大叶子节点数10, 每次分类的最大特征数6
dt=DecisionTreeClassifier(max_depth=15,min_samples_split=2,min_samples_leaf=1,max_leaf_nodes=10,max_features=6)
dt.fit(X_train, y_train)
dt.score(X_test, y_test)