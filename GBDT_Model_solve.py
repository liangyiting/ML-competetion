# -*- coding: utf-8 -*-
"""
#coding:utf-8
Created on Mon Nov 16 08:52:38 2015

@author: Administrator
"""
import os
import xlrd
from numpy import *
#os.getcwd()
os.chdir('F:\下载中心\拍拍贷魔镜比赛\比赛程序')

'''1. 提取训练数据'''
import csv
import datetime

def write_txt(x):
    result='';
    n,m=x.shape
    for i in range(n):
        line='';
        for j in range(m):
            if j<m-1:
                line=line+str(x[i,j])+' '
            else:
                line=line+str(x[i,j])+'\n'
        result=result+line
    return(result)   
    
def write_txt_1(x):
    line='';
    n=len(x)
    for i in range(n):
        if i<n-1:
            line=line+str(x[i])+' '
        else:
            line=line+str(x[i])
    return(line)    

'''下面是从matlab写的csv文件中读取的程序'''
def readx_1(datax):
    n=len(datax)
    m=datax[1].count(',')+1
    X=zeros([n,m])
    i=0
    for line in datax:
        a=line.split(',')
        a[-1]=a[-1][0:-1]
        j=0
        for s in a:
            X[i,j]=s
            j=j+1
        i=i+1
    return X
    
def f_predict(xval,Yam):
    lr = joblib.load('gbdt_2.model')
    probs=lr.predict(xval)+Yam.flatten();
    lr = joblib.load('gbdt_8.model')
    probs=probs+lr.predict(xval)+Yam.flatten();
    lr = joblib.load('gbdt_7.model')
    probs=probs+lr.predict(xval)+Yam.flatten();
    lr = joblib.load('gbdt_13.model')
    probs=probs+lr.predict(xval)+Yam.flatten();
    lr = joblib.load('gbdt_61.model')
    probs=probs+lr.predict(xval)+Yam.flatten();
    lr = joblib.load('gbdt_51.model')
    probs=probs+lr.predict(xval)+Yam.flatten();
    lr = joblib.load('gbdt_43.model')
    probs=probs+lr.predict(xval)+Yam.flatten();
    lr = joblib.load('gbdt_32.model')
    probs=probs+lr.predict(xval)+Yam.flatten();
    lr = joblib.load('gbdt_25.model')
    probs=probs+lr.predict(xval)*2.5+Yam.flatten();
    lr = joblib.load('gbdt_43.model')
    probs=probs+lr.predict(xval)+Yam.flatten();
    lr = joblib.load('gbdt_39.model')
    probs=probs+lr.predict(xval)+Yam.flatten();
    return(probs)
    
def f_predict1(xval,Yam):
    probs=3*Yam.flatten();
    lr = joblib.load('s25.model')
    probs=probs+0.6*lr.predict(xval);
    lr = joblib.load('s24.model')
    probs=probs+0.6*(lr.predict(xval));
    lr = joblib.load('s16.model')
    probs=probs+0.6*(lr.predict(xval));
    lr = joblib.load('s34.model')
    probs=probs+0.4*(lr.predict(xval));
    lr = joblib.load('s32.model')
    probs=probs+0.3*(lr.predict(xval));
    lr = joblib.load('s28.model')
    probs=probs+0.3*(lr.predict(xval));
    lr = joblib.load('s30.model')
    probs=probs+0.3*(lr.predict(xval));
    lr = joblib.load('s35.model')
    probs=probs+0.2*(lr.predict(xval));
    lr = joblib.load('s29.model')
    probs=probs+0.2*(lr.predict(xval));
    lr = joblib.load('s10.model')
    probs=probs+0.2*(lr.predict(xval));
    lr = joblib.load('s15.model')
    probs=probs+0.2*(lr.predict(xval));
    return(probs)
    

file=open('F:\下载中心\拍拍贷魔镜比赛\比赛程序\数据处理/Xt.csv')
data=file.readlines();
file.close()
Xt=readx_1(data)


file=open('F:\下载中心\拍拍贷魔镜比赛\比赛程序\数据处理/X.csv')
data=file.readlines();
file.close()
X=readx_1(data)

file=open('F:\下载中心\拍拍贷魔镜比赛\比赛程序\数据处理/Y.csv')
data=file.readlines();
file.close()
Y=readx_1(data)

file=open('F:\下载中心\拍拍贷魔镜比赛\比赛程序\数据处理/Xa.csv')
data=file.readlines();
file.close()
Xa=readx_1(data)

file=open('F:\下载中心\拍拍贷魔镜比赛\比赛程序\数据处理/Ya.csv')
data=file.readlines();
file.close()
Ya=readx_1(data)


file=open('F:\下载中心\拍拍贷魔镜比赛\比赛程序\数据处理/Xg.csv')
data=file.readlines();
file.close()
Xg=readx_1(data)

file=open('F:\下载中心\拍拍贷魔镜比赛\比赛程序\数据处理/Xag.csv')
data=file.readlines();
file.close()
Xag=readx_1(data)

file=open('F:\下载中心\拍拍贷魔镜比赛\比赛程序\数据处理/x0.csv')
data=file.readlines();
file.close()
x0=readx_1(data)

''' 导入matlab数据'''
file=open('F:\下载中心\拍拍贷魔镜比赛\比赛程序\数据处理/x.csv')
data=file.readlines();
file.close()
x=readx_1(data)

file=open('F:\下载中心\拍拍贷魔镜比赛\比赛程序\数据处理/y.csv')
data=file.readlines();
file.close()
y=readx_1(data)

file=open('F:\下载中心\拍拍贷魔镜比赛\比赛程序\数据处理/y0.csv')
data=file.readlines();
file.close()
y0=readx_1(data)

file=open('F:\下载中心\拍拍贷魔镜比赛\比赛程序\数据处理/Yam.csv')
data=file.readlines();
file.close()
Yam=readx_1(data)

file=open('F:\下载中心\拍拍贷魔镜比赛\比赛程序\数据处理/xa.csv')
data=file.readlines();
file.close()
xa=readx_1(data)

file=open('F:\下载中心\拍拍贷魔镜比赛\比赛程序\数据处理/x0t.csv')
data=file.readlines();
file.close()
x0t=readx_1(data)

file=open('F:\下载中心\拍拍贷魔镜比赛\比赛程序\数据处理/ytest0.csv')
data=file.readlines();
file.close()
ytest0=readx_1(data)

xfit=x;yfit=y-y0;
yfit=y;
xval=xa;yval=Ya;#'ls', 'lad', 'huber', 'quantile'

model=GradientBoostingRegressor(loss='ls',alpha=0.8,n_estimators=1000,\
learning_rate=0.002,max_depth=12,min_samples_leaf=10,\
max_features=0.04,subsample=0.16,max_leaf_nodes=60)

model = joblib.load('gbdt_107.model')
model = joblib.load('gbdt_119.model')
model = joblib.load('gbdt_125.model')
model = joblib.load('gbdt_132.model')
model = joblib.load('gbdt_151.model')
model = joblib.load('gbdt_170.model')
model.learning_rate=0.005
model.n_estimators=3000
model.max_leaf_nodes=60
model.max_depth=10;
model.subsample=0.16;
model.max_features=0.04
#lr=joblib.load('gbdt_97.model')
#lr=model
#model=GradientBoostingRegressor(init=lr,loss='ls',n_estimators=50,\
#learning_rate=0.01,max_depth=10,min_samples_leaf=20,\
#max_features=0.05,subsample=0.2,max_leaf_nodes=60)
bm=dtr(max_depth=6,min_samples_leaf=2,max_leaf_nodes=60,splitter='random')
#model=BaggingRegressor(base_estimator=bm,n_estimators=2000,bootstrap=True,\
#bootstrap_features=1,max_samples=0.16,max_features=0.05)
model=AdaBoostRegressor(n_estimators=300,learning_rate=0.03,\
loss='square',base_estimator=bm)

model.fit(xfit,yfit.flatten())
probs=1*model.predict(xval)+1*Yam.flatten();
fpr, tpr, thresholds = roc_curve(yval, probs)
roc_auc=auc(fpr,tpr)
print(roc_auc)

probs=f_predict1(xval,Yam)
ytest=f_predict1(x0t,ytest0)




probs=1*Yam.flatten();
lr = joblib.load('s1.model')
probs=probs+0.2*lr.predict(xval);
lr = joblib.load('s2.model')
probs=probs+0.4*(lr.predict(xval));
lr = joblib.load('s3.model')
probs=probs+0.2*(lr.predict(xval));
lr = joblib.load('s4.model')
probs=probs+0.2*(lr.predict(xval));
lr = joblib.load('s5.model')
probs=probs+0.2*(lr.predict(xval));

lr = joblib.load('s6.model')
probs=probs+0.2*lr.predict(xval);
lr = joblib.load('s7.model')
probs=probs+0.2*(lr.predict(xval));
lr = joblib.load('s8.model')
probs=probs+0.2*(lr.predict(xval));
lr = joblib.load('s9.model')
probs=probs+0.2*(lr.predict(xval));
lr = joblib.load('s10.model')
probs=probs+0.4*(lr.predict(xval));

lr = joblib.load('s11.model')
probs=probs+0.2*lr.predict(xval);
lr = joblib.load('s12.model')
probs=probs+0.2*(lr.predict(xval));
lr = joblib.load('s13.model')
probs=probs+0.2*(lr.predict(xval));
lr = joblib.load('s14.model')
probs=probs+0.2*(lr.predict(xval));
lr = joblib.load('s15.model')
probs=probs+0.4*(lr.predict(xval));

lr = joblib.load('s16.model')
probs=probs+0.8*lr.predict(xval);
lr = joblib.load('s17.model')
probs=probs+0.2*(lr.predict(xval));
lr = joblib.load('s18.model')
probs=probs+0.2*(lr.predict(xval));
lr = joblib.load('s19.model')
probs=probs+0.2*(lr.predict(xval));
lr = joblib.load('s20.model')
probs=probs+0.2*(lr.predict(xval));

probs=1.8*Yam.flatten();
lr = joblib.load('s21.model')
probs=probs+0.4*lr.predict(xval);
lr = joblib.load('s22.model')
probs=probs+0.2*(lr.predict(xval));
lr = joblib.load('s23.model')
probs=probs+0.2*(lr.predict(xval));
lr = joblib.load('s24.model')
probs=probs+0.4*(lr.predict(xval));
lr = joblib.load('s25.model')
probs=probs+0.4*(lr.predict(xval));

lr = joblib.load('s26.model')
probs=probs+0.2*lr.predict(xval);
lr = joblib.load('s27.model')
probs=probs+0.2*(lr.predict(xval));
lr = joblib.load('s28.model')
probs=probs+0.2*(lr.predict(xval));
lr = joblib.load('s29.model')
probs=probs+0.2*(lr.predict(xval));
lr = joblib.load('s30.model')
probs=probs+0.2*(lr.predict(xval));

fpr, tpr, thresholds = roc_curve(yval, probs)
roc_auc=auc(fpr,tpr)
print(roc_auc)




lr = joblib.load('gbdt_164.model')
#model=RandomForestRegressor(n_estimators=10,max_depth=7,\
#max_leaf_nodes=10,min_samples_leaf=10)
#
#bm=Ridge(alpha=3000)
#model=AdaBoostRegressor(n_estimators=100,learning_rate=0.01,\
#loss='square')
#model=BaggingRegressor(base_estimator=bm,n_estimators=10,bootstrap=True,\
#bootstrap_features=1,max_samples=0.5,max_features=1.0)


probs=5.0*Yam.flatten();
#lr = joblib.load('gbdt_95.model')
#probs=probs+0.1*(lr.predict(xval));
#lr = joblib.load('gbdt_96.model')
#probs=probs+0.4*(lr.predict(xval));
#lr = joblib.load('gbdt_99.model')
#probs=probs+0.5*(lr.predict(xval));
#
#lr = joblib.load('gbdt_97.model')
#probs=probs+0.2*lr.predict(xval);
#lr = joblib.load('gbdt_98.model')
#probs=probs+0.12*lr.predict(xval);
#lr = joblib.load('gbdt_102.model')
#probs=probs+0.13*(lr.predict(xval));
#lr = joblib.load('gbdt_101.model')
#probs=probs+0.12*(lr.predict(xval));
#lr = joblib.load('gbdt_104.model')
#probs=probs+0.13*(lr.predict(xval));
#lr = joblib.load('gbdt_105.model')
#probs=probs+0.12*(lr.predict(xval));

probs=2*Yam.flatten();

lr = joblib.load('gbdt_99.model')
probs=probs+0.2*lr.predict(xval);
lr = joblib.load('gbdt_95.model')
probs=probs+0.2*(lr.predict(xval));
lr = joblib.load('gbdt_96.model')
probs=probs+0.2*(lr.predict(xval));
lr = joblib.load('gbdt_97.model')
probs=probs+0.2*(lr.predict(xval));
lr = joblib.load('gbdt_106.model')
probs=probs+0.2*(lr.predict(xval));

lr = joblib.load('gbdt_107.model')
probs=probs+0.3*(lr.predict(xval));
lr = joblib.load('gbdt_108.model')
probs=probs+0.4*(lr.predict(xval));
lr = joblib.load('gbdt_109.model')
probs=probs+0.3*(lr.predict(xval));

lr = joblib.load('gbdt_110.model')
probs=probs+0.2*(lr.predict(xval));
lr = joblib.load('gbdt_111.model')
probs=probs+0.3*(lr.predict(xval));
lr = joblib.load('gbdt_112.model')
probs=probs+0.2*(lr.predict(xval));
lr = joblib.load('gbdt_113.model')
probs=probs+0.3*(lr.predict(xval));

lr = joblib.load('gbdt_98.model')
probs=probs+0.2*(lr.predict(xval));
lr = joblib.load('gbdt_100.model')
probs=probs+0.3*(lr.predict(xval));
lr = joblib.load('gbdt_102.model')
probs=probs+0.2*(lr.predict(xval));
lr = joblib.load('gbdt_101.model')
probs=probs+0.3*(lr.predict(xval));

lr = joblib.load('gbdt_103.model')
probs=probs+0.3*(lr.predict(xval));
lr = joblib.load('gbdt_104.model')
probs=probs+0.2*(lr.predict(xval));
lr = joblib.load('gbdt_105.model')
probs=probs+0.3*(lr.predict(xval));

lr = joblib.load('gbdt_117.model')
probs=probs+0.2*(lr.predict(xval));
lr = joblib.load('gbdt_118.model')
probs=probs+0.3*(lr.predict(xval));
lr = joblib.load('gbdt_119.model')
probs=probs+0.2*(lr.predict(xval));
lr = joblib.load('gbdt_121.model')
probs=probs+0.3*(lr.predict(xval))

lr = joblib.load('gbdt_122.model')
probs=probs+0.2*(lr.predict(xval));
lr = joblib.load('gbdt_123.model')
probs=probs+0.3*(lr.predict(xval));
lr = joblib.load('gbdt_124.model')
probs=probs+0.2*(lr.predict(xval));
lr = joblib.load('gbdt_125.model')
probs=probs+0.3*(lr.predict(xval))

probs=0*Yam.flatten();

lr = joblib.load('gbdt_126.model')
probs=probs+0.2*(lr.predict(xval));
lr = joblib.load('gbdt_127.model')
probs=probs+0.3*(lr.predict(xval));
lr = joblib.load('gbdt_128.model')
probs=probs+0.5*(lr.predict(xval));
lr = joblib.load('gbdt_129.model')
probs=probs+0.5*(lr.predict(xval))

lr = joblib.load('gbdt_130.model')
probs=probs+0.8*(lr.predict(xval));
lr = joblib.load('gbdt_131.model')
probs=probs+0.8*(lr.predict(xval))
lr = joblib.load('gbdt_132.model')
probs=probs+0.5*(lr.predict(xval));
lr = joblib.load('gbdt_133.model')
probs=probs+0.8*(lr.predict(xval))

probs=2*Yam.flatten();

lr = joblib.load('gbdt_134.model')
probs=probs+0.8*(lr.predict(xval));
lr = joblib.load('gbdt_135.model')
probs=probs+0.8*(lr.predict(xval))
lr = joblib.load('gbdt_136.model')
probs=probs+0.8*(lr.predict(xval));
lr = joblib.load('gbdt_137.model')
probs=probs+0.8*(lr.predict(xval))

lr = joblib.load('gbdt_138.model')
probs=probs+0.3*(lr.predict(xval));
lr = joblib.load('gbdt_139.model')
probs=probs+0.3*(lr.predict(xval))
lr = joblib.load('gbdt_140.model')
probs=probs+0.3*(lr.predict(xval));
lr = joblib.load('gbdt_141.model')
probs=probs+0.3*(lr.predict(xval))

lr = joblib.load('gbdt_142.model')
probs=probs+0.3*(lr.predict(xval));
lr = joblib.load('gbdt_143.model')
probs=probs+0.3*(lr.predict(xval))

probs=2*Yam.flatten();
lr = joblib.load('gbdt_127.model')
probs=probs+0.3*(lr.predict(xval));
lr = joblib.load('gbdt_128.model')
probs=probs+0.3*(lr.predict(xval))
lr = joblib.load('gbdt_132.model')
probs=probs+0.3*(lr.predict(xval));
lr = joblib.load('gbdt_125.model')
probs=probs+0.3*(lr.predict(xval));


lr = joblib.load('gbdt_144.model')
probs=probs+0.3*(lr.predict(xval));
lr = joblib.load('gbdt_145.model')
probs=probs+0.3*(lr.predict(xval))
lr = joblib.load('gbdt_146.model')
probs=probs+0.3*(lr.predict(xval));
lr = joblib.load('gbdt_147.model')
probs=probs+0.3*(lr.predict(xval))

lr = joblib.load('gbdt_148.model')
probs=probs+0.3*(lr.predict(xval));
lr = joblib.load('gbdt_149.model')
probs=probs+0.3*(lr.predict(xval))
lr = joblib.load('gbdt_150.model')
probs=probs+0.3*(lr.predict(xval));
lr = joblib.load('gbdt_151.model')
probs=probs+0.3*(lr.predict(xval));
#lr = joblib.load('gbdt_152.model')
#probs=probs+0.3*(lr.predict(xval));
lr = joblib.load('gbdt_153.model')
probs=probs+0.3*(lr.predict(xval));
lr = joblib.load('gbdt_154.model')
probs=probs+0.3*(lr.predict(xval));
lr = joblib.load('gbdt_155.model')
probs=probs+0.3*(lr.predict(xval))

lr = joblib.load('gbdt_146.model')
probs=probs+0.3*(lr.predict(xval));
lr = joblib.load('gbdt_150.model')
probs=probs+0.3*(lr.predict(xval));
lr = joblib.load('gbdt_151.model')
probs=probs+0.3*(lr.predict(xval))
lr = joblib.load('gbdt_148.model')
probs=probs+0.3*(lr.predict(xval))

probs=2*Yam.flatten();

lr = joblib.load('gbdt_159.model')
probs=probs+0.3*(lr.predict(xval));
lr = joblib.load('gbdt_160.model')
probs=probs+0.3*(lr.predict(xval));
lr = joblib.load('gbdt_161.model')
probs=probs+0.3*(lr.predict(xval))
lr = joblib.load('gbdt_162.model')
probs=probs+0.3*(lr.predict(xval))

lr = joblib.load('gbdt_163.model')
probs=probs+0.3*(lr.predict(xval));
lr = joblib.load('gbdt_164.model')
probs=probs+0.3*(lr.predict(xval))
lr = joblib.load('gbdt_165.model')
probs=probs+0.3*(lr.predict(xval))

lr = joblib.load('gbdt_166.model')
probs=probs+0.3*(lr.predict(xval));
lr = joblib.load('gbdt_167.model')
probs=probs+0.3*(lr.predict(xval))
lr = joblib.load('gbdt_168.model')
probs=probs+0.3*(lr.predict(xval))

lr = joblib.load('gbdt_169.model')
probs=probs+0.3*(lr.predict(xval));
lr = joblib.load('gbdt_170.model')
probs=probs+0.3*(lr.predict(xval))


fpr, tpr, thresholds = roc_curve(yval, probs)
roc_auc=auc(fpr,tpr)
print(roc_auc)

probs=f_predict(xval,Yam)
ytest=f_predict(x0t,ytest0)

file=open('F:\下载中心\拍拍贷魔镜比赛\比赛程序\数据处理/uid.csv')
data=file.readlines();
file.close()
uid=readx_1(data)
result='Idx,score\n';
n=len(ytest)
for i in range(n):
    result=result+str(int(uid[i]))+','+str(ytest[i])+'\n'
    
f=open('F:\下载中心\拍拍贷魔镜比赛\比赛程序/score0331.csv','w',encoding='utf-8')
f.write(result)
f.close()





lr = joblib.load('gbdt_85.model')
probs=lr.predict(xval)+Yam.flatten();
#probs=probs+lr.predict(xval)+Yam.flatten();
lr = joblib.load('gbdt_88.model')
probs=probs+lr.predict(xval)+Yam.flatten();
lr = joblib.load('gbdt_89.model')
probs=probs+lr.predict(xval)+Yam.flatten();
lr = joblib.load('gbdt_92.model')
probs=probs+lr.predict(xval)+Yam.flatten();
lr = joblib.load('gbdt_84.model')
probs=probs+lr.predict(xval)+Yam.flatten();
probs=probs/3;

fpr, tpr, thresholds = roc_curve(yval, probs)
roc_auc=auc(fpr,tpr)
print(roc_auc)



from sklearn.externals import joblib
#lr是一个LogisticRegression模型

joblib.dump(model, 'gbdt_16.model')#0.7627
joblib.dump(model, 'gbdt_6.model')#0.6986

joblib.dump(model, 'gbdt_5.model')#0.7604
joblib.dump(model, 'gbdt_10.model')#0.7606355
joblib.dump(model, 'gbdt_19.model')#0.7608

joblib.dump(model, 'gbdt_15.model')#0.7610
joblib.dump(model, 'gbdt_11.model')#0.76102
joblib.dump(model, 'gbdt.model')#0.761188
joblib.dump(model, 'gbdt_14.model')#0.76169
joblib.dump(model, 'gbdt_12.model')#0.76179
joblib.dump(model, 'gbdt_3.model')#0.761748
joblib.dump(model, 'gbdt_9.model')#0.761938

joblib.dump(model, 'gbdt_17.model')#0.762412
joblib.dump(model, 'gbdt_4.model')#0.76313
joblib.dump(model, 'gbdt_18.model')#0.7635
joblib.dump(model, 'gbdt_8.model')#0.76404
joblib.dump(model, 'gbdt_2.model')#0.7646
joblib.dump(model, 'gbdt_7.model')#0.7646708
joblib.dump(model, 'gbdt_13.model')#0.765314

lr = joblib.load('gbdt_2.model')
probs=lr.predict(xval)+Yam.flatten();
lr = joblib.load('gbdt_8.model')
probs=probs+lr.predict(xval)+Yam.flatten();
lr = joblib.load('gbdt_7.model')
probs=probs+lr.predict(xval)+Yam.flatten();
lr = joblib.load('gbdt_13.model')
probs=probs+lr.predict(xval)+Yam.flatten();
probs0=probs;

lr = joblib.load('gbdt.model')
probs=lr.predict(xval)+Yam.flatten();
lr = joblib.load('gbdt_3.model')
probs=probs+lr.predict(xval)+Yam.flatten();
#lr = joblib.load('gbdt_5.model')
#probs=probs+lr.predict(xval)+Yam.flatten();
#lr = joblib.load('gbdt_6.model')
#probs=probs+lr.predict(xval)+Yam.flatten();
#lr = joblib.load('gbdt_10.model')
#probs=probs+lr.predict(xval)+Yam.flatten();
lr = joblib.load('gbdt_9.model')
probs=probs+lr.predict(xval)+Yam.flatten();
lr = joblib.load('gbdt_11.model')
probs=probs+lr.predict(xval)+Yam.flatten();
lr = joblib.load('gbdt_14.model')
probs=probs+lr.predict(xval)+Yam.flatten();
lr = joblib.load('gbdt_12.model')
probs=probs+lr.predict(xval)+Yam.flatten();
lr = joblib.load('gbdt_15.model')
probs=probs+lr.predict(xval)+Yam.flatten();
lr = joblib.load('gbdt_18.model')
probs=probs+lr.predict(xval)+Yam.flatten()
lr = joblib.load('gbdt_17.model')
probs=probs+lr.predict(xval)+Yam.flatten()
lr = joblib.load('gbdt_4.model')
probs=probs+lr.predict(xval)+Yam.flatten();
probs1=probs/3;

probs=probs0+probs1;
fpr, tpr, thresholds = roc_curve(yval, probs)
roc_auc=auc(fpr,tpr)
print(roc_auc)



nfit=20000;xfit=Xg[:nfit];yfit=Y[:nfit]
xval=Xag;yval=Ya;

nfit=20000;xfit=X[:nfit];yfit=Y[:nfit]
xval=Xa;yval=Ya;
xval=X[nfit:];yval=Y[nfit:]
#model.fit(xfit,yfit)
#ypre=model.predict(xval)
nfit=20000;xfit=Xo[:nfit];yfit=Y[:nfit]
xval=Xo[nfit:];yval=Y[nfit:]

'''2. 在已有的数据集上进行算法的验证和测试'''

from sklearn.svm import SVR
from sklearn.svm import NuSVR
model=SVR()
from sklearn.linear_model import *
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression as LR
model=LR(C=0.004)
model=LR(C=0.01,penalty='l1')
from sklearn.linear_model import BayesianRidge as BR
model=BR(alpha_1=1e2,alpha_2=3e2,lambda_1=1e-9,lambda_2=1e-9,compute_score=False)
from sklearn.linear_model import (LinearRegression,Lasso,RandomizedLasso,Ridge)
from sklearn.feature_selection import (RFE,f_regression) 
from sklearn.ensemble import RandomForestRegressor as rfr  
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.neighbors import NearestNeighbors
model=RadiusNeighborsRegressor(radius=0.5,p=2)
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(n_estimators=10,max_depth=8,\
min_samples_split=2)
from sklearn.ensemble import AdaBoostRegressor
model=AdaBoostRegressor(n_estimators=400)
from sklearn.ensemble import GradientBoostingRegressor
model=GradientBoostingRegressor(n_estimators=100,\
learning_rate=0.1,max_depth=10)
from sklearn.ensemble import BaggingRegressor
mb=model;
model=BaggingRegressor(base_estimator=mb,n_estimators=20,bootstrap=1,\
bootstrap_features=1,max_samples=0.3,max_features=0.3)
model=LR(C=0.004)
model=LR(C=0.01,penalty='l1')
model=rfr(n_estimators=2000,max_depth=1,min_samples_leaf=20,\
min_samples_split=100)
model=BR(alpha_1=1e2,alpha_2=3e2,lambda_1=1e-9,\
lambda_2=1e-9,compute_score=False)
model_dtr=dtr(max_depth=3,min_samples_leaf=5,max_leaf_nodes=200,\
min_weight_fraction_leaf=0.05)
model=dtr(max_depth=4,min_samples_leaf=5,max_leaf_nodes=200,\
min_weight_fraction_leaf=0.05)

model=dtr(max_depth=3,min_samples_leaf=2,max_leaf_nodes=20,splitter='random')

model=AdaBoostRegressor(n_estimators=50,learning_rate=0.01,\
loss='square',base_estimator=mb)

model=Lasso(alpha=0.001)
model=Ridge(alpha=3000)
model=ElasticNetCV(normalize=True,l1_ratio=0.005)

model=KNeighborsRegressor(n_neighbors=200,p=1,weights='distance')

model_rfr=RandomForestRegressor(n_estimators=5,max_depth=5,\
max_leaf_nodes=10,min_samples_leaf=10)

bm=Ridge(alpha=3000)
model=AdaBoostRegressor(n_estimators=60,learning_rate=0.01,\
loss='square',base_estimator=bm)

bm=dtr(max_depth=3,min_samples_leaf=2,max_leaf_nodes=10,splitter='random')
model=BaggingRegressor(base_estimator=bm,n_estimators=500,bootstrap=True,\
bootstrap_features=1,max_samples=0.3,max_features=0.05)
model=AdaBoostRegressor(n_estimators=100,learning_rate=0.02,\
loss='square',base_estimator=bm)

model=Lars(normalize=False,n_nonzero_coefs=100)

model=BR(alpha_1=1e2,alpha_2=3e2,lambda_1=1e-9,lambda_2=1e-9,compute_score=False)


#nfit=10000;xfit=X[:nfit];yfit=Y[:nfit]
#xval=X[nfit:];yval=Y[nfit:]
model.fit(xfit,yfit.flatten())
probs=model.predict(xval)
probs1=(probs-probs.min())/(probs.max()-probs.min())
fpr, tpr, thresholds = roc_curve(yval, probs)
roc_auc=auc(fpr,tpr)
print(roc_auc)

#from sklearn.cross_validation import cross_val_score
#scores=cross_val_score(model,Xall,Yall.flatten())

probs=model.predict(xfit)
probs1=(probs-probs.min())/(probs.max()-probs.min())
fpr, tpr, thresholds = roc_curve(yfit, probs1)
roc_auc=auc(fpr,tpr)
print(roc_auc)

x=Xall;y=Yall;
model.fit(x,y.flatten())
probs=model.predict(x)
probs1=(probs-probs.min())/(probs.max()-probs.min())
fpr, tpr, thresholds = roc_curve(y, probs1)
roc_auc=auc(fpr,tpr)
print(roc_auc)

probs=model.predict(Xtest)
y_off=(probs-probs.min())/(probs.max()-probs.min())
result='"Idx","score"\n';
n=len(ytest)
for i in range(n):
    result=result+str(int(test_label[i]))+','+str(y_off[i])+'\n'
    
f=open('F:\下载中心\datacastle\网贷信誉预测\score.txt','w',encoding='utf-8')
f.write(result)
f.close()

import xlrd
import xlwt   
workbook=xlwt.Workbook()
sheetdata=result
ym=workbook.add_sheet('sheet1')       
(n,m)=shape(sheetdata)
for i in range(n):
    for j in range(m):
        ym.write(i,j,float(sheetdata[i,j]))
workbook.save('F:\下载中心\datacastle\网贷信誉预测\score0225.xls')


from sklearn import datasets
from sklearn.semi_supervised import LabelSpreading
model= LabelSpreading()
nfit=10000;
y1=Yall;y1[nfit:]=-1;
model.fit(x,y1.flatten())
probs=model.predict(xval)
probs1=(probs-probs.min())/(probs.max()-probs.min())
fpr, tpr, thresholds = roc_curve(yval, probs1)
roc_auc=auc(fpr,tpr)
print(roc_auc)

