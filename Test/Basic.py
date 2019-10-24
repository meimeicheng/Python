#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 08:54:14 2019

@author: powerchip
"""
##https://www.tutorialspoint.com/python/index.htm
### Hello world
#hello = 'Hello'
#print(hello)


####常用資料處理套件  numpy
##tutorial => https://numpy.org/devdocs/user/quickstart.html
#import numpy as np
#a=np.array([10,20,30,40])   # array([10, 20, 30, 40])
#b=np.arange(4)              # array([0, 1, 2, 3])
#print(a)
#print(b)
#print(a*2)
#print(a-b)
#c=np.sin(a)
#for item in a:
#    print(item)
#    
#d=np.array([[1,2,3],[2,4,6]])
#for row in d:
#    print(row)
#for item in d.flat:
#    print(item)

####常用資料處理套件  Pandas 
##tutorial => https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html
import pandas as pd
import numpy as np

###Series : index , data
##s = pd.Series([1,3,6,np.nan,44,1])
##print(s)
#
#DataFrame 表格
#dates = pd.date_range('20160101',periods=6)
#df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=['a','b','c','d'])
#print(df)

df2 = pd.DataFrame({'A' : 1.,
                    'B' : pd.Timestamp('20130102'),
                    'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                    'D' : np.array([3] * 4,dtype='int32'),
                    'E' : pd.Categorical(["test","train","test","train"]),
                    'F' : 'foo'})
print(df2.E)
print(df2[0:3])
print(df2.loc[:,['A','B']]) 
print(df2.iloc[3,0]) #顯示row3,column0



