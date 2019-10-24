#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import chardet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import matplotlib.dates as mdates
#from matplotlib.finance import candlestick_ohlc
from mpl_finance import candlestick_ohlc
from matplotlib.dates import MONDAY
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator , MonthLocator
from matplotlib.dates import date2num
import codecs
#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')

#directoy_loc = 'D:/M/AI/CC/Data/'
directoy_loc = '/Users/powerchip/Projects/AI/Test/Data/'
test2 = 'u65e5\u671f'.decode('utf8')

def format_date(s):
    s = s.split('/')
    w = list(map(int,s))
    return datetime.datetime(w[0] + 1911,w[1],w[2])


def get_df(str_date,end_date,stock_id):   
    str_month = int(str_date[4:7])
    end_month = int(end_date[4:7])
    str_year = int(str_date[:4])
    end_year = int(end_date[:4])
   
    year_list = range(str_year,end_year+1)
    #month_list = range(str_month,end_month+1)
    df = pd.DataFrame()
    ini_flag = 0
    limit_month = 13
    for year in year_list:   
        if ini_flag==0 :
            monthcnt = str_month
        else :
            monthcnt = 1
            if year == end_year:
                limit_month = end_month

        while monthcnt < limit_month:
            file_loc = directoy_loc+stock_id+'/'+str(year)+'/'+str(year)+str(monthcnt).zfill(2)+'.csv'
            with open(file_loc, 'rb') as f:
                result = chardet.detect(f.readline())  # or readline if the file is large         
            df = df.append(pd.read_csv(file_loc,sep=',',encoding=result['encoding']),ignore_index=True)
            #df = df.append(pd.read_csv(file_loc,sep=',',encoding='utf8'),ignore_index=True)
            monthcnt = monthcnt+1
        ini_flag = ini_flag+1
    return df



df_all_data =pd.DataFrame() 
#set date range and stockid
df_all_data = get_df('201801','201910','3443')
#print(df_all_data.columns.values)
#print(df_all_data[u'\u65e5\u671f'])
#print(df_all_data[u'日期'])
df_all_data[u'日期'] = df_all_data[u'日期'].apply(lambda x:format_date(x))
#df_all_data.loc[:,[u'日期']] = df_all_data.loc[:,[u'日期']] .apply(lambda x:format_date(x))
#print(df_all_data)
#print(df_all_data.info())


stock = df_all_data.set_index([u'日期'],drop=False, append=False, inplace=False, verify_integrity=False)

print('=========stock===========')
print(stock)
stock = stock.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',',''), errors='coerce'))
stock = stock[::-1]
stock.index.rename('date', inplace=True)

stock.rename(columns={u'開盤價':u'open', u'最高價':u'high', u'最低價':u'low', u'收盤價':u'close'}, inplace=True)
print(stock.info())
print(stock)
#stock_data['成交股數'].plot(grid=True)


#繪製K線圖
def pandas_candlestick_ohlc(stock_data, otherseries=None):
   
   # 設置繪圖參數，主要是坐標軸
   mondays = WeekdayLocator(MONDAY)
   alldays = DayLocator()
   #alldays.MAXTICKS = 10000
   dayFormatter = DateFormatter('%d')
   
   fig, ax = plt.subplots(figsize=(50,80))
   fig.subplots_adjust(bottom=0.2)

   if stock_data.index[0] - stock_data.index[-1] < pd.Timedelta('300 days'):
       weekFormatter = DateFormatter('%b %d')
       ax.xaxis.set_major_locator(mondays)
       ax.xaxis.set_minor_locator(alldays)
       ax.xaxis.set_major_formatter(weekFormatter)
   else:
       monthFormatter = DateFormatter('%Y %b')
       ax.xaxis.set_major_locator(MonthLocator(range(1, 13), bymonthday=1, interval=1))
       ax.xaxis.set_major_formatter(monthFormatter)
   ax.grid(True)

   # 創建K線圖
   stock_array = np.array(stock_data.reset_index()[['date','open','high','low','close']])
   stock_array[:,0] = date2num(stock_array[:,0])
   candlestick_ohlc(ax, stock_array, colorup = "red", colordown="green", width=0.6)

   # 可同時繪製其他折線圖
   if otherseries is not None:
       for each in otherseries:
           plt.plot(stock_data[each], label=each)
   plt.legend()
   ax.xaxis_date()
   ax.autoscale_view()
   #Mei : 2017/12/18 升級pandas to 0.21.1 解決與matplotlib不相容問題
   plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
   plt.show()

#計算ma5/ma20
#stock['ma5'] =  pd.rolling_mean(stock['close'], 5)
#stock['ma20'] = pd.rolling_mean(stock['close'], 20)

stock['ma5'] =  stock['close'].rolling(5).mean()
stock['ma20'] = stock['close'].rolling(20).mean()

 
#stock['ma5'] = np.round(stock["close"].rolling(window = 5, center = False).mean(), 2)
#stock['ma20'] =np.round(stock["close"].rolling(window = 20, center = False).mean(), 2)
#print(stock)
#print(stock.info())
#pandas_candlestick_ohlc(stock)
pandas_candlestick_ohlc(stock,['ma5','ma20'])
