# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 21:23:41 2017

@author: cheng.f.m
"""

import requests
import json
import csv
import time, datetime,os
from bs4 import BeautifulSoup
import codecs
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

dt = datetime.datetime.now()
dt.year
dt.month

#依傳入年份,月份,股票id,設定網址,開始爬蟲
def get_webmsg (year, month, stock_id):
    date = str (year) + "{0:0=2d}".format(month) +'01' 
    sid = str(stock_id)
    url_twse = 'http://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date='+date+'&stockNo='+sid
    res =requests.post(url_twse,)
    soup = BeautifulSoup(res.text , 'html.parser')
    #將資料放入json
    smt = json.loads(soup.text)   
    return smt

#
def write_csv(stock_id,directory,filename,smt) :
    writefile = directory + filename               
    #outputFile = open(writefile,'w',newline='', encoding='utf-8')
    #outputWriter = csv.writer(outputFile)
#    head = ''.join(smt['title'].split())
#    a = [head,""]
#    outputWriter.writerow(a)
    #outputWriter.writerow(smt['fields'])
    #for data in (smt['data']):
    #    outputWriter.writerow(data)

    #outputFile.close()
    
    with open(writefile, 'w') as csvFile:
        csvFile.write(codecs.BOM_UTF8)
        writer = csv.writer(csvFile)
        writer.writerow(smt['fields'])
        for data in (smt['data']):
            writer.writerow(data)

    csvFile.close()


#建立寫檔資料夾
def makedirs (year, month, stock_id):
    sid = str(stock_id)
    yy      = str(year)
#    mm       = str(month)
    directory = '/Users/powerchip/Projects/AI/Test/Data'+'/'+sid +'/'+ yy
    if not os.path.isdir(directory):
        os.makedirs (directory)  
        
#設定要爬的股票id 
# Mei : 可以用 id_list =['2002','2888','2882'] 一次多個股票代碼
id_list = ['3443'] #inout the stock IDs
now = datetime.datetime.now()
year_list = range (2018,now.year+1) #設定年份區間
month_list = range(1,13)  # 12個月


for stock_id in id_list:
    for year in year_list:
        for month in month_list:
            if (dt.year == year and month > dt.month) :break  
            sid = str(stock_id)
            yy  = str(year)
            mm  = month
            directory = '/Users/powerchip/Projects/AI/Test/Data'+'/'+sid +'/'+yy +'/'       
            filename = str(yy)+str("%02d"%mm)+'.csv'          
            smt = get_webmsg(year ,month, stock_id)  
            #print(smt) 
            makedirs (year, month, stock_id)                  
            write_csv (stock_id,directory, filename, smt)
            time.sleep(5)