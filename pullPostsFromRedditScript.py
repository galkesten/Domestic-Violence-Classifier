import pandas as pd
import requests #Pushshift accesses Reddit via an url so this is needed
import json #JSON manipulation
import csv #To Convert final table into a csv file to save to your machine
import time
import datetime

def getPushshiftData(before,sub):
    #Build URL
    url = 'https://api.pushshift.io/reddit/search/submission/?&size=1000&subreddit='+str(sub)+'&before='+str(before)
    #Print URL to show user
    print(url)
    #Request URL
    r = requests.get(url)
    #Load JSON data from webpage into data variable
    data = json.loads(r.text)
    #return the data element which contains all the submissions data
    # print("No Error")
    return data['data']



postList = []

for i in range(200,300,10):
    num = str(i)
    data = getPushshiftData(before=(num+'d'), sub='abusiverelationships')
    for dict in data:
        if 'selftext' in dict.keys():
            if len(dict['selftext']) < 1000 and dict['selftext'] != "[removed]" and dict['selftext'] != "[deleted]":
                postList.append(dict['selftext'])

df = pd.DataFrame(postList)
df.to_csv('abusiverelationships.csv', index=False)