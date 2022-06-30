#Import Fake News Dataset 

from operator import index
import pandas as pd 

#Reading False Dataset 
dataset_false = pd.read_csv('./Input/Fake.csv')
#Reading True Dataset 
dataset_true = pd.read_csv('./Input/True.csv')

#Dataset contains 23481 rows and 4 columns. 
#Columns Present in the Dataset are as follows : 
   #1. title : represent title of the news ( short subject or short summary of the news)
   #2. text   : text of the news 
   #3. subject : 'News', 'politics', 'Government News', 'left-news', 'US_News','Middle-east'
   #4. date : date of news 

#Concatenate both datasets - but first create a flag to represent which is a fake news vs true news 
dataset_false['label'] = 0
dataset_true['label']  = 1

# Label in the final dataset will represent if it is a true news or a fake news 
# label = 1 represents it is a true news , label = 0 represents it is a fake news
dataset_final = pd.concat([dataset_true,dataset_false],axis=0)

# To get the counts of label
dataset_final['label'].value_counts()
# 0    23481
# 1    21417







