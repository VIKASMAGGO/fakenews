import Dataset 
import nltk,re,string,unicodedata
from nltk import pos_tag
from nltk.corpus import wordnet,stopwords
from nltk.stem.porter import PorterStemmer
import re

ps = PorterStemmer()

# Dataset final contains is a superset of true and false news 
# Remove columns not required 

#vview columns again  
dataset_final.columns

# as we are planning to run NLP so we do not need date , and subject 
dataset_final = dataset_final.drop(["subject","date"],axis=1)

#combine the title and text as they both represents same information 
# feature engineering - created extra column using additional columns 

dataset_final["full_text"] = dataset_final["title"] + str(" ") + dataset_final["text"]
#remove the columns that are not required now 
dataset_final = dataset_final.drop(["title","text"],axis=1)

#count of missing data in the dataframe 
count_of_missing = dataset_final['full_text'].isna().sum()

if count_of_missing > 0 : 
    dataset_final=dataset_final.dropna()

#Removing stopwords: 
# There are a lot of words that add no value to any text no matter the data. 
# For example, “I”, “a”, “am”, etc. 
# These words have no informational value and hence can be removed to reduce the size of our corpus 

#Exracting Stop Words and Punctuations that needed to be removed from the full_text column 
nltk.download('stopwords')
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)
    
#Stemming the words: 
# Stemming and Lemmatization are the techniques to reduce the words to their stems or roots. 
# The main advantage of this step is to reduce the size of the vocabulary. 
# For example, words like Play, Playing, Played will be reduced to “Play”. 
# More Examples :
# Stay, Stays, Staying, Stayed — -> Stay
# House, Houses, Housing — -> House

def cleaning_data(text) :
    """ Apply all cleaning functions here """
    #convert text into lower case 
    text = text.lower()
    #regex to remove any numbers or special characters from the text 
    text = re.sub('[^a-zA-Z]','',text)

    #split the data and make token 
    token = text.split() 
    
    #Lemmatize the words and remove stop words as explained above 
    text = [ps.stem(word)  for word in token if not word in stop] 
    cleaned_news = ' '.join(text)
    
    return cleaned_news

    
dataset_final['full_text_final']=dataset_final['full_text'].apply(cleaning_data)



