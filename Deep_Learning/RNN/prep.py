import pandas as pd
import string
import unicodedata
from collections import Counter
import re
import json

allowed_characters = string.ascii_letters + ".,;:'"


def normalize_unicode_to_ascii(text):
    #separate accents -> keep only ascii characters -> convert back to string
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")

def tokenize(text):
    return text.lower().split()  # Tokenize by splitting on spaces

def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters and numbers
    text = text.lower()  # Convert to lowercase
    text = " ".join(text.split())  # Remove extra spaces
    return text

pd.set_option('display.max_rows',20)
pd.set_option('display.max_columns',20)

df = pd.read_csv('website_classification.csv')
df = df[(df.Category != 'Adult')]
df = df[['cleaned_website_text','Category']]

df['text'] = df['cleaned_website_text']
df.drop(columns = 'cleaned_website_text',inplace = True)

df['text'] = df['text'].apply(normalize_unicode_to_ascii)
df['text'] = df['text'].apply(clean_text)

df.to_csv('preped.csv')

#Get text from dataframe
text = [row['text'] for _,row in df.iterrows()]

#split the words in every item
tokens = [tokenize(txt) for txt in text]

#get every word in the list
all_words = [word for sentence in tokens for word in sentence]

#count the occurencies of every word
word_count = Counter(all_words)
word_count = Counter({key : value for key,value in word_count.items()})

#Get all unique words with keys method
unique = list(word_count.keys())

#build the vocabulary for the rnn
vocab = {word: i+2 for i, word in enumerate(unique)}

print(df)
#Add special tokens

vocab["<PAD>"] = 0  #Padding token
vocab["<UNK>"] = 1
with open('vocab.json','w') as f:
    json.dump(vocab, f)




