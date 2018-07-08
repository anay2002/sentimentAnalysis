import requests
from bs4 import BeautifulSoup
from IPython import display
import math
from pprint import pprint
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
import tokenize
import time
import datetime

stockItself = input("Please enter the name of the stock (with the first letter capitalized and the rest lowercase): ")
stockTicker = input("Please enter the corresponding stock ticker: ")
url = "https://www.marketwatch.com/investing/stock/" + str(stockTicker)
page = requests.get(url)
soup = BeautifulSoup(page.text, 'html.parser')
articleTitles = []
articleLinks = []

stockTickerCapitalized = stockTicker.upper()

sub_links = soup.find_all("h3", {"class": "article__headline"})

div = soup.find_all("div", class_ = "article__content")
for links in div:
    if links.a and (stockTickerCapitalized in links.a.text or stockItself in links.a.text):
        articleTitles.append(links.a.text.replace("\n", " ").strip())
        ''.join(articleTitles).split()
        articleLinks.append(links.a['href'])
    
'''
for links2 in (sub_links2):
    print(links2.a)
    category_links = links2['href']
    if (stockTickerCapitalized or stockItself in links2.a):
        articleLinks.append(category_links)
for eachArticleLink in articleLinks:
    print(eachArticleLink)
'''

#sub_links = soup.find_all(class_="article-headline")
'''
for links in sub_links:
    if links.a and (stockTickerCapitalized in links.a.text or stockItself in links.a.text):
        #print(links.a.text.replace("\n", " ").strip())
        articleTitles.append(links.a.text.replace("\n", " ").strip())
        ''.join(articleTitles).split()
'''
uniqueList = list(set(articleTitles))
uniqueList2 = list(set(articleLinks))
#for links2 in sub_links2:
    #category_links = links['href']
    #articleLinks.append(category_links)
for eachArticleTitle in uniqueList:
    print(eachArticleTitle)
for eachArticleLink in uniqueList2:
    print(eachArticleLink)

sia = SIA()
results = []
length = len(uniqueList)

for line in uniqueList:
    pol_score = sia.polarity_scores(line)
    pol_score['headline'] = line
    results.append(pol_score)
    
#pprint(results[:length], width=100)


#Makes a nice frame
df = pd.DataFrame.from_records(results)
df.head()

# Decides if its positive or negative

df['label'] = 0
df.loc[df['compound'] > 0.2, 'label'] = 1
df.loc[df['compound'] < -0.2, 'label'] = -1
#df.head()

#df2 = df[['headline', 'label']]

date = datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y-%m%d %H:%M:%S')
tlDf = pd.DataFrame([[date, stockTicker]], columns=['headline', 'label'])
df2 = df[['headline', 'label']]
df2 = df2.append(tlDf)
df2.to_csv('reddit_headlines_labels.csv', mode='a', encoding='utf-8', index=False)
'''
#Prints out the different positive and negative headlines
print("Positive headlines:\n")
pprint(list(df[df['label'] == 1].headline)[:5], width=200)
print("\nNegative headlines:\n")
pprint(list(df[df['label'] == -1].headline)[:5], width=200)
'''

#Prints out the added values of positive and negative
#print(df.label.value_counts())

print(df.label.value_counts(normalize=True) * 100)

#Print a bar chart of data
fig, ax = plt.subplots(figsize=(8, 8))

counts = df.label.value_counts(normalize=True) * 100

sns.barplot(x=counts.index, y=counts, ax=ax)

ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
ax.set_ylabel("Percentage")

plt.show()

