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

#Takes the name of the stock we are trying to analyze
stockItself = input("Please enter the name of the stock (with the first letter capitalized and the rest lowercase): ")

#Takes the stock ticker of the stock we are trying to analyze
stockTicker = input("Please enter the corresponding stock ticker: ")

#Generates the Marketwatch url we will use to extract the needed article titles
url = "https://www.marketwatch.com/investing/stock/" + str(stockTicker)

#Takes all the html from the page
page = requests.get(url)
soup = BeautifulSoup(page.text, 'html.parser')

#Creating an empty index to store the titles for later
articleTitles = []

#Normally Stock Tickers are reffered to in capital form, so we will
#make all the letters capital
stockTickerCapitalized = stockTicker.upper()

#We are finding all the code inside the specfic class we specify
sub_links = soup.find_all("h3", {"class": "article__headline"})

#We extract only the text from this code and remove any extra whitespace
for links in sub_links:
    if links.a and (stockTickerCapitalized in links.a.text or stockItself in links.a.text):
        articleTitles.append(links.a.text.replace("\n", " ").strip())
        ''.join(articleTitles).split()

#By doing this we ensure there are no repeat articles
uniqueList = list(set(articleTitles))

#SIA is the SentimentIntensityAnalyzer package we imported
sia = SIA()

#Creating an empty index to store the values the Sentiment Analyzer returns
results = []

#Setting a variable to add 1 after every loop in order to print a certain index
#value of the sentiment analysis
sentimentCounter = 0

#Generates the sentiment for each news article title, adds it to the results
#list and then prints out the specific sentiment value with the headline it
#corresponds to
for line in uniqueList:
    pol_score = sia.polarity_scores(line)
    pol_score['headline'] = line
    results.append(pol_score)
    print()
    print(uniqueList[sentimentCounter])
    pprint(results[sentimentCounter], width=100)
    sentimentCounter += 1


#Makes a nice frame
df = pd.DataFrame.from_records(results)
df.head()

#Sets the label for future reference

df['label'] = 0

#Gives the current date and stores it in the csv file to see the progression
#and change of the news articles
date = datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y-%m%d %H:%M:%S')
tlDf = pd.DataFrame([[date, stockTicker]], columns=['headline', 'label'])
df2 = df[['headline', 'label']]
df2 = df2.append(tlDf)


#Adds all the article titles and their specific sentiment value into the specified
#csv file, but the sentiment is rounded to either -1, 0, or 1, deciding the
#overall sentiment
df2.to_csv('reddit_headlines_labels.csv', mode='a', encoding='utf-8', index=False)

#Prints out the added values of positive and negative

print()
print("Total value of Negative, Positive, and Neutral")
print(df.label.value_counts(normalize=True) * 100)

#Print a bar chart of data
fig, ax = plt.subplots(figsize=(8, 8))

counts = df.label.value_counts(normalize=True) * 100

sns.barplot(x=counts.index, y=counts, ax=ax)

ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
ax.set_ylabel("Percentage")

plt.show()

