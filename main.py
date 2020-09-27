
import os
os.chdir(r'C:\Users\USER\PycharmProjects\datamining\project2')

#Loading NLTK
import nltk
#nltk.download('averaged_perceptron_tagger')
import matplotlib.pyplot as plt
import re
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import numpy as np
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import pickle
from financialmodelingprep import FinancialModelingPrep
import finnhub_wrap
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.sectorperformance import SectorPerformances
from alpha_vantage.cryptocurrencies import CryptoCurrencies
import requests
import bs4 as bs
import csv
import time
import pandas as pd



##fuction that gets a list of tokens and returs b's answer
def b_function(tokenized_word):

    #Frequency Distribution
    from nltk.probability import FreqDist
    fdist = FreqDist(tokenized_word)

    #Ranking tokens by frequency
    ranks = []
    freqs = []
    for rank, word in enumerate(fdist):
        ranks.append(rank + 1)
        freqs.append(fdist[word])

    # log log plot of frequency vs rank
    plt.figure(figsize=[10,10])
    plt.loglog(ranks,freqs)
    plt.title('log log plot of frequency vs rank',fontsize = 16, fontweight = 'bold')
    plt.ylabel('frequency(f)',fontsize = 14, fontweight = 'bold')
    plt.xlabel('rank(r)',fontsize = 14, fontweight = 'bold')
    plt.grid(True)
    plt.show()



    #getting a list of 20 most common tokens and their frequency
    vocabulary1 = fdist.most_common(20)
    print('list of 20 most common tokens and their frequency ')
    print(vocabulary1)

dfmain = pd.read_pickle("export_dataframe_pickle_final1")
dfmain=dfmain.drop(["start date news","end date news","start date stock","end date stock"], axis=1)
profil = pd.read_csv("profiles.csv")
result = pd.merge(dfmain, profil, on='symbol')
#for ind in result.index:
#    result["company news"][ind] = str(result["company news"][ind]).replace(result["symbol"][ind], " ")

grouped = result.groupby('profile.sector').apply(lambda x: x.sum())

for ind in grouped.index:
#for ind in range(1):

    ###readind the THE JEWISH STATE book
    text=str(grouped["company news"][ind])
    text=text.lower()
    ##removing two problematoc sings
    text=text.replace("*", " ")
    text = text.replace('"', " ")
    text=text.replace("_", " ")
    text = text.replace("  ", " ")

    text_raw=text

    ###removing all pancthuation
    text = re.sub('[^A-Za-z]+', ' ', text)
    text=text.replace(" nyse ",' ')
    text=text.replace(" inc ",' ')
    text=text.replace(" feb ",' ')
    text = text.replace(" quarter ", ' ')
    text = text.replace(" quarters ", ' ')
    text = text.replace(" quartered ", ' ')
    text = text.replace(" quarterly ", ' ')
    text = text.replace(" earnings ", ' ')
    text = text.replace(" fourth ", ' ')
    text = text.replace(" 4th ", ' ')
    text = text.replace(" xa ", ' ')
    text = text.replace("\\xa0", " ")
    text = text.replace(u'\xa0', ' ')
    text = text.rstrip('\r\n')
# senternces tokenization of text
    tokenized_text = sent_tokenize(text)
    # Word tokenizer breaks text paragraph into words.
    tokenized_word = word_tokenize(text.replace(' xa ',' '))
    #####b_function(tokenized_word)

    # c
    # creating a set if of words in english commonly regarded as stopwords
    stop_words = set(stopwords.words("english"))
    # print("This is the set of stopwords we will take in consideration in our analysis:")
    stop_words = list(stop_words)
    # print(stop_words)

    # Removing Stopwords
    filtered_sent = []
    for w in tokenized_word:
        w = w.lower()
        if w not in stop_words:
            filtered_sent.append(w)

    # print("Tokenized Sentence:",tokenized_word)
    # print("Filterd Sentence:",filtered_sent)

    ###call b function for tokenized words without Stopwords
    ######b_function(filtered_sent)
    # Stemming
    ps = PorterStemmer()
    stemmed_words = []
    for w in filtered_sent:
        stemmed_words.append(ps.stem(w))

    # print("Filtered Sentence:",filtered_sent)
    # print("Stemmed Sentence:",stemmed_words)

    ###call b function for tokenized stemmed words without Stopwords
    ######b_function(stemmed_words)
    #tokenized_word_raw = word_tokenize(text_raw)
    ###start by using tokenized word list of original text with ","  and "."
    a = nltk.pos_tag(tokenized_word)
    noun_adj_prasese = []

    ##main loop that runs on every token
    i = 0
    while i < (len(a)):
        tag = a[i][1]
        chack = 0  # if 1 =adj happaend if 2 then adj+noun happned

        ##if adj chack next tokens and move chack possition to 1
        if tag in ('JJR', 'JJS', 'JJ'):
            prash = []  # new prase
            prash.append(a[i][0])
            chack = 1  # updating chack condition
            for j in range(i + 1, len(a)):  # runing on next tokens
                ##appaned if another adj and update i
                if a[j][1] in ('JJR', 'JJS', 'JJ') and chack == 1:
                    prash.append(a[j][0])
                    i = j
                ##appaned if another noun and update i
                if a[j][1] in ('NN', 'NNS', 'NNP', 'NNPS') and chack == 2:
                    prash.append(a[j][0])
                    i = j
                ##appaned if noun after adj and update i and chack condition
                if a[j][1] in ('NN', 'NNS', 'NNP', 'NNPS') and chack == 1:
                    prash.append(a[j][0])
                    chack = 2
                    i = j
                ##appaned prash to final list if no fore nouns ,  update chack condition and brake to start new prash
                ##here we dont update i to make sure we dont loss a new prash
                if a[j][1] not in ('NN', 'NNS', 'NNP', 'NNPS') and chack == 2:
                    chack = 0
                    str1 = ' '.join(prash)
                    noun_adj_prasese.append(str1)
                    prash = []
                    break
                ##here we dont update i to make sure we dont loss a new prash and go back to main loop
                if a[j][1] not in ('NN', 'NNS', 'NNP', 'NNPS', 'JJR', 'JJS', 'JJ') and chack == 1:
                    chack = 0
                    prash = []
                    break

        i += 1

    # print("adj+noun phrases:",noun_adj_prasese)
    # print("worng POS tags:",a)

    ###call b function for tokenized adj+noun phrases
    ######b_function(noun_adj_prasese)
    text_noun_adj_prasese = ' '.join(noun_adj_prasese)

    # Generate a word cloud image
    mask = np.array(Image.open("israel.png"))
    wordcloud_ita = WordCloud(background_color="white", max_words=300, mask=mask).generate(text_noun_adj_prasese)

    # create coloring from image
    image_colors = ImageColorGenerator(mask)
    plt.figure(figsize=[15, 15])
    plt.imshow(wordcloud_ita.recolor(color_func=image_colors), interpolation="bilinear")
    plt.axis("off")
    # store to file
    plt.savefig("text_noun_adj_prasese_" + str(ind) + ".png", format="png")
    plt.show()
    # g Create a Tag cloud (word cloud) of proper nouns (NNP, NNPS).
    # creating text of proper nouns
    proper_nouns = []
    for i in range(len(a)):
        if a[i][1] in ('NNP', 'NNPS'):
            proper_nouns.append(a[i][0])
    text_proper_nouns = ' '.join(proper_nouns)

    # Generate a word cloud image
    mask = np.array(Image.open("israel.png"))
    wordcloud_ita = WordCloud(background_color="white", max_words=300, mask=mask).generate(text_proper_nouns)

    # create coloring from image
    image_colors = ImageColorGenerator(mask)
    plt.figure(figsize=[15, 15])
    plt.imshow(wordcloud_ita.recolor(color_func=image_colors), interpolation="bilinear")
    plt.axis("off")
    # store to file
    plt.savefig("proper_nouns_"+str(ind)+".png", format="png")
    plt.show()

    adj = []
    for i in range(len(a)):
        if a[i][1] in ('JJR', 'JJS', 'JJ'):
            adj.append(a[i][0])
    text_adj = ' '.join(adj)
    text_noun_adj_prasese = ' '.join(noun_adj_prasese)

    # Generate a word cloud image
    mask = np.array(Image.open("israel.png"))
    wordcloud_ita = WordCloud(background_color="white", max_words=300, mask=mask).generate(text_adj)

    # create coloring from image
    image_colors = ImageColorGenerator(mask)
    plt.figure(figsize=[15, 15])
    plt.imshow(wordcloud_ita.recolor(color_func=image_colors), interpolation="bilinear")
    plt.axis("off")
    # store to file
    plt.savefig("adj_"+str(ind)+".png", format="png")
    plt.show()