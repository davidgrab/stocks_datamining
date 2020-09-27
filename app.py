
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.graph_objs as go

import io
import flask
import pandas as pd
import time
import os
import pickle
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
from wordcloud import WordCloud, STOPWORDS
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot


def plotly_wordcloud(text):
    mask = np.array(Image.open("israel.png"))
    wc = WordCloud(background_color="white", max_words=300, mask=mask).generate(text)

    word_list = []
    freq_list = []
    fontsize_list = []
    position_list = []
    orientation_list = []
    color_list = []

    for (word, freq), fontsize, position, orientation, color in wc.layout_:
        word_list.append(word)
        freq_list.append(freq)
        fontsize_list.append(fontsize)
        position_list.append(position)
        orientation_list.append(orientation)
        color_list.append(color)

    # get the positions
    x = []
    y = []
    for i in position_list:
        x.append(i[0])
        y.append(i[1])

    # get the relative occurence frequencies
    new_freq_list = []
    for i in freq_list:
        new_freq_list.append(i * 100)
    new_freq_list

    trace = go.Scatter(x=x,
                       y=y,
                       textfont=dict(size=new_freq_list,
                                     color=color_list),
                       hoverinfo='text',
                       hovertext=['{0}{1}'.format(w, f) for w, f in zip(word_list, freq_list)],
                       mode="text",
                       text=word_list
                       )

    layout = go.Layout(
        xaxis=dict(showgrid=False,
                   showticklabels=False,
                   zeroline=False,
                   automargin=True),
        yaxis=dict(showgrid=False,
                   showticklabels=False,
                   zeroline=False,
                   automargin=True)
    )

    fig = go.Figure(data=[trace], layout=layout)

    return fig

def show_word_cloud(company_data):
    text=str(company_data["company news"].tolist())
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

    ps = PorterStemmer()
    stemmed_words = []
    for w in filtered_sent:
        stemmed_words.append(ps.stem(w))

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

    text_noun_adj_prasese = ' '.join(noun_adj_prasese)

    # Generate a word cloud image
    return(plotly_wordcloud(text_noun_adj_prasese))



# Step 1. Launch the application
app = dash.Dash()

# Step 2. Import the dataset
st = pd.read_pickle("export_dataframe_pickle_final1")

with open("sp500tickers.pickle", "rb") as f:
    x = pickle.load(f)
symbols=[a.replace('\n','') for a in x ]

# dropdown options
opts = [{'label': i, 'value': i} for i in symbols]


# Step 3. Create a plotly figure
company_data = st[st['symbol'] == 'A']
fig = show_word_cloud(company_data)

# Step 4. Create a Dash layout
app.layout = html.Div([
    # a header and a paragraph
    html.Div([
        html.H1("news word cloud of s&p 500 stocks"),
        html.P("by Michael Ben Izhak and David Grabois")
    ],
        style={'padding': '50px',
               'backgroundColor': '#3aaab2'}),
    # adding a plot
    dcc.Graph(id='plot', figure=fig),
    # dropdown
    html.P([
        html.Label("Choose a stock symbol"),
        dcc.Dropdown(id='opt', options=opts,
                     value=symbols[1])
    ]),

])


# Step 5. Add callback functions
@app.callback(Output('plot', 'figure'),
              [Input('opt', 'value')])
def update_figure(input1):
    # filtering the data
    fig = show_word_cloud(st[st['symbol'] == input1])
    return fig


# Step 6. Add the server clause
if __name__ == '__main__':
    app.run_server(debug=True)