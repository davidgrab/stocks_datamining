

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
import random
import time
from datetime import date


def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    return tickers
#simbs=save_sp500_tickers()


def arrange_data_frame(df):
    df.reset_index(level=1, inplace=True)
    df['symbol'] = df.index
    df['Year'] = df['date'].dt.year
    df['Month'] = df['date'].dt.month
    df['Day'] = df['date'].dt.day
    return df

with open("sp500tickers.pickle", "rb") as f:
    x = pickle.load(f)
symbols=[a.replace('\n','') for a in x ]
#symbols=random.sample(symbols,25)
#symbols=symbols[89:99]



column_names = ["symbol","peers","start date news","end date news","start date stock","end date stock","stock % of change in this period","company news"]
dfmain = pd.DataFrame(columns=column_names)


fin = finnhub_wrap.FinnHub('bqjdrbfrh5r89luquukg')
ts = TimeSeries(key='4YWV8Y5RPMBC8DSH')
tch = TechIndicators(key='4YWV8Y5RPMBC8DSH')
i=1

start_date_news='2019-10-01'
start_date_time_stock = date(2019, 10, 1)
start_date_time_stemp = time.mktime(start_date_time_stock.timetuple())


end_date_news='2020-09-20'
end_date_time_stock = date(2020, 9, 20)
end_date_time_stemp = time.mktime(end_date_time_stock.timetuple())


for sim in symbols:
    print(i)
    i+=1
    try:
        peers = fin.get_stock_peers(sim)
        price = fin.get_stock_price_matric(sim)
        last_quote=fin.get_stock_last_quote(sim)
        time.sleep(5)
        company_news_main=fin.get_company_news(sim,fromm=start_date_news,too=end_date_news)
        time.sleep(5)
        stock_candles=fin.get_stock_candles_by_timerange(sim, resolution="D",start=int(start_date_time_stemp),end=int(end_date_time_stemp))
        time.sleep(5)
        stock_change_in_this_period=(stock_candles['c'][-1]-stock_candles['c'][0])/stock_candles['c'][0]
        company_news=str([*[list(idx.values())[7] for idx in company_news_main]])
        #company_news_related = str([*[list(idx.values())[5] for idx in company_news_main]])

        df1=pd.DataFrame({"symbol":[sim],"peers":[peers],
                          "start date news":start_date_news,"end date news":end_date_news,
                          "start date stock":start_date_time_stock,"end date stock":end_date_time_stock,
                          "stock % of change in this period":stock_change_in_this_period
                             ,"company news": str(company_news)+" ".replace(u'\xa0', ' ') })
        dfmain=dfmain.append(df1, ignore_index = True)

    except:
        print('eror in ' + str(i) + "simbol")

dfmain.to_csv (r'export_dataframe.csv', index = False, header=True)
dfmain.to_pickle('export_dataframe_pickle_final1')

#dfmain = pd.read_pickle("export_dataframe_pickle")

#print("sdas")