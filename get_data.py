import requests
import pandas as pd

def get_ticker(stri):
    url = f'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={stri}&apikey=DJ5ATAVVQLJE2IFG'
    r = requests.get(url)
    data = r.json()
    return str(pd.DataFrame(data["bestMatches"]).iloc[0][0])

def get_data(ticker="TIME_SERIES_INTRADAY",sym="IBM"):
    text="TIME_SERIES_INTRADAY"
    url = f"https://www.alphavantage.co/query?function={ticker}&symbol={sym}&interval=5min&apikey=DJ5ATAVVQLJE2IFG"
    r = requests.get(url)
    data = r.json()
    return data

def clean_col_names(df):
    for i in df.columns:
        df.rename(columns={i:i.split(sep=" ")[1].upper()}, inplace=True)
        df.index = pd.to_datetime(df.index)
        for i in df.columns:
            df[i]=pd.to_numeric(df[i])
    return df

def get_details(data):
    print(f'The {data["Meta Data"].get("1. Information")} with stock = {data["Meta Data"].get("2. Symbol")} was last refreshed at {data["Meta Data"].get("3. Last Refreshed")} with timezone = {data["Meta Data"].get("5. Time Zone")}')

data=get_data(ticker="TIME_SERIES_DAILY_ADJUSTED",sym=get_ticker(input()))
to_get=list(data)[1]

df = pd.DataFrame(data[to_get]).T

clean_col_names(df)
get_details(data)