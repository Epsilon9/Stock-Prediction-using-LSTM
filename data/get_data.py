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