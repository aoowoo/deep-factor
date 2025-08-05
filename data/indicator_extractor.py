import pandas as pd

df=pd.read_csv('./raw_data/10_Crypto_4h.csv',index_col='date',parse_dates=['date'])
print(df.iloc[0])


df['buy_ratio']=df['buy_amount']/df['amount']
symbols=list(set(df['symbol']))
print(symbols)
print(len(symbols))
print(df)

def merge_df(indicator_df,orgin_df,col_name,rank_norm=False):
    #标准化函数，去除缺失值和极值，并且进行标准化处理
    def norm_factor(factor,rank_norm=False):
        factor = factor.dropna(how='all')
        if rank_norm:
            factor = factor.rank(axis=1, method='min')  # method='min' 表示相同值取最小排名
        factor_clip = factor.apply(lambda x:x.clip(x.quantile(0.2), x.quantile(0.8)),axis=1)
        factor_norm = factor_clip.add(-factor_clip.mean(axis=1),axis ='index').div(factor_clip.std(axis=1),axis ='index')
        return factor_norm

    indicator_df[symbols]=norm_factor(indicator_df[symbols],rank_norm=rank_norm)
    indicator_df=indicator_df.melt(id_vars=['date'], var_name='symbol', value_name=col_name)
    df=pd.merge(orgin_df, indicator_df, on=['date', 'symbol'], how='left')
    print(df.isna().sum())
    return df

df_close = df.pivot_table(index="date", columns="symbol", values="close").reset_index()
momentum_3h = (df_close[symbols] - df_close[symbols].shift(3))/df_close[symbols].shift(3)
momentum_3h['date']=df_close['date']
df=merge_df(momentum_3h,df,'momentum_3h')
print(df)

momentum_24h = (df_close[symbols] - df_close[symbols].shift(24))/df_close[symbols].shift(24)
momentum_24h['date']=df_close['date']
df=merge_df(momentum_24h,df,'momentum_24h')
print(df)

df_volume = df.pivot_table(index="date", columns="symbol", values="volume").reset_index()
volume_indicator = (df_volume[symbols].rolling(24).mean() / df_volume[symbols].rolling(96).mean())
volume_indicator['date']=df_close['date']
df=merge_df(volume_indicator,df,'volume_indicator')
print(df)


########################################################################################
"""
这三个指标到股票上面用不了，干脆不用了，使用raw_volume_rank，vwap_deviation，price_change_rank
"""

raw_volume_rank_indicator = df_volume.copy()
raw_volume_rank_indicator['date']=df_close['date']
df=merge_df(raw_volume_rank_indicator,df,'raw_volume_rank_indicator',rank_norm=True)
print(df)

df_amount = df.pivot_table(index="date",columns="symbol",values="amount").reset_index()
vwap = (df_amount[symbols] * df_close[symbols]).rolling(window=24).sum() / df_amount[symbols].rolling(window=24).sum()
vwap_deviation_indicator = (df_close[symbols] - vwap) / df_close[symbols].rolling(window=24).std()
vwap_deviation_indicator['date']=df_close['date']
df=merge_df(vwap_deviation_indicator,df,'vwap_deviation_indicator')
print(df)

price_change_rank_indicator = (df_close[symbols] - df_close[symbols].shift(1)) / df_close[symbols].shift(1)
price_change_rank_indicator['date'] = df_close['date']
df = merge_df(price_change_rank_indicator, df, 'price_change_rank_indicator',rank_norm=True)
print(df)

df_open=df.pivot_table(index="date", columns="symbol", values="open").reset_index()
# 波动率因子
volatility_indicator = (df_close[symbols]/df_open[symbols]).rolling(24).std()
volatility_indicator['date']=df_close['date']
df=merge_df(volatility_indicator,df,'volatility_indicator')
print(df)
# 成交量与收盘价相关性因子
close_volume_corr_indicator=df_close[symbols].rolling(96).corr(df_volume[symbols])
close_volume_corr_indicator['date']=df_close['date']
df=merge_df(close_volume_corr_indicator,df,'close_volume_corr_indicator')
print(df)

df.to_csv('10_crypto_4h_indicators.csv',index=False)

df=pd.read_csv('10_crypto_4h_indicators.csv',parse_dates=['date'])
print(df)
df = df.loc[:,['date','symbol','close','momentum_3h','momentum_24h','volume_indicator','raw_volume_rank_indicator','vwap_deviation_indicator','price_change_rank_indicator','volatility_indicator','close_volume_corr_indicator']]
print(df)
df.to_csv('important_10_crypto_4h_indicators.csv',index=False)

df=pd.read_csv('important_10_crypto_4h_indicators.csv',parse_dates=['date'])
print(df)

df.set_index('date',inplace=True)
threshold_time_start = pd.to_datetime('2024-01-01 00:00:00')
threshold_time_end = pd.to_datetime('2024-12-31 00:00:00')
df=df[(df.index>=threshold_time_start) & (df.index<=threshold_time_end)]
print(df.head(10))

df.index=df.index.factorize()[0]

train_threshold_time_start = pd.to_datetime('2023-01-01 00:00:00')
train_threshold_time_end = pd.to_datetime('2023-12-31 00:00:00')
train=df[(df['date']>=train_threshold_time_start) & (df['date']<=train_threshold_time_end)]

print(len(df.loc[0,:]['symbol']))

print(df.loc[1:])