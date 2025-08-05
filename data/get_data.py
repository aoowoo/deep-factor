from datetime import date,datetime
import time
import pandas as pd
import requests, zipfile, io
import os
import talib as ta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


retry_strategy = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("https://", adapter)
session.mount("http://", adapter)

# BTC ETH XRP SOL BNB DOGE ADA TRX LINK AVAX


symbols=['BTCUSDT','ETHUSDT','XRPUSDT','SOLUSDT','BNBUSDT','DOGEUSDT','ADAUSDT','TRXUSDT','LINKUSDT','AVAXUSDT']
symbols2=['LTCUSDT','XLMUSDT','XMRUSDT','EOSUSDT','BCHUSDT','IOTAUSDT','ETCUSDT','ALGOUSDT','VETUSDT','ATOMUSDT']

# symbols=symbols+symbols2
print(len(set(symbols)))
print(len(symbols))
temp=['BTCUSDT', 'BCHUSDT', 'VETUSDT', 'XLMUSDT', 'ETHUSDT', 'LTCUSDT', 'AVAXUSDT', 'ADAUSDT', 'ETCUSDT', 'BNBUSDT', 'SOLUSDT', 'EOSUSDT', 'XMRUSDT', 'TRXUSDT', 'LINKUSDT', 'ATOMUSDT', 'DOGEUSDT', 'XRPUSDT', 'ALGOUSDT']

print(list(set(symbols)^set(temp)))

# Info = requests.get('https://fapi.binance.com/fapi/v1/exchangeInfo')
#
# all_symbols = [s['symbol'] for s in Info.json()['symbols']]
# all_symbols = list(filter(lambda x: x[-4:] == 'USDT', [s.split('_')[0] for s in symbols]))
#
# print(len(set(all_symbols)))
# print(Counter(all_symbols))

# #获取任意周期K线的函数
def GetKlines(symbol='BTCUSDT',start='2025-01-01',end='2024-12-31',period='1d',base='api',v = 'v3'):
    Klines = []
    start_time = int(time.mktime(datetime.strptime(start, "%Y-%m-%d").timetuple()))*1000 + 8*60*60*1000
    end_time =  min(int(time.mktime(datetime.strptime(end, "%Y-%m-%d").timetuple()))*1000 + 8*60*60*1000,time.time()*1000)
    intervel_map = {'m':60*1000,'h':60*60*1000,'d':24*60*60*1000}
    check=True
    while start_time < end_time:
        mid_time = start_time+1000*int(period[:-1])*intervel_map[period[-1]]
        url = 'https://'+base+'.binance.com/'+base+'/'+v+'/klines?symbol=%s&interval=%s&startTime=%s&endTime=%s&limit=1000'%(symbol,period,start_time,mid_time)
        try:
            res = session.get(url)
            res_list = res.json()
            if type(res_list) == list and len(res_list) > 0:
                start_time = res_list[-1][0] + int(period[:-1]) * intervel_map[period[-1]]
                Klines += res_list
            if type(res_list) == list and len(res_list) == 0:
                start_time = start_time + 1000 * int(period[:-1]) * intervel_map[period[-1]]
            if mid_time >= end_time:
                break
            if type(res_list) != list:
                check = False
                print(f"{symbol} is None")
                break
            time.sleep(0.05)
        except requests.exceptions.SSLError as e:
            print(f"SSLError occurred: {e}. Retrying...")
            time.sleep(0.5)  # 等待0.5秒后重试
            continue
    df = pd.DataFrame(Klines,columns=['time','open','high','low','close','amount','end_time','volume','count','buy_amount','buy_volume','null']).astype('float')
    df['symbol']=pd.Series(data=[symbol for _ in range(len(df))])
    df['time']=pd.to_datetime(df['time'],unit='ms')
    df.set_index('time',inplace=True)
    df.index.name='date'
    return df if check else None

def get_ema_judge(symbol='BTCUSDT',period='1d',start='2019-01-01',end='2025-12-31',ema_period=200):
    df_s=GetKlines(symbol=symbol,period=period,start=start,end=end)
    df_s['symbol']=pd.Series(data=[symbol for _ in range(len(df_s))])
    df_s['date']=df_s.index
    print(f"symbol {symbol} 空数量 {df_s.isnull().sum().sum()}")
    output_dir='./'
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f'{symbol}_{period}_ema{ema_period}.csv')
    col_name='ema'+str(ema_period)

    # 计算 EMA
    df_s[col_name] = ta.EMA(df_s['close'],timeperiod=ema_period)
    # 删除 EMA 为空值的行
    df_s.dropna(subset=[col_name], inplace=True)

    df_s=df_s.loc[:,['date','close',col_name]]
    # 保存到 CSV 文件
    df_s.to_csv(file_path, index=False)

"""
获取k线数据，单独保存为一个文件，然后把symbols里面的所有数据合成起来保存文件
"""
def GetPeriodKlines(period,filename,start='2021-01-01',end='2024-12-31'):
    df_all=None
    for symbol in symbols:
        df_s = GetKlines(symbol=symbol,period=period,start=start,end=end)
        print(f"symbol {symbol} 空数量 {df_s.isnull().sum().sum() if df_s is not None else -1}")
        output_dir='data'
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f'{symbol}_{period}.csv')
        df_s.to_csv(file_path, index=True)
        if df_all is None:
            df_all=df_s
        else:
            df_all=pd.concat([df_all,df_s],axis=0).sort_values(by=['date','symbol'])

    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, f'{filename}_{period}.csv')
    df_all.to_csv(file_path, index=True)



GetPeriodKlines('4h',filename='10_crypto',start='2019-10-01',end='2024-12-31')