from pandas_datareader import data, wb
import pandas as pd
import numpy as np
import datetime

start = datetime.datetime(2011,1,1)
end = datetime.datetime(2019,4,28)

tcs = data.DataReader("^NSEBANK",'yahoo',start,end)

tcs.to_csv("niftybank.csv",index=True,header=True)

print(tcs.head())

print(tcs.tail())
