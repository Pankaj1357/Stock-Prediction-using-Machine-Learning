import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


c_name = "nifty"

def get_rsi(sdata,m,mem):
    neg = 0
    pos = 0
    RS = 0
    RSI = 100

    upcloses = 0
    downcloses = 0

    n = m
    k = m-1

    for p in range(mem):
       diff = sdata[n,3] - sdata[k,3]
       if (diff>=0):
        upcloses = upcloses + diff
        pos = pos+1
       else:
        downcloses = downcloses + diff
        neg = neg+1
    
       n = n-1
       k = k-1
 
    downcloses = -downcloses
    if(neg == 0):
        return 100
    else:
        RS = (upcloses*neg)/(downcloses*pos)

    
    RSI = 100 - (100/(1+RS))

    return RSI
    

def get_mfi(sdata,m,mem):
    neg = 0
    pos = 0
    MFR = 0
    MFI = 100

    pmflow = 0
    nmflow = 0

    n = m
    k = m-1

    for p in range(mem):

       typ_pricec = (sdata[n,0] + sdata[n,1] + sdata[n,3])/3
       typ_pricep = (sdata[k,0] + sdata[k,1] + sdata[k,3])/3
     
      # print(typ_price,sdata[n,0],sdata[n,1],sdata[n,3])

       if (typ_pricec>=typ_pricep):
         pmflow = pmflow + ((sdata[n,4])*(typ_pricec))
         pos = pos+1
       else:
         nmflow = nmflow + ((sdata[n,4])*(typ_pricec))
         neg = neg+1
       
       n = n-1
       k = k-1
    
    if(neg == 0):
        return 100
    else:
        MFR = pmflow/nmflow

    MFI = 100 - (100/(1+MFR))

    return MFI

def get_ema(sdata,m,mem,EMAp):
   
   EMA = sdata[m,3]*(2/(1+mem)) + (1-(2/(1+mem)))*EMAp 
   print("EMA",EMA)

   return EMA

def get_so(sdata,m,mem):
   
   SO = ((sdata[m,3]-sdata[m,1])/(sdata[m,0]-sdata[m,1]))*100
   print("SO",SO)
   return SO



with open('./'+ c_name +'.csv', 'r') as csvFile:             #Open the CSV file containing traffic data
 data = list(csv.reader(csvFile,delimiter=','))

memory = 14

print(data[0])

sdat = pd.DataFrame(data[1:])
sdat2 = sdat.drop([0,6],axis = 1)
sdat3 = np.array(sdat2,dtype=np.float32)


print(sdat3[0])

df1 = pd.DataFrame(columns=['Open','High','Low','Close','Volume','RSI','MFI','EMA','SO','CloseNext'])

df2 = pd.DataFrame(columns=['Close','RSI','MFI','EMA','SO','CloseNext'])

arr = np.array(df1.values)


print("Printing j\n")

EMAp = 0
acc = 0

for i in range(memory):
    acc = acc + sdat3[i,3]


EMAp = acc / memory
    



#for i in range(len(sdat3) -memory):
for i in range(len(sdat3)-1 -memory):
    j = i + memory

    RSI = get_rsi(sdat3,j,memory)
    print("RSI:",RSI)

    MFI = get_mfi(sdat3,j,memory)
    print("MFI:",MFI)

    EMA = get_ema(sdat3,j,memory,EMAp)
    EMAp = EMA

    SO = get_so(sdat3,j,memory)

    N_close = sdat3[j+1,3]

    rec1 = [sdat3[j,2],sdat3[j,0],sdat3[j,1],sdat3[j,3],sdat3[j,4],RSI,MFI,EMA,SO,N_close]
    rec2 = [sdat3[j,3],RSI,MFI,EMA,SO,N_close]

    if(sdat3[j,4]!=0):
        d1 = {"Open":sdat3[j,2],"High":sdat3[j,0],"Low":sdat3[j,1],"Close":sdat3[j,3],"Volume":sdat3[j,4],"RSI":RSI,"MFI":MFI,"EMA":EMA,"SO":SO,"CloseNext":N_close}
        df1.loc[i] = rec1
        df2.loc[i] = rec2



df1.to_csv(c_name + "_l.csv",index=True,header=True)
df2.to_csv(c_name +"_s.csv",index=True,header=True)


    #rec4 = pd.DataFrame({'Close':sdat3[j,3],'RSI':RSI,'MFI':MFI,'EMA':EMA,'SO':SO,'CloseNext':N_close})

