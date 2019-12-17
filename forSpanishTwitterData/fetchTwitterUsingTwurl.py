import os
import pandas as pd
import numpy as np

df = pd.read_csv('./uniformly_sampled.tsv', sep='\t')

#had to add the following code to continue progress from where it stopped after timeout with twitter api.
rowFound = False 

#count = 0
#had to modify count to continue where I left off
count = 4861
for index, row in df.iterrows():
    
    #get only spanish tweets as marked in the prior research. that way can check if they are bilingual. this is a bit over 10000 tweets
    if row[0]== 'es' and rowFound:
        os.system('sleep 1')
        os.system('twurl "/1.1/statuses/lookup.json?id=%s&trim_user=true" | jq -c ".[]|[.id_str, .text]" >tmp%s.txt'%(row[1],count))
        count+=1
    
    #had to add the following code to continue progress after twitter api timeout. 
    #note that because I had the if statements in the opposite order initially, I accidentally duplicated one single spanish tweet so
    #my 4860 and 4861 txt files are the same
    if row[1]==488863629622538240:
        rowFound = True