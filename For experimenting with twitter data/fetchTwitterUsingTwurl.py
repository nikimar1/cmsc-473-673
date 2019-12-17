import os
import pandas as pd
import numpy as np

df = pd.read_csv('./uniformly_sampled.tsv', sep='\t')

count = 0
for index, row in df.iterrows():
    #get only undefined tweets as marked in the prior research. that way can check if they are bilingual. this is a bit over 10000 tweets
    if row[0]== 'und':
        os.system('sleep 1')
        os.system('twurl "/1.1/statuses/lookup.json?id=%s&trim_user=true" | jq -c ".[]|[.id_str, .text]" >tmp%s.txt'%(row[1],count))
        count+=1