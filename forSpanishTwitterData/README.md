# Spanish Twitter Language Data

#### Obtained via similar methods to unknown data. This contains approximately 6500 spanish tweets we were able to extract.

#### For reference we used the following:
https://blog.twitter.com/engineering/en_us/a/2015/evaluating-language-identification-performance.html

##### This time we will be manually looking at spanish identified tweets to see if any of them contain english and are therefore bilingual

#### Everything is run in the same way as the unknown twitter data parse. We simply look at different labels in our code.

#### In order to create the out.csv file from merged twitter es data, run the following.
python mergefile_modify_data.py

###### To later manually pick out bilingual spanglish tweets, we renamed this file and moved it to the folder "forFindingBilingualSpanglish"
###### Consult that folder to see how we generated our corpora manually after all of this data processing. 