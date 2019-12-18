# For Backoff Multilanguage classifier

#### If you create your own data using our utilities to parse out Spanish English tweets or procure some external bilingual corpora with csv comma seperated words and line seperated sentences, you will be able to run using those csv files for the model. Note that our utilities create a lot of empty comma seperated values after cleaning data.
#### Delete those extra commas if possible as otherwise blanks are more likely to be identified as Dutch due to our training data. Otherwise, a higher quality corpora
#### might work better with our model. For a tiny 7 sentence corpora of high quality spanglish sentences, we were able to get very accurate results. For two lower quality 
#### corpora we had greater difficulties. 

#### Note, our data sets that you can use are as follows. 
biList.csv was a dataset one of us chose while first extracting data. It was picked randomly but with a high degree of selectiveness for long sentences
that we are certain are "Spanglish". These were high quality manually chosen tweets but there are only 7 of them.
Our model was very good at picking out the two languages these tweets were written in getting only one false negative. 

Data.csv was created by others using the utility we created for picking out sentences. However, there is an issue in that many sentences in said set 
were actually purely Spanish but annotated by persons who were not familiar with Spanish and therefore chosen erroneously. 
This was not the best quality dataset but it contains 76 entries which were found after looking over 1800 rows of our approximately 11500 tweets.
This illustrates how difficult it was to create a corpora. 

combinedCleaned.csv is a dataset that merges the two datasets above while also removing erroneous entries to the best of our group ability. However, this is still not
a very high quality dataset because some sentences contain garbage url's or other things. We did not want to corrupt our results by manually editing data. This is the
corpora which achieved .425 recall and precision (although to use recall and precision is redundant when one is looking at supposedly only true results)