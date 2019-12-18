# Unknown Twitter Language Data

##### This is similar to our experimental folder. However this is a complete set of all data tagged as unknown language in the research we are referencing.
##### This readme or our report will be updated with a citation. For now I am linking the url. We will be manually parsing this data for bilingual tweets
##### in order to form a corpora. Note that in better circumstances we would have annotates proficient in several languages confirm languages and skip 
##### in those cases when the language is not known to them. Instead we are doing a cursory manual analysis and parse on our own.

#### Contains a little over 5500 tweets. Data and references obtained from:

https://blog.twitter.com/engineering/en_us/a/2015/evaluating-language-identification-performance.html

#### A useful tidbit of information

##### The following url with id substituted for "id" will open tweets given the id's in our tab delimited file. 
##### This is useful for manually checking if tweets are bilingual or not. Url below:

twitter.com/anyuser/status/ "id" 

##### Note that we had to remove some items from the merge file manually such as a tweet "[" which broke our regex
##### as well as the character at the file end. However, this file can print out all of the tweet data parsing it out.
##### We will later be using this to procure a dev set after we have manually pruned the file for only spanish english bilingual tweets.
##### Afterwards, we will modify some of our algorithms for language classification to take in the same connlu training data as before
##### but to afterwards  classify the two most likely languages from the parsed spanish-english .txt file corpora we create. 

###### First, we obtained tweets using a ruby utility. It was installed using ruby gems so you will likely be unable to run this unless you follow
###### guidelines from the main folder branch. However, the generated text files are in this repository.
python fetchTwitterUsingTwurl.py

###### Since some tweets were deleted, we had empty text files. We deleted those by running the following:
python removeEmptyFiles.py

###### Later we merged these files using 'python merge.py'
###### This created mergefile.txt

###### Finally we processed them into a cleaned up parsed csv. 
###### To do this we used the wordsegment python module which is 
###### an Apache2 licensed module for English word segmentation based on a trillion-word corpus.
###### This module had functionality for seperating words with no spaces into words as well as utilities for cleaning up data
###### by removing emojis and other fluff. We used this to parse and clean tweets. In order to use this utility,
###### one would have to pip install wordsegment in our conda env. 
###### We also used fnmatch but did not have to install that module. This was used to find all text that started with "#"
###### We then ran wordsegment on those words to seperate them out into words with removed punctuation lowercase and other preprocessing.
###### We also cleaned up non hashtag words using the wordsegment utility to preprocess and remove emojis and some other things.

###### Our program for reading in the merged text file and parsing it out to an out.csv file can be called as follows
python mergefile_modify_data.py

###### To later manually pick out bilingual spanglish tweets, we renamed this file and moved it to the folder "forFindingBilingualSpanglish"
###### Consult that folder to see how we generated our corpora manually after all of this data processing. 