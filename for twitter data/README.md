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

#### For now, here is the command to print all the merged and formatted tweets prior to parsing out non bilingual ones.
python mergefile_modify_data.py