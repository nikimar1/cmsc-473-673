# cmsc-473-673 

### Use yaml file if needed to set up conda env
#### conda env create --file 473NLPEmptyEnv.yaml
#### conda activate 473NLPEmptyEnv

## This is now using all 6 Languages in the following order: 
#### Lang1 is French, Lang2 is English, Lang3 is German
#### Lang 4 is Spanish, Lang 5 is Italian, Lang 6 is Dutch

### To train on train with training set and evaluate on dev with laplace constant of .1:

python "laplace6Lang.py" ./fr_gsd-ud-train.conllu ./en_ewt-ud-train.conllu ./de_gsd-ud-train.conllu ./es_gsd-ud-train.conllu ./it_isdt-ud-train.conllu ./nl_alpino-ud-train.conllu ./fr_gsd-ud-dev.conllu ./en_ewt-ud-dev.conllu ./de_gsd-ud-dev.conllu ./es_gsd-ud-dev.conllu ./it_isdt-ud-dev.conllu ./nl_alpino-ud-dev.conllu .1

### To output results to text file
python "laplace6Lang.py" ./fr_gsd-ud-train.conllu ./en_ewt-ud-train.conllu ./de_gsd-ud-train.conllu ./es_gsd-ud-train.conllu ./it_isdt-ud-train.conllu ./nl_alpino-ud-train.conllu ./fr_gsd-ud-dev.conllu ./en_ewt-ud-dev.conllu ./de_gsd-ud-dev.conllu ./es_gsd-ud-dev.conllu ./it_isdt-ud-dev.conllu ./nl_alpino-ud-dev.conllu .1 >outputlaplace.txt

### Later after tuning on dev set and finding best laplace constant, check results on test sets. Substitute .1 with best laplace constant
python "laplace6Lang.py" ./fr_gsd-ud-train.conllu ./en_ewt-ud-train.conllu ./de_gsd-ud-train.conllu ./es_gsd-ud-train.conllu ./it_isdt-ud-train.conllu ./nl_alpino-ud-train.conllu ./fr_gsd-ud-test.conllu ./en_ewt-ud-test.conllu ./de_gsd-ud-test.conllu ./es_gsd-ud-test.conllu ./it_isdt-ud-test.conllu ./nl_alpino-ud-test.conllu .1

### For backoff model run the following commands with optional commands for uniBackoff and biBackoff constants which default to .5 otherwise
python "backoff6lang.py" ./fr_gsd-ud-train.conllu ./en_ewt-ud-train.conllu ./de_gsd-ud-train.conllu ./es_gsd-ud-train.conllu ./it_isdt-ud-train.conllu ./nl_alpino-ud-train.conllu ./fr_gsd-ud-dev.conllu ./en_ewt-ud-dev.conllu ./de_gsd-ud-dev.conllu ./es_gsd-ud-dev.conllu ./it_isdt-ud-dev.conllu ./nl_alpino-ud-dev.conllu .5 .5

## For Twitter Data

#### First attempt
At first, I attempted to search through the twitter api with queries based on geocode (location) and language. 
I.e. looking for English tweets in Madrid in hopes they would be bilingual spanish and English. 
I also looked for tweets in southern California that are in Spanish. This did not prove an effective method.
To create my credentials for accessing twitter api, I ran a command with a file similar to the command:

python redactedCreateTwitterCredentials.py

This created a json file I use in my first twitter search program for credentials. 
The difference is I had to fill in various personal keys and tokens that my twitter development account provided for me. I will not be sharing those publicly
nor will I share the json file.

Then I had to run the following in my conda env: 

pip install twython

and potentially: 

pip install json 

in order to be able to use those python modules. Twython was one module for querying twitter api which I tried using.

Then I ran the following which uses said json file:
python twitsearch.py

You can see a few of the queries I tried in that file and I have one csv file containing all Spanish tweet results I was able to get within a radius
of the Southern California location I found. Unfortunately, there are not many and they are purely Spanish not bilingual. The file is:

calspanishtweets.csv

#### Second attempt. 
I used some other utility for generating a jsonl file with english tweets in Madrid. I do not recall 
what I used to create this but it was not my own code and there was a lot more output. I believe
my twitter developer account and the twython utility were for some reason limited in terms of output and queries while 
this online utility provided greater functionality. However, I once again found that many English tweets in Madrid
were not bilingual and I furthermore found that this was not a very useful method, as well as generating a lot of data I did not need.
It was quite difficult to parse this data

To see the output, one can open the following file:
englishTweetsInMadrid.jsonl

#### Third attempt at generating a bilingual Spanish English corpus 
##### (Note that I will be citing the papers I used for reference properly in the final project report.)

I then consulted some prior research that looked at multilingual code switching. Code switching is defined as 
the phenomenon by which multilingual speakers switch back and forth between their common languages in written or spoken communication.
We planned to create multiclass linguistic classifiers using several models we initially developed for single language classification. 

Firstly, I found the following mention of a multilingual corpus:
https://books.google.com/books?id=h_GiDQAAQBAJ&pg=PA12&lpg=PA12&dq=CoMAprend+corpus&source=bl&ots=0r4RYXZ88P&sig=ACfU3U1N0zbl9TegjYtRg1jImvipJbO8fA&hl=en&sa=X&ved=2ahUKEwiD-NeX9rrmAhWMY98KHWCqDZQQ6AEwAHoECAwQAQ#v=onepage&q=CoMAprend%20corpus&f=false

The CoMAprend corpus contains most of the languages we intended to look at in a single corpus being Spanish, French, German, English, Italian. However, we were unable
to find this corpus or the larger COMET corpus it is a part of for download. 

I then consulted the following paper from the University of Houston:
https://www.aclweb.org/anthology/W15-1608.pdf

They mentioned that one methodology for finding tweets was that they found twitter accounts through bilingual friends or on their own that tended to tweet multilingual
code-switched language. They also crowdsourced tweets in order to create their corpora. They discussed many of the difficulties we ourselves found in creating this sort of corpus.
On that note, I did manage to find a few multilingual tweets manually but due to our lack of crowdsourcing resources, this was not a feasible method for us to do on our own. 

I then looked at the following blog post created by twitter engineers with regards to their language evaluation:

https://blog.twitter.com/engineering/en_us/a/2015/evaluating-language-identification-performance.html

This cited some of their methodology for language classification and made a mention that languages that were mixed sufficiently such that a primary language
could not be found would be labelled as unknown. There were various other criteria for labelling a language as unknown but this at least provided an option
for a potential multilingual dataset. They also mentioned that languages which featured two languages but were predominately one language and featured only a word or two
in the other language would be labelled as just the primary language. Thus, we decided to look at the languages they labelled 
unknown and spanish. They provided a tsv file with ID's of tweets they used to evaluate their classifier as well as the labels said classifier assigned. 

Using that tsv file and the utility they reccomended for extracting tweets, I was able to get all non-deleted tweets from their model that were labelled as Unknown or Spanish.

Consult the folders: "for twitter data" and "forSpanishTwitterData" to see our methodology for extracting these tweets. 
Note that our code used commandline arguments called by python in a loop to generate text files of the tweets. 
Thus, one had to install the ruby along with ruby gems, and then use ruby gems to install the twurl utility for accessing twitter api data.
One then had to authenticate twurl on the commandline followed by running our programs. We will not provide detailed information about the ruby installation
or the changes to path variables that we made to enable the commandline arguments we used. Suffice it to say that we installed Ruby and added it to our path followed by
installing twurl. The following is a guide for working with twurl:

https://github.com/twitter/twurl

We also had to pip install jq in our environment for parsing json data generated by our queries into an easier to use format.

Regardless, the data we extracted and later concatenated is available even if you are unable to run our code for generating it. 

#### Processing data extracted by twurl

##### For processing the bilingual data, we ran a commandline utility we created which will parse the two tables of cleaned data. This generates a csv file of bilingual data after we manually input our decisions and parse our tweets. In better circumstances, we now know that outsourcing a large body of Spanish twitter data for annotation of English Spanish bilingualism would have been more effective as a means of creating a corpora. Regardless, for our methods of generating a corpora consult the folder "forFindingBilingualSpanglish".
###### Note: before running our utility that in order to avoid overwriting the bilingual csv file we generated we recommend renaming the file currently named biList.csv before running our utility. 

#### Bilingual classification attempt. 

###### Due to lack of corpora and data, we simply attempted to classify a small dev set of spanglish tweets we ourselves generated. We did not have sufficient data to create training and test sets or sets for all languages and we therefore used the connlu universal dependency training sets for training our 6 languages, instead evaluating our own generated corpora as spanish english multiclasss or not.
###### We may have erroneously annotated tweets although to the best of our knowledge our corpora is sound. We are not Spanish speakers.

###### To run this classifier which is a backoff noisy channel classifier, run the following command with input unigram and bigram backoff constants or nothing for a default:
python "twitterBackoff.py" ./fr_gsd-ud-train.conllu ./en_ewt-ud-train.conllu ./de_gsd-ud-train.conllu ./es_gsd-ud-train.conllu ./it_isdt-ud-train.conllu ./nl_alpino-ud-train.conllu ./combinedCleaned.csv .5 .5  

#### Feel free to use another csv file you generate yourself as our data was not the best. We got an f1 score of .425 but if one looks at the output of our code, the model was able to predict many of bilingual sentences only making mistakes with a 3 word sentence like "hat, sombrero, 'some url with a lot of garbage in it'. Another case where our model failed was a case where the only english word was no, noooo. With good data points like ['isabel472235', 'lmao', 'pues', 'ya', 'sabes', 'que', 'yo', 'si', 'se', 'de', 'musica', '', 'perrona', '', '', 'pues', 'ahi', 'me', 'dices', 'mas', 'so', 'we', 'can', 'buy', 'the', 'tickets'] which contained a long unambiguously bilingual sentence, our model had no trouble. Looking over sentences and what labels we generated in output it is easy to see that the issue is with data containing url's and other garbage despite our best attempts at cleaning as well as data having very short internet twitter sentences with some language ambiguity that even we as humans could not decide upon. We must also admit that some false classifications resulted form models working with latin based languages that have commonalities and are therefore showing false multiclass reuslts of spanish and french and so forth. That being said, this was a rare occurence. Our model methodology has potential if we perform better data extraction and procure enough data for good test training and dev sets. 

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

##### Csv file cleaning
We have to note that the csv files genreated by our programs and cleaned up by the word segment utility ended up having many empty 
spaces where emojis or other things were removed. We could easily have created a utility for removing any cases of more than one comma 
in a row to fix this but we manually cleaned out the blanks from our csv files because they were frankly small datasets and did not warrant creating a tool for. 


##### Mutliclass functionality
This will generate precision recall and f1 scores after training our classifier model and then evaluating its top two language scores
If the top two scores are Spanish and English in any order, it will output that multiclass but otherwise it will output other. 
This is because of the limited devset and our ability to only test for this. Regardless, Spanish English bilingual or not will be the threshold
for evaluating precision recall and f1.

#### Scikitlearn
Lastly, we also created a maxent and laplace model using scikitlearn. These models were well optimized and far better able to classify data than our own models.
F1 scores for both models are .999 or higher using English, French, German, and Italian on the LEPZIG corpus. 
However, we were unable to figure out a way to use scikitlearn to classify for two classes which have no priority. I.e. if we
have a sentence that is Spanish and English and we do not care if the system thinks Spanish or English is more likely as long as we identify both languages,
we would need a system that is able to reflect that result. We were neither able to utilize scikit for predicting two most likely languages nor were we able
to find a way to ignore the order of the class predictions. It may be the case that scikit learn or some other module could accomplish the results we needed. 
We were unable to persue a good multiclass bilingual code switching maxent model because we were too busy procuring data. We were however very successful with creating
various models using both modules and our own code to predict single language class. 

No good corpus for code-switching multilngual sentences in the French, English, German, Spanish, Italian, and Dutch languages was found. Our analysis of
twitter data was only partially successful due to limits to our free developer twitter api access as well as limits in resources for annotating and procuring data.

#### To run our scikit learn laplace and maxent language classifiers, one must be in the folder "scikitLearnMaxentLaplace" and run the following command:
python project_code.py
