from itertools import islice

from io import open
from conllu import parse_incr
import pandas as pd
import numpy as np
from collections import Counter 
import string 
import math
import pickle

#for deep copy
import copy

import sys

#for precision and recall
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

"""
#for debugging 
import warnings
import traceback

	
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

	log = file if hasattr(file,'write') else sys.stderr
	traceback.print_stack(file=log)
	log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback
warnings.simplefilter("always")
"""

def bigramLaplace(pathToTrainLang1, pathToTrainLang2, pathToTuneLang1, pathToTuneLang2, pathToSerialize, oovConstant = 1, laplaceConstant = .1):
	
	#I am populating some unknown token list with a minimal training set of tokens with frequency one. 
	#This is because having no probability for unknown tokens is bad form in my opinion and this distributes at least a small probability to <UNK>
	if int(oovConstant)>=1:
		oovFrequency = int(oovConstant)
	else:
		oovFrequency = 1
	
	#get laplace constant from hyperparamters. cannot be 0 or less
	if float(laplaceConstant)> 0:
		laplace = float (laplaceConstant)
	else:
		laplace = .1

	#open both files 
	train1 = open(pathToTrainLang1, "r", encoding="utf-8")
	train2 = open(pathToTrainLang2, "r", encoding="utf-8") 

	#used as temporary storage per each sentence as the connlu parser iterates over them
	tempSentence = list()

	#list for storing word tokens
	list1 = list()
	list2 = list()
	
	#at first storing observed bigrams using dictionary 
	bigramCounts1 = {}
	bigramCounts2 = {}

	#list for storing sentences which will later be used to update bigram
	sentenceList1 = list()
	sentenceList2 = list()

	#for storing number of sentences in languages
	#numSentences1 = 0
	#numSentences2 = 0
	
	#connlu parse and update bigram and unigram counts 
	for tokenlist in parse_incr(train1):
		for token in tokenlist:
			##adding to temporary sentence which will be parsed into bigrams
			#making it lower case as a means of preprocessing. words of different case but same spelling are the same type for my purposes
			tempSentence.append(token["form"].lower())
			#adding to list of tokens. this will later be used to get unigram counts 
			list1.append(token["form"].lower())
			
		#now adding eos and bos tags to the sentence
		tempSentence.insert(0, "<BOS>")
		tempSentence.append("<EOS>")
		
		#now adding them to unigram token list 
		list1.append("<BOS>")
		list1.append("<EOS>")

		#count of sentences in language one used for initial probability
		#numSentences1+=1
		
		#add the parsed sentence list of words to nested list of sentences
		sentenceList1.append(tempSentence)

		#resetting the temporary list of words per each sentence 
		tempSentence = []

	train1.close()
	
	tempSentence = []
	
	#connlu parse and update bigram and unigram counts 
	for tokenlist in parse_incr(train2):
		for token in tokenlist:
			##adding to temporary sentence which will be parsed into bigrams
			#making it lower case as a means of preprocessing. words of different case but same spelling are the same type for my purposes
			tempSentence.append(token["form"].lower())
			#adding to list of tokens. this will later be used to get unigram counts 
			list2.append(token["form"].lower())
			
		#now adding eos and bos tags to the sentence
		tempSentence.insert(0, "<BOS>")
		tempSentence.append("<EOS>")
		
		#now adding them to unigram token list 
		list2.append("<BOS>")
		list2.append("<EOS>")
		
		#count of sentences in language two used for initial probability
		#numSentences2+=1

		#add the parsed sentence list of words to nested list of sentences
		sentenceList2.append(tempSentence)

		#resetting the temporary list of words per each sentence 
		tempSentence = []

	train2.close()
	
	tempSentence = []

	#create dataframe containing all tokens and convert to series of counts per word type.
	df1 = pd.DataFrame(list1)
	series1= df1[0].value_counts()
	
	df2 = pd.DataFrame(list2)
	series2= df2[0].value_counts()

	#for storing filtered vocab filtered by frequency 
	filteredList1 = pd.Series()
	#extract frequencies below oovFrequency cutoff constant and set them to some unknown token
	unknownCount = 0
	for index, value in series1.items():
		if value > oovFrequency:
			filteredList1.at[index] = value
		else:
			unknownCount+=value

	#add unknown count to list with token <UNK> as the index 
	filteredList1.at['<UNK>'] = unknownCount
	
	#for storing filtered vocab filtered by frequency 
	filteredList2 = pd.Series()
	#extract frequencies below oovFrequency cutoff constant and set them to some unknown token
	unknownCount = 0
	for index, value in series2.items():
		if value > oovFrequency:
			filteredList2.at[index] = value
		else:
			unknownCount+=value

	#add unknown count to list with token <UNK> as the index 
	filteredList2.at['<UNK>'] = unknownCount

	#for storing list of words (types) in the vocab which will be used for indexing the rows and columns of the dataframe
	wordList1 = filteredList1.keys()
	wordList2 = filteredList2.keys()

	#get number of types
	sizeOfVocab1=len(wordList1)
	sizeOfVocab2=len(wordList2)

	#filter out unknown words and make observed bigrams into dictionary 
	for tempSentence in sentenceList1:
		#first filtering out the out of vocab words 
		for i in range( len(tempSentence)):
			if not tempSentence[i] in wordList1: 
				tempSentence[i] = "<UNK>"
		#parsing bigrams by pythonically creating them with islice and zip 
		tempBigram = zip(tempSentence, islice(tempSentence, 1, None))
		#iterating over created list of bigrams and adding new ones to the dictionary while incrementing counts for existing bigrams 
		for wordPair in tempBigram :
			if (wordPair[0], wordPair[1]) in bigramCounts1:
				bigramCounts1[(wordPair[0], wordPair[1])] += 1
			else:
				bigramCounts1[(wordPair[0], wordPair[1])] = 1
				
	#filter out unknown words and make observed bigrams into dictionary 
	for tempSentence in sentenceList2:
		#first filtering out the out of vocab words 
		for i in range( len(tempSentence)):
			if not tempSentence[i] in wordList2: 
				tempSentence[i] = "<UNK>"
		#parsing bigrams by pythonically creating them with islice and zip 
		tempBigram = zip(tempSentence, islice(tempSentence, 1, None))
		#iterating over created list of bigrams and adding new ones to the dictionary while incrementing counts for existing bigrams 
		for wordPair in tempBigram :
			if (wordPair[0], wordPair[1]) in bigramCounts2:
				bigramCounts2[(wordPair[0], wordPair[1])] += 1
			else:
				bigramCounts2[(wordPair[0], wordPair[1])] = 1

	#creating sparse matrix dataframe
	bigramLaplaceFrame1 = pd.DataFrame(np.zeros(shape=(sizeOfVocab1,sizeOfVocab1)))

	#creating sparse matrix dataframe
	bigramLaplaceFrame2 = pd.DataFrame(np.zeros(shape=(sizeOfVocab2,sizeOfVocab2)))

	#set index headers to be the list of words
	bigramLaplaceFrame1.index = wordList1
	#set column headers to be all of the words for comparison
	bigramLaplaceFrame1.columns = wordList1
	
	#set index headers to be the list of words
	bigramLaplaceFrame2.index = wordList2
	#set column headers to be all of the words for comparison
	bigramLaplaceFrame2.columns = wordList2

	#set default value in matrix to laplace constant
	for col in bigramLaplaceFrame1.columns:
		bigramLaplaceFrame1[col].values[:]=laplace

	#set default value in matrix to laplace constant
	for col in bigramLaplaceFrame2.columns:
		bigramLaplaceFrame2[col].values[:]=laplace

	#increment and add observed bigram counts into the matrix on top of the default values
	for x in bigramCounts1 :
		bigramLaplaceFrame1.loc[x[0],x[1]] += bigramCounts1[x]
		
	#increment and add observed bigram counts into the matrix on top of the default values
	for x in bigramCounts2 :
		bigramLaplaceFrame2.loc[x[0],x[1]] += bigramCounts2[x]

	#find V value to be added to all denominators for laplace smoothing bigram probability
	V1 = laplace *sizeOfVocab1
	V2 = laplace *sizeOfVocab2

	#dividing all columns by the word counts per row of said column + V
	bigramLaplaceFrame1 = bigramLaplaceFrame1.div( (filteredList1 + V1 ), axis=0)
	
	#dividing all columns by the word counts per row of said column + V
	bigramLaplaceFrame2 = bigramLaplaceFrame2.div( (filteredList2 + V2 ), axis=0)
	
	#To make serialization easier on this huge structure. convert my data from 64 bit to 32 bit floats. Otherwise, I would have to serialize a little under 2.7 gb
	#I actually already did this before but I am now fixing the code to avoid that.
	bigramLaplaceFrame1 = bigramLaplaceFrame1.astype(np.float32)
	bigramLaplaceFrame2 = bigramLaplaceFrame2.astype(np.float32)

	#for debugging purposes printing bigram. 
	print(bigramLaplaceFrame1)
	print(bigramLaplaceFrame2)

	#Now train model by going over all sentences and classifying them using markov chain bigram probability products to get estimated p(sentence)
	#Then calculate probability using posterior and prior probability. Iterate this process for all sentences in both languages to train the predictor. 
	#Later trained predictor is used with maximum of probabilities designated as the language classification and it is tested/tuned against the dev set
	#The math behing what I am doing is using bigrams to calc log (p(sentence| l)) + log (p (l)) for each language and setting that as the probability

	#initial probability is using simply count of sentences in one language over total count of sentences
	probLang1 = len(sentenceList1) / (len(sentenceList1) + len(sentenceList2))
	probLang2 = len(sentenceList2) / (len(sentenceList1) + len(sentenceList2))
	
	#I will take turns picking a sentence from each language to train the model so that I am never using the same language twice in a row unless the other is depeleted of sentences
	#This is to randomize the sentence access and I believe it might improve the model rather than progressively getting a really high probability for one language if I train using it first
	
	#I will check if this number is even or not to decide
	decision = 2 
	
	#I am tracking how many sentences in the training sets I have iterated over to avoid going past the end
	
	countLang1=0
	countLang2 = 0
	
	#while loop to train probability for every sentence in both languages
	while ( countLang1<(len(sentenceList1))or (countLang2< len(sentenceList2))) :


		if ((decision %2) == 0 ) and (countLang1<len(sentenceList1)):
			currentSentence1 = sentenceList1[countLang1]
			currentSentence2 = copy.deepcopy(currentSentence1)
			
			#set temp p(sentence|l) to default values before calculating the probability via multiplication (markov chain estimate from bigrams)
			probSentenceGivenL1 = 1
			probSentenceGivenL2 = 1
			
			for i in range( len(currentSentence1)):
				#handle oov words for both languages
				if not currentSentence1[i] in wordList1: 
					currentSentence1[i] = "<UNK>"
				if not currentSentence2[i] in wordList2: 
					currentSentence2[i] = "<UNK>"
			#parsing bigrams by pythonically creating them with islice and zip 
			tempBigram1 = zip(currentSentence1, islice(currentSentence1, 1, None))
			tempBigram2 = zip(currentSentence2, islice(currentSentence2, 1, None))
			#iterating over created list of bigrams and markov chain probability by multiplying all bigram probabilities
			for wordPair in tempBigram1 :
				probSentenceGivenL1*=(bigramLaplaceFrame1.loc[wordPair[0], wordPair[1]])
			for wordPair in tempBigram2 :
				probSentenceGivenL2*=(bigramLaplaceFrame2.loc[wordPair[0], wordPair[1]])
			
			
			#In some cases that probability multiplication markov chain gets below the threshold where floats can be represented and becomes 0.
			#In those cases, p(sentence|l) will be set to almost the smallest representable python decimal
			
			if probSentenceGivenL1 == 0:
				probSentenceGivenL1 = (2**-126)
			if probSentenceGivenL2 == 0:
				probSentenceGivenL2 = (2**-126)
			
			#get probabilities by using noisy channel/ bayesian probabilities
			tempProb1 = probSentenceGivenL1*probLang1
			tempProb2 = probSentenceGivenL1*probLang2
			
			#normalize these probabilities 
			probLang1 = tempProb1/(tempProb1+ tempProb2)
			probLang2 = tempProb2/(tempProb1+ tempProb2)
			
			countLang1+=1

		elif (countLang2<len(sentenceList2)):
			
			#print()
			#print(countLang2)
			#print(sentenceList2[countLang2])
			#print("\n\n\n")

			currentSentence2 = sentenceList2[countLang2]
			currentSentence1 = copy.deepcopy(currentSentence2)
			
			#set p(sentence|l) to default values before calculating the probability via multiplication
			probSentenceGivenL1 = 1
			probSentenceGivenL2 = 1
			
			for i in range( len(currentSentence2)):
				#handle oov words for both languages
				if not currentSentence1[i] in wordList1: 
					currentSentence1[i] = "<UNK>"
				if not currentSentence2[i] in wordList2: 
					currentSentence2[i] = "<UNK>"
			#parsing bigrams by pythonically creating them with islice and zip 
			tempBigram1 = zip(currentSentence1, islice(currentSentence1, 1, None))
			tempBigram2 = zip(currentSentence2, islice(currentSentence2, 1, None))
			#iterating over created list of bigrams and markov chain probability by multiplying all bigram probabilities
			for wordPair in tempBigram1 :
				probSentenceGivenL1*=(bigramLaplaceFrame1.loc[wordPair[0], wordPair[1]])
			for wordPair in tempBigram2 :
				probSentenceGivenL2*=(bigramLaplaceFrame2.loc[wordPair[0], wordPair[1]])
			
			#Get new probability by normalizing bayseian probability where p(l|sentence) is approximately p(sentence|l) * p(l) prior
			
			#In some cases that probability multiplication markov chain gets below the threshold where floats can be represented and becomes 0.
			#In those cases, p(sentence|l) will be set to the smallest representable python float32
			
			if probSentenceGivenL1 == 0:
				probSentenceGivenL1 = (2**-126)
			if probSentenceGivenL2 == 0:
				probSentenceGivenL2 = (2**-126)
			
			if probLang1 == 0:
				probLang1 = (2**-126)
			if probLang2 == 0:
				probLang2 = (2**-126)
			
			#get probabilities by using noisy channel/ bayesian probabilities
			tempProb1 = probSentenceGivenL1*probLang1
			tempProb2 = probSentenceGivenL1*probLang2
			
			#normalize these probabilities 
			probLang1 = tempProb1/(tempProb1+ tempProb2)
			probLang2 = tempProb2/(tempProb1+ tempProb2)

			countLang2+=1

		decision +=1

	#final trained probabilities
	print("\nFinal trained probability of the first language passed to this program:")
	print(probLang1)
	print("Final trained probability of the second language passed to this program:")
	print(probLang2)

	#Code for checking results in dev set to be used for tuning hyperparamters
	#This dev run will not tweak trained probabilities. it will merely evaluate them

	en_dev1 = open(pathToTuneLang1, "r", encoding="utf-8")
	
	en_dev2 = open(pathToTuneLang2, "r", encoding="utf-8")
	
	#for storing dev bigrams
	devBigramCount1 = {}
	devBigramCount2 = {}
	
	tempSentence1 = list()
	tempSentence2 = list()
	
	#lists for storing predicted languages vs actual langs
	predictedLang = []
	actualLang = []
	
	#connlu parse and update bigram and unigram counts 
	for tokenlist in parse_incr(en_dev1):
		for token in tokenlist:
			##adding to temporary sentence which will be parsed into bigrams
			#making it lower case as a means of preprocessing. words of different case but same spelling are the same type for my purposes
			
			#if out of vocab word, append unk instead
			if not (token["form"].lower()) in bigramLaplaceFrame1.index :
				tempSentence1.append('<UNK>')
			else:
				tempSentence1.append(token["form"].lower())
			if not (token["form"].lower()) in bigramLaplaceFrame2.index :
				tempSentence2.append('<UNK>')
			else:
				tempSentence2.append(token["form"].lower())

		#now adding eos and bos tags to the sentence
		tempSentence1.insert(0, "<BOS>")
		tempSentence1.append("<EOS>")
		tempSentence2.insert(0, "<BOS>")
		tempSentence2.append("<EOS>")
		
		probSentenceGivenL1 = 1
		probSentenceGivenL2 = 1

		tempBigram1 = zip(tempSentence1, islice(tempSentence1, 1, None))
		tempBigram2 = zip(tempSentence2, islice(tempSentence2, 1, None))
		#iterating over created list of bigrams and adding new ones to the dictionary while incrementing counts for existing bigrams 
		for wordPair in tempBigram1 :
			probSentenceGivenL1*=(bigramLaplaceFrame1.loc[wordPair[0], wordPair[1]])
		for wordPair in tempBigram2 :
			probSentenceGivenL2*=(bigramLaplaceFrame2.loc[wordPair[0], wordPair[1]])
			
		#In some cases that probability multiplication markov chain gets below the threshold where floats can be represented and becomes 0.
		#In those cases, p(sentence|l) will be set to almost the smallest representable python decimal	
		if probSentenceGivenL1 == 0:
			probSentenceGivenL1 = (2**-126)
		if probSentenceGivenL2 == 0:
			probSentenceGivenL2 = (2**-126)
		
		if probLang1 == 0:
			probLang1 = (2**-126)
		if probLang2 == 0:
			probLang2 = (2**-126)
		
		#predict which language it is using logs
		logProb1 = math.log(probSentenceGivenL1) + math.log(probLang1)
		logProb2 = math.log(probSentenceGivenL2) + math.log(probLang2)
		
		if(logProb1 < logProb2):
			predictedLang.append('Lang2')
		else:
			predictedLang.append('Lang1')
		
		actualLang.append('Lang1')
		
		#resetting the temporary list of words per each sentence 
		tempSentence1 = []
		tempSentence2 = []
		
	tempSentence1 = []
	tempSentence2 = []
	
	en_dev1.close()
	
	
	#connlu parse and update bigram and unigram counts 
	for tokenlist in parse_incr(en_dev2):
		for token in tokenlist:
			##adding to temporary sentence which will be parsed into bigrams
			#making it lower case as a means of preprocessing. words of different case but same spelling are the same type for my purposes
			
			#if out of vocab word, append unk instead
			if not (token["form"].lower()) in bigramLaplaceFrame1.index :
				tempSentence1.append('<UNK>')
			else:
				tempSentence1.append(token["form"].lower())
			if not (token["form"].lower()) in bigramLaplaceFrame2.index :
				tempSentence2.append('<UNK>')
			else:
				tempSentence2.append(token["form"].lower())

		#now adding eos and bos tags to the sentence
		tempSentence1.insert(0, "<BOS>")
		tempSentence1.append("<EOS>")
		tempSentence2.insert(0, "<BOS>")
		tempSentence2.append("<EOS>")
		
		probSentenceGivenL1 = 1
		probSentenceGivenL2 = 1
		
		##########
		####change code for final project. don't use pandas 2 dimensional dataframe. 
		####instead keeping dictionaries but using laplace constant for probabilities that are 0
		####change this to add log sums instead. also change how it counts unknown vocab. 
		####count it at 0. calculate normalizing z in the end as well
		##########
		
		tempBigram1 = zip(tempSentence1, islice(tempSentence1, 1, None))
		tempBigram2 = zip(tempSentence2, islice(tempSentence2, 1, None))
		#iterating over created list of bigrams and adding new ones to the dictionary while incrementing counts for existing bigrams 
		for wordPair in tempBigram1 :
			probSentenceGivenL1*=(bigramLaplaceFrame1.loc[wordPair[0], wordPair[1]])
		for wordPair in tempBigram2 :
			probSentenceGivenL2*=(bigramLaplaceFrame2.loc[wordPair[0], wordPair[1]])
		
		############commented out because I am now adding logarithms instead
		#In some cases that probability multiplication markov chain gets below the threshold where floats can be represented and becomes 0.
		#In those cases, p(sentence|l) will be set to almost the smallest representable python decimal	
		#if probSentenceGivenL1 == 0:
		#	probSentenceGivenL1 = (2**-126)
		#if probSentenceGivenL2 == 0:
		#	probSentenceGivenL2 = (2**-126)
		
		#predict which language it is using logs
		logProb1 = math.log(probSentenceGivenL1) + math.log(probLang1)
		logProb2 = math.log(probSentenceGivenL2) + math.log(probLang2)
		
		if(logProb1 < logProb2):
			predictedLang.append('Lang2')
		else:
			predictedLang.append('Lang1')
		
		actualLang.append('Lang2')
		
		#resetting the temporary list of words per each sentence 
		tempSentence1 = []
		tempSentence2 = []
		
	tempSentence1 = []
	tempSentence2 = []
	
	en_dev2.close()
	
	precision = precision_score(actualLang, predictedLang,average = "macro")
	
	recall = recall_score(actualLang, predictedLang,average = "macro")
	
	print("\nPrecision is:")
	print(precision)
	
	print("Recall is:")
	print(recall)

	f1Score = (2*precision*recall)/(precision+recall)
	print("F1Score is:")
	print(f1Score)

	#calculate most common baseline
	baselineprobLang1 = len(sentenceList1) / (len(sentenceList1) + len(sentenceList2))
	baselineprobLang2 = len(sentenceList2) / (len(sentenceList1) + len(sentenceList2))
	
	baselines = []
	if baselineprobLang2 > baselineprobLang1:
		for i in range (len(actualLang)):
			baselines.append ('Lang2')
	else:
		for i in range(len(actualLang)):
			baselines.append ('Lang1')

	#note that this baseline returns a warning because I never predict one of the two classes in this baseline sample
	#the warning can be ignored
	precisionbase = precision_score(actualLang, baselines,average = "macro")
	
	recallbase = recall_score(actualLang, baselines,average = "macro")
	
	print("\nFollowing is for baseline:")
	
	print("\nPrecision is:")
	print(precisionbase)
	
	print("Recall is:")
	print(recallbase)

	f1Scorebase = (2*precisionbase*recallbase)/(precisionbase+recallbase)
	print("F1Score is:")
	print(f1Scorebase)
	
    ###########change up how serialization works to make up for changes to my code above
    ########### commenting out for now
    
	#####allow option to not serialize. If path with ignore case is equal to the string none, skip serialization.
	####if not (str(pathToSerialize).lower()=='none'):
	####now serialize my model
	#####I am passing along features needed to recreate my data structures as opposed to passing all of my data structures.
		
	#####In hindsight this is a mome memory efficient but sometimes less runtime efficient method for creating models and deserializing models.
	#####In cases like katz backoff where my files were almost 700 mb, this might have been useful as the files were a tad bit big. I did somewhat use this for katz 
	#####but could have saved more space. On the other hand my deserialization runtime is very quick for katz which is nice. It is a tradeoff. 
	
	#####Create inner container dataframe.
	####tempDf = pd.DataFrame(index =['Container'], columns = [0,1,2,3,4,5,6,7,8] )
	#####lists of word counts
	####tempDf.iat[0,0] = filteredList1
	####tempDf.iat[0,1] = filteredList2
	####tempDf.iat[0,2] = laplace
	####tempDf.iat[0,3] = bigramCounts1
	####tempDf.iat[0,4]= bigramCounts2
	####tempDf.iat[0,5] = probLang1
	####tempDf.iat[0,6] = probLang2
	####tempDf.iat[0,7] = sentenceList1
	####tempDf.iat[0,8] = sentenceList2

    #####serialize via pickle
	####tempDf.to_pickle(pathToSerialize)
    
#check if correct number of arguments 
if (len(sys.argv) <7  ) :
	print("Incorrect number of arguments for the script")

else:
	if len(sys.argv) >= 8:
		bigramLaplace(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],sys.argv[6],sys.argv[7])
	elif len(sys.argv) == 7: 
		bigramLaplace(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
	else:
		bigramLaplace(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],sys.argv[5])