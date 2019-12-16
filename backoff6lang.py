from itertools import islice

from io import open
from conllu import parse_incr
import pandas as pd
import numpy as np
from collections import Counter 
import string 
import math

#for deep copy
import copy

#for commandline input
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

##### STILL NEED TO REMOVE OOV CUTOFF MAYBE

def bigramLaplace(pathToTrainLang1, pathToTrainLang2, pathToTrainLang3, pathToTrainLang4, pathToTrainLang5, pathToTrainLang6
    , pathToTuneLang1, pathToTuneLang2, pathToTuneLang3, pathToTuneLang4, pathToTuneLang5, pathToTuneLang6, uniBackoff= .5, biBackoff= .5):
    
    #not error checking the input. presume user inputs correct values below 1 and above 0
    uniBackoff = float(uniBackoff)
    biBackoff = float(biBackoff)
    
    #removed oov constant implementation 
    #I am populating some unknown token list with a minimal training set of tokens with frequency one. 
    #This is because having no probability for unknown tokens is bad form in my opinion and this distributes at least a small probability to <UNK>
    #if int(oovConstant)>=1:
    #    oovFrequency = int(oovConstant)
    #else:
    #    oovFrequency = 1
    
    #get laplace constant from hyperparamters. cannot be 0 or less. should probably be less than 1 for better performance
    #if float(laplaceConstant)> 0:
    #    laplace = float (laplaceConstant)
    #else:
    #    laplace = .1

    #open both files 
    train1 = open(pathToTrainLang1, "r", encoding="utf-8")
    train2 = open(pathToTrainLang2, "r", encoding="utf-8")
    train3 = open(pathToTrainLang3, "r", encoding="utf-8")
    train4= open(pathToTrainLang4, "r", encoding="utf-8")
    train5 = open(pathToTrainLang5, "r", encoding="utf-8") 
    train6 = open(pathToTrainLang6, "r", encoding="utf-8") 

    #used as temporary storage per each sentence as the connlu parser iterates over them
    tempSentence = list()

    #list for storing word tokens
    list1 = list()
    list2 = list()
    list3 = list()
    list4 = list()
    list5 = list()
    list6=list()
    
    #at first storing observed bigrams using dictionary 
    bigramCounts1 = {}
    bigramCounts2 = {}
    bigramCounts3 = {}
    bigramCounts4 = {}
    bigramCounts5 = {}
    bigramCounts6 = {}

    #list for storing sentences which will later be used to update bigram
    sentenceList1 = list()
    sentenceList2 = list()
    sentenceList3 = list()
    sentenceList4 = list()
    sentenceList5 = list()
    sentenceList6 = list()
    
    #word/token counts in order to calculate unigram probabilities and some other things
    wordCount1 = 0
    wordCount2 = 0
    wordCount3 = 0
    wordCount4 = 0
    wordCount5 = 0
    wordCount6 = 0
    
    print("Reading in data from connlu files\n")
    
    #connlu parse and update bigram and unigram counts 
    for tokenlist in parse_incr(train1):
        for token in tokenlist:
            ##adding to temporary sentence which will be parsed into bigrams
            #making it lower case as a means of preprocessing. words of different case but same spelling are the same type for my purposes
            tempSentence.append(token["form"].lower())
            #adding to list of tokens. this will later be used to get unigram counts 
            list1.append(token["form"].lower())
            wordCount1+=1
            
        #now adding eos and bos tags to the sentence
        tempSentence.insert(0, "<BOS>")
        tempSentence.append("<EOS>")
        
        #now adding them to unigram token list 
        list1.append("<BOS>")
        list1.append("<EOS>")
        wordCount1+=2

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
            wordCount2+=1
            
        #now adding eos and bos tags to the sentence
        tempSentence.insert(0, "<BOS>")
        tempSentence.append("<EOS>")
        
        #now adding them to unigram token list 
        list2.append("<BOS>")
        list2.append("<EOS>")
        wordCount2+=2
        
        #count of sentences in language two used for initial probability
        #numSentences2+=1

        #add the parsed sentence list of words to nested list of sentences
        sentenceList2.append(tempSentence)

        #resetting the temporary list of words per each sentence 
        tempSentence = []

    train2.close()
    
    tempSentence = []
    
    #connlu parse and update bigram and unigram counts 
    for tokenlist in parse_incr(train3):
        for token in tokenlist:
            ##adding to temporary sentence which will be parsed into bigrams
            #making it lower case as a means of preprocessing. words of different case but same spelling are the same type for my purposes
            tempSentence.append(token["form"].lower())
            #adding to list of tokens. this will later be used to get unigram counts 
            list3.append(token["form"].lower())
            wordCount3+=1
            
        #now adding eos and bos tags to the sentence
        tempSentence.insert(0, "<BOS>")
        tempSentence.append("<EOS>")
        
        #now adding them to unigram token list 
        list3.append("<BOS>")
        list3.append("<EOS>")
        wordCount3+=2

        #count of sentences in language one used for initial probability
        #numSentences1+=1
        
        #add the parsed sentence list of words to nested list of sentences
        sentenceList3.append(tempSentence)

        #resetting the temporary list of words per each sentence 
        tempSentence = []

    train3.close()
    
    tempSentence = []
    
    #connlu parse and update bigram and unigram counts 
    for tokenlist in parse_incr(train4):
        for token in tokenlist:
            ##adding to temporary sentence which will be parsed into bigrams
            #making it lower case as a means of preprocessing. words of different case but same spelling are the same type for my purposes
            tempSentence.append(token["form"].lower())
            #adding to list of tokens. this will later be used to get unigram counts 
            list4.append(token["form"].lower())
            wordCount4+=1
            
        #now adding eos and bos tags to the sentence
        tempSentence.insert(0, "<BOS>")
        tempSentence.append("<EOS>")

        #now adding them to unigram token list 
        list4.append("<BOS>")
        list4.append("<EOS>")
        wordCount4+=2

        #count of sentences in language one used for initial probability
        #numSentences1+=1
        
        #add the parsed sentence list of words to nested list of sentences
        sentenceList4.append(tempSentence)

        #resetting the temporary list of words per each sentence 
        tempSentence = []

    train4.close()
    
    tempSentence = []
    
    #connlu parse and update bigram and unigram counts 
    for tokenlist in parse_incr(train5):
        for token in tokenlist:
            ##adding to temporary sentence which will be parsed into bigrams
            #making it lower case as a means of preprocessing. words of different case but same spelling are the same type for my purposes
            tempSentence.append(token["form"].lower())
            #adding to list of tokens. this will later be used to get unigram counts 
            list5.append(token["form"].lower())
            wordCount5+=1
            
        #now adding eos and bos tags to the sentence
        tempSentence.insert(0, "<BOS>")
        tempSentence.append("<EOS>")
        
        #now adding them to unigram token list 
        list5.append("<BOS>")
        list5.append("<EOS>")
        wordCount5+=2

        #count of sentences in language one used for initial probability
        #numSentences1+=1
        
        #add the parsed sentence list of words to nested list of sentences
        sentenceList5.append(tempSentence)

        #resetting the temporary list of words per each sentence 
        tempSentence = []

    train5.close()
    
    tempSentence = []
    
    #connlu parse and update bigram and unigram counts 
    for tokenlist in parse_incr(train6):
        for token in tokenlist:
            ##adding to temporary sentence which will be parsed into bigrams
            #making it lower case as a means of preprocessing. words of different case but same spelling are the same type for my purposes
            tempSentence.append(token["form"].lower())
            #adding to list of tokens. this will later be used to get unigram counts 
            list6.append(token["form"].lower())
            wordCount6+=1
            
        #now adding eos and bos tags to the sentence
        tempSentence.insert(0, "<BOS>")
        tempSentence.append("<EOS>")
        
        #now adding them to unigram token list 
        list6.append("<BOS>")
        list6.append("<EOS>")
        wordCount6+=2

        #count of sentences in language one used for initial probability
        #numSentences1+=1
        
        #add the parsed sentence list of words to nested list of sentences
        sentenceList6.append(tempSentence)

        #resetting the temporary list of words per each sentence 
        tempSentence = []

    train6.close()
    
    tempSentence = []

    #create dataframe containing all tokens and convert to series of counts per word type.
    df1 = pd.DataFrame(list1)
    series1= df1[0].value_counts()
    
    df2 = pd.DataFrame(list2)
    series2= df2[0].value_counts()
    
    df3 = pd.DataFrame(list3)
    series3= df3[0].value_counts()
    
    df4 = pd.DataFrame(list4)
    series4= df4[0].value_counts()
    
    df5 = pd.DataFrame(list5)
    series5= df5[0].value_counts()
    
    df6 = pd.DataFrame(list6)
    series6= df6[0].value_counts()
    
    #setting aside some count of 1 for unknown. smallest possible whole word count 
    series1.at['<UNK>'] = 1
    series2.at['<UNK>'] = 1
    series3.at['<UNK>'] = 1
    series4.at['<UNK>'] = 1
    series5.at['<UNK>'] = 1
    series6.at['<UNK>'] = 1
    
    """
    #left in previous code for oov cutoff. using 1 instead of oovfreqency hyperparameter cutoff variable
    #decided to not include the feature at this time due to time constraints for evaluating hyperparameters.
    
    #for storing filtered vocab filtered by frequency 
    filteredList1 = pd.Series()
    #extract frequencies below oovFrequency cutoff constant and set them to some unknown token
    unknownCount = 0
    for index, value in series1.items():
        if value > 0:
            filteredList1.at[index] = value
        else:
            unknownCount+=value
            
    ####### maybe do this differently? unsure
    #add unknown count to list with token <UNK> as the index 
    #filteredList1.at['<UNK>'] = laplace

    #####add unknown count to list with token <UNK> as the index
    #####I am commenting this out because I am not assigning probability to unknown this way.
    #####Instead I will be using laplace constant.
    #####filteredList1.at['<UNK>'] = unknownCount
    
    #for storing filtered vocab filtered by frequency 
    filteredList2 = pd.Series()
    #extract frequencies below oovFrequency cutoff constant and set them to some unknown token
    unknownCount = 0
    for index, value in series2.items():
        if value > 0:
            filteredList2.at[index] = value
        else:
            unknownCount+=value
            
    ####### maybe do this differently? unsure
    #add unknown count to list with token <UNK> as the index 
    #filteredList2.at['<UNK>'] = laplace

    #####add unknown count to list with token <UNK> as the index
    #####I am commenting this out because I am not assigning probability to unknown this way.
    #####Instead I will be using laplace constant.
    #####filteredList2.at['<UNK>'] = unknownCount. 

    #for storing filtered vocab filtered by frequency 
    filteredList3 = pd.Series()
    #extract frequencies below oovFrequency cutoff constant and set them to some unknown token
    unknownCount = 0
    for index, value in series3.items():
        if value > 0:
            filteredList3.at[index] = value
        else:
            unknownCount+=value
            
    ####### maybe do this differently? unsure
    #add unknown count to list with token <UNK> as the index 
    #filteredList3.at['<UNK>'] = laplace
    
    #for storing filtered vocab filtered by frequency 
    filteredList4 = pd.Series()
    #extract frequencies below oovFrequency cutoff constant and set them to some unknown token
    unknownCount = 0
    for index, value in series4.items():
        if value > 0:
            filteredList4.at[index] = value
        else:
            unknownCount+=value
            
    ####### maybe do this differently? unsure
    #add unknown count to list with token <UNK> as the index 
    #filteredList4.at['<UNK>'] = laplace
    
    #for storing filtered vocab filtered by frequency 
    filteredList5 = pd.Series()
    #extract frequencies below oovFrequency cutoff constant and set them to some unknown token
    unknownCount = 0
    for index, value in series5.items():
        if value > 0:
            filteredList5.at[index] = value
        else:
            unknownCount+=value
            
    ####### maybe do this differently? unsure
    #add unknown count to list with token <UNK> as the index 
    #filteredList5.at['<UNK>'] = laplace
    
    #for storing filtered vocab filtered by frequency 
    filteredList6 = pd.Series()
    #extract frequencies below oovFrequency cutoff constant and set them to some unknown token
    unknownCount = 0
    for index, value in series6.items():
        if value > 0:
            filteredList6.at[index] = value
        else:
            unknownCount+=value
            
    ####### maybe do this differently? unsure
    #add unknown count to list with token <UNK> as the index 
    #filteredList6.at['<UNK>'] = laplace
    
    """
    print("Start of unigram backoff\n")
    
    #store unigram count copy for backoff
    backedOffList1 = series1.copy()
    backedOffList2 = series2.copy()
    backedOffList3 = series3.copy()
    backedOffList4 = series4.copy()
    backedOffList5 = series5.copy()
    backedOffList6 = series6.copy()
    
    #for debugging
    #print(backedOffList1)
    
    #perform backoff by backoff constant on every unigram entry
    backedOffList1-= uniBackoff
    backedOffList2-=uniBackoff
    backedOffList3-=uniBackoff
    backedOffList4-=uniBackoff
    backedOffList5-=uniBackoff
    backedOffList6-=uniBackoff
    
    #for debugging
    #print(backedOffList1)
    
    #calculate how much value was backed off in total per each language unigram
    #this is 1 - sum (c(y) - d1 / count(tokens))
    redistributedUnigram1 = (1- (backedOffList1.values.sum()/wordCount1))
    redistributedUnigram2 = (1- (backedOffList2.values.sum()/wordCount2))
    redistributedUnigram3 = (1- (backedOffList3.values.sum()/wordCount3))
    redistributedUnigram4 = (1- (backedOffList4.values.sum()/wordCount4))
    redistributedUnigram5 = (1- (backedOffList5.values.sum()/wordCount5))
    redistributedUnigram6 = (1- (backedOffList6.values.sum()/wordCount6))
    
    #for debugging
    #print(redistributedUnigram1)

    #for storing list of words (types) in the vocab which will be used for indexing the rows and columns of the dataframe
    wordList1 = series1.keys().tolist()
    wordList2 = series2.keys().tolist()
    wordList3 = series3.keys().tolist()
    wordList4 = series4.keys().tolist()
    wordList5 = series5.keys().tolist()
    wordList6 = series6.keys().tolist()

    #get number of types
    sizeOfVocab1=len(wordList1)
    sizeOfVocab2=len(wordList2)
    sizeOfVocab3=len(wordList3)
    sizeOfVocab4=len(wordList4)
    sizeOfVocab5=len(wordList5)
    sizeOfVocab6=len(wordList6)

    print("Creating sparse bigram counts\n")
    
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
                bigramCounts1[(wordPair[0], wordPair[1])] = 1.0
                
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
                bigramCounts2[(wordPair[0], wordPair[1])] = 1.0
                
    #filter out unknown words and make observed bigrams into dictionary 
    for tempSentence in sentenceList3:
        #first filtering out the out of vocab words 
        for i in range( len(tempSentence)):
            if not tempSentence[i] in wordList3: 
                tempSentence[i] = "<UNK>"
        #parsing bigrams by pythonically creating them with islice and zip 
        tempBigram = zip(tempSentence, islice(tempSentence, 1, None))
        #iterating over created list of bigrams and adding new ones to the dictionary while incrementing counts for existing bigrams 
        for wordPair in tempBigram :
            if (wordPair[0], wordPair[1]) in bigramCounts3:
                bigramCounts3[(wordPair[0], wordPair[1])] += 1
            else:
                bigramCounts3[(wordPair[0], wordPair[1])] = 1.0
                
    #filter out unknown words and make observed bigrams into dictionary 
    for tempSentence in sentenceList4:
        #first filtering out the out of vocab words 
        for i in range( len(tempSentence)):
            if not tempSentence[i] in wordList4: 
                tempSentence[i] = "<UNK>"
        #parsing bigrams by pythonically creating them with islice and zip 
        tempBigram = zip(tempSentence, islice(tempSentence, 1, None))
        #iterating over created list of bigrams and adding new ones to the dictionary while incrementing counts for existing bigrams 
        for wordPair in tempBigram :
            if (wordPair[0], wordPair[1]) in bigramCounts4:
                bigramCounts4[(wordPair[0], wordPair[1])] += 1
            else:
                bigramCounts4[(wordPair[0], wordPair[1])] = 1.0
    #filter out unknown words and make observed bigrams into dictionary 
    for tempSentence in sentenceList5:
        #first filtering out the out of vocab words 
        for i in range( len(tempSentence)):
            if not tempSentence[i] in wordList5: 
                tempSentence[i] = "<UNK>"
        #parsing bigrams by pythonically creating them with islice and zip 
        tempBigram = zip(tempSentence, islice(tempSentence, 1, None))
        #iterating over created list of bigrams and adding new ones to the dictionary while incrementing counts for existing bigrams 
        for wordPair in tempBigram :
            if (wordPair[0], wordPair[1]) in bigramCounts5:
                bigramCounts5[(wordPair[0], wordPair[1])] += 1
            else:
                bigramCounts5[(wordPair[0], wordPair[1])] = 1.0

    #filter out unknown words and make observed bigrams into dictionary 
    for tempSentence in sentenceList6:
        #first filtering out the out of vocab words 
        for i in range( len(tempSentence)):
            if not tempSentence[i] in wordList6: 
                tempSentence[i] = "<UNK>"
        #parsing bigrams by pythonically creating them with islice and zip 
        tempBigram = zip(tempSentence, islice(tempSentence, 1, None))
        #iterating over created list of bigrams and adding new ones to the dictionary while incrementing counts for existing bigrams 
        for wordPair in tempBigram :
            if (wordPair[0], wordPair[1]) in bigramCounts6:
                bigramCounts6[(wordPair[0], wordPair[1])] += 1
            else:
                bigramCounts6[(wordPair[0], wordPair[1])] = 1.0
    
    print("Start of bigram backoff\n")
    
    #convert these bigram count dictionaries to series so I can subtract from the series in pandas          
    backedOffBigram1 = pd.Series(bigramCounts1)
    backedOffBigram2 = pd.Series(bigramCounts2)
    backedOffBigram3 = pd.Series(bigramCounts3)
    backedOffBigram4 = pd.Series(bigramCounts4)
    backedOffBigram5 = pd.Series(bigramCounts5)
    backedOffBigram6 = pd.Series(bigramCounts6)
    
    #dictionary for holding redistributed value for every word. i.e. for holding a(x) = 1 - (∑(c(x,y) - δ2)/( c(x))) for all x words
    #start out with value of one and then subtract accordingly
    redistributedBigram1 =  dict.fromkeys(wordList1, 1.0)
    for key, value in backedOffBigram1.items():
        #backoff by bigram backoff constant
        backedOffBigram1[key] -= biBackoff
        
        #get key of x value in (x,y) bigram
        tempWordX = key[0]
        
        #subtract from list of redistributed bigram values per word x as per the formula 1 - (c(x,y) - d2 / c(x))
        #by looping I am able to subtract the summation of these existing (c(x,y) - d2 / c(x)) values
        redistributedBigram1[tempWordX]-= (backedOffBigram1[key]/series1[tempWordX])
        
    redistributedBigram2 =  dict.fromkeys(wordList2, 1.0)
    for key, value in backedOffBigram2.items():
        #backoff by bigram backoff constant
        backedOffBigram2[key] -= biBackoff
        
        #get key of x value in (x,y) bigram
        tempWordX = key[0]
        
        #subtract from list of redistributed bigram values per word x as per the formula 1 - (c(x,y) - d2 / c(x))
        #by looping I am able to subtract the summation of these existing (c(x,y) - d2 / c(x)) values
        redistributedBigram2[tempWordX]-= (backedOffBigram2[key]/series2[tempWordX])
    
    redistributedBigram3 =  dict.fromkeys(wordList3, 1.0)
    for key, value in backedOffBigram3.items():
        #backoff by bigram backoff constant
        backedOffBigram3[key] -= biBackoff
        
        #get key of x value in (x,y) bigram
        tempWordX = key[0]
        
        #subtract from list of redistributed bigram values per word x as per the formula 1 - (c(x,y) - d2 / c(x))
        #by looping I am able to subtract the summation of these existing (c(x,y) - d2 / c(x)) values
        redistributedBigram3[tempWordX]-= (backedOffBigram3[key]/series3[tempWordX])
        
    redistributedBigram4 =  dict.fromkeys(wordList4, 1.0)
    for key, value in backedOffBigram4.items():
        #backoff by bigram backoff constant
        backedOffBigram4[key] -= biBackoff
        
        #get key of x value in (x,y) bigram
        tempWordX = key[0]
        
        #subtract from list of redistributed bigram values per word x as per the formula 1 - (c(x,y) - d2 / c(x))
        #by looping I am able to subtract the summation of these existing (c(x,y) - d2 / c(x)) values
        redistributedBigram4[tempWordX]-= (backedOffBigram4[key]/series4[tempWordX])
        
    redistributedBigram5 =  dict.fromkeys(wordList5, 1.0)
    for key, value in backedOffBigram5.items():
        #backoff by bigram backoff constant
        backedOffBigram5[key] -= biBackoff
        
        #get key of x value in (x,y) bigram
        tempWordX = key[0]
        
        #subtract from list of redistributed bigram values per word x as per the formula 1 - (c(x,y) - d2 / c(x))
        #by looping I am able to subtract the summation of these existing (c(x,y) - d2 / c(x)) values
        redistributedBigram5[tempWordX]-= (backedOffBigram5[key]/series5[tempWordX])
        
    redistributedBigram6 =  dict.fromkeys(wordList6, 1.0)
    for key, value in backedOffBigram6.items():
        #backoff by bigram backoff constant
        backedOffBigram6[key] -= biBackoff
        
        #get key of x value in (x,y) bigram
        tempWordX = key[0]
        
        #subtract from list of redistributed bigram values per word x as per the formula 1 - (c(x,y) - d2 / c(x))
        #by looping I am able to subtract the summation of these existing (c(x,y) - d2 / c(x)) values
        redistributedBigram6[tempWordX]-= (backedOffBigram6[key]/series6[tempWordX])
        
    #for debugging
    #print(backedOffBigram6)
    #print(redistributedBigram6)
    
    ####get entire vocab. maybe don't need this code
    wordList = wordList1+wordList2+wordList3+wordList4+wordList5+wordList6

    #initial probability is using simply count of sentences in one language over total count of sentences
    probLang1 = len(sentenceList1) / (len(sentenceList1) + len(sentenceList2) + len(sentenceList3) + len(sentenceList4) + len(sentenceList5) + len(sentenceList6))
    probLang2 = len(sentenceList2) / (len(sentenceList1) + len(sentenceList2) + len(sentenceList3) + len(sentenceList4) + len(sentenceList5) + len(sentenceList6))
    probLang3 = len(sentenceList3) / (len(sentenceList1) + len(sentenceList2) + len(sentenceList3) + len(sentenceList4) + len(sentenceList5) + len(sentenceList6))
    probLang4 = len(sentenceList4) / (len(sentenceList1) + len(sentenceList2) + len(sentenceList3) + len(sentenceList4) + len(sentenceList5) + len(sentenceList6))
    probLang5 = len(sentenceList5) / (len(sentenceList1) + len(sentenceList2) + len(sentenceList3) + len(sentenceList4) + len(sentenceList5) + len(sentenceList6))
    probLang6 = len(sentenceList6) / (len(sentenceList1) + len(sentenceList2) + len(sentenceList3) + len(sentenceList4) + len(sentenceList5) + len(sentenceList6))
    
    #initial  probabilities
    print("\nInitial probability of the first language passed to this program:")
    print(probLang1)
    print("Initial probability of the second language passed to this program:")
    print(probLang2)
    print("Initial probability of the third language passed to this program:")
    print(probLang3)
    print("Initial probability of the fourth language passed to this program:")
    print(probLang4)
    print("Initial probability of the fifth language passed to this program:")
    print(probLang5)
    print("Initial probability of the sixth language passed to this program:")
    print(probLang6)
    
    print("\nEvaluating development or training set on data\n")
    
    #evaluate on dev set (or look at results on test set if you input test set file paths)
    en_dev1 = open(pathToTuneLang1, "r", encoding="utf-8")
    en_dev2 = open(pathToTuneLang2, "r", encoding="utf-8")
    en_dev3 = open(pathToTuneLang3, "r", encoding="utf-8")
    en_dev4 = open(pathToTuneLang4, "r", encoding="utf-8")
    en_dev5 = open(pathToTuneLang5, "r", encoding="utf-8")
    en_dev6 = open(pathToTuneLang6, "r", encoding="utf-8")
    
    tempSentence1 = list()
    tempSentence2 = list()
    tempSentence3 = list()
    tempSentence4 = list()
    tempSentence5 = list()
    tempSentence6 = list()
    
    #lists for storing predicted languages vs actual langs
    predictedLang = []
    actualLang = []
    
    #for tracking and printing progress through dev set
    countDevLang1=0
    countDevLang2=0
    countDevLang3=0
    countDevLang4=0
    countDevLang5=0
    countDevLang6=0
    
    #connlu parse and update bigram and unigram counts 
    for tokenlist in parse_incr(en_dev1):
        for token in tokenlist:         
            #If sentence is in entire vocab from all languages, add the word to the sentence. Otherwise add the unknown token to the sentence.
            #Regardless, if the bigram doesn't exist in a specific language one will have laplace + 0 bigram count /unigram count+ laplace + V. 
            #If the unigram doesn't exist, one will have laplace + 0 bigram count / laplace + 0 + V. If the bigram exists one instead has bigram count+laplace
            # divided by unigram count + laplace + V
            if not (token["form"].lower()) in wordList :
                tempSentence1.append('<UNK>')
                tempSentence2.append('<UNK>')
                tempSentence3.append('<UNK>')
                tempSentence4.append('<UNK>')
                tempSentence5.append('<UNK>')
                tempSentence6.append('<UNK>')
            else:
                ##adding to temporary sentence which will be parsed into bigrams
                #making it lower case as a means of preprocessing. words of different case but same spelling are the same type for my purposes
                tempSentence1.append(token["form"].lower())
                tempSentence2.append(token["form"].lower())
                tempSentence3.append(token["form"].lower())
                tempSentence4.append(token["form"].lower())
                tempSentence5.append(token["form"].lower())
                tempSentence6.append(token["form"].lower())


        #now adding eos and bos tags to the sentence
        tempSentence1.insert(0, "<BOS>")
        tempSentence1.append("<EOS>")
        tempSentence2.insert(0, "<BOS>")
        tempSentence2.append("<EOS>")
        tempSentence3.insert(0, "<BOS>")
        tempSentence3.append("<EOS>")
        tempSentence4.insert(0, "<BOS>")
        tempSentence4.append("<EOS>")
        tempSentence5.insert(0, "<BOS>")
        tempSentence5.append("<EOS>")
        tempSentence6.insert(0, "<BOS>")
        tempSentence6.append("<EOS>")
        
        #this is math.log(1) since I am adding log probabilities to avoid multiplication
        probSentenceGivenL1 = 0
        probSentenceGivenL2 = 0
        probSentenceGivenL3 = 0
        probSentenceGivenL4 = 0
        probSentenceGivenL5 = 0
        probSentenceGivenL6 = 0
        
        #for zipping to bigram tuples
        tempBigram1 = zip(tempSentence1, islice(tempSentence1, 1, None))
        tempBigram2 = zip(tempSentence2, islice(tempSentence2, 1, None))
        tempBigram3 = zip(tempSentence3, islice(tempSentence3, 1, None))
        tempBigram4 = zip(tempSentence4, islice(tempSentence4, 1, None))
        tempBigram5 = zip(tempSentence5, islice(tempSentence5, 1, None))
        tempBigram6 = zip(tempSentence6, islice(tempSentence6, 1, None))
        
        #for storing values that will later have redistributed unigram probability
        tempUnigramBackoffList1 = {}
        tempUnigramBackoffList2 = {}
        tempUnigramBackoffList3 = {}
        tempUnigramBackoffList4 = {}
        tempUnigramBackoffList5 = {}
        tempUnigramBackoffList6 = {}
        
        for wordPair in tempBigram1 :
            #if c(x,y)>0 just count probability as c(x,y)-d2 / c(x)
            if wordPair in backedOffBigram1:
                normalizingConstant = (series1[wordPair[0]])
                tempProbability = (backedOffBigram1[wordPair])/(normalizingConstant)
                probSentenceGivenL1+=math.log(tempProbability)
            else:
                #otherwise check if c(x)>0 and if it is, count a(x) * (c(x)/ summation of c(v) across all V) as the probability  
                if wordPair[0] in backedOffList1:
                    tempProbability = redistributedBigram1[wordPair[0]] *( backedOffList1[wordPair[0]] / wordCount1)
                    probSentenceGivenL1+=math.log(tempProbability)
                else:
                    #otherwise add the word pair as a key to a list of double backed off values
                    if wordPair in tempUnigramBackoffList1:
                        tempUnigramBackoffList1[wordPair] += 1
                    else:
                        tempUnigramBackoffList1[wordPair] = 1

        for wordPair in tempBigram2 :
            #if c(x,y)>0 just count probability as c(x,y)-d2 / c(x)
            if wordPair in backedOffBigram2:
                normalizingConstant = (series2[wordPair[0]])
                tempProbability = (backedOffBigram2[wordPair])/(normalizingConstant)
                probSentenceGivenL2+=math.log(tempProbability)
            else:
                #otherwise check if c(x)>0 and if it is, count a(x) * (c(x)/ summation of c(v) across all V) as the probability  
                if wordPair[0] in backedOffList2:
                    tempProbability = redistributedBigram2[wordPair[0]] *( backedOffList2[wordPair[0]] / wordCount2)
                    probSentenceGivenL2+=math.log(tempProbability)
                else:
                    #otherwise add the word pair as a key to a list of double backed off values
                    if wordPair in tempUnigramBackoffList2:
                        tempUnigramBackoffList2[wordPair] += 1
                    else:
                        tempUnigramBackoffList2[wordPair] = 1
                
        for wordPair in tempBigram3 :
            #if c(x,y)>0 just count probability as c(x,y)-d2 / c(x)
            if wordPair in backedOffBigram3:
                normalizingConstant = (series3[wordPair[0]])
                tempProbability = (backedOffBigram3[wordPair])/(normalizingConstant)
                probSentenceGivenL3+=math.log(tempProbability)
            else:
                #otherwise check if c(x)>0 and if it is, count a(x) * (c(x)/ summation of c(v) across all V) as the probability  
                if wordPair[0] in backedOffList3:
                    tempProbability = redistributedBigram3[wordPair[0]] *( backedOffList3[wordPair[0]] / wordCount3)
                    probSentenceGivenL3+=math.log(tempProbability)
                else:
                    #otherwise add the word pair as a key to a list of double backed off values
                    if wordPair in tempUnigramBackoffList3:
                        tempUnigramBackoffList3[wordPair] += 1
                    else:
                        tempUnigramBackoffList3[wordPair] = 1
                
        for wordPair in tempBigram4 :
            #if c(x,y)>0 just count probability as c(x,y)-d2 / c(x)
            if wordPair in backedOffBigram4:
                normalizingConstant = (series4[wordPair[0]])
                tempProbability = (backedOffBigram4[wordPair])/(normalizingConstant)
                probSentenceGivenL4+=math.log(tempProbability)
            else:
                #otherwise check if c(x)>0 and if it is, count a(x) * (c(x)/ summation of c(v) across all V) as the probability  
                if wordPair[0] in backedOffList4:
                    tempProbability = redistributedBigram4[wordPair[0]] *( backedOffList4[wordPair[0]] / wordCount4)
                    probSentenceGivenL4+=math.log(tempProbability)
                else:
                    #otherwise add the word pair as a key to a list of double backed off values
                    if wordPair in tempUnigramBackoffList4:
                        tempUnigramBackoffList4[wordPair] += 1
                    else:
                        tempUnigramBackoffList4[wordPair] = 1
                
        for wordPair in tempBigram2 :
            #if c(x,y)>0 just count probability as c(x,y)-d2 / c(x)
            if wordPair in backedOffBigram5:
                normalizingConstant = (series5[wordPair[0]])
                tempProbability = (backedOffBigram5[wordPair])/(normalizingConstant)
                probSentenceGivenL5+=math.log(tempProbability)
            else:
                #otherwise check if c(x)>0 and if it is, count a(x) * (c(x)/ summation of c(v) across all V) as the probability  
                if wordPair[0] in backedOffList5:
                    tempProbability = redistributedBigram5[wordPair[0]] *( backedOffList5[wordPair[0]] / wordCount5)
                    probSentenceGivenL5+=math.log(tempProbability)
                else:
                    #otherwise add the word pair as a key to a list of double backed off values
                    if wordPair in tempUnigramBackoffList5:
                        tempUnigramBackoffList5[wordPair] += 1
                    else:
                        tempUnigramBackoffList5[wordPair] = 1
                
        for wordPair in tempBigram6 :
            #if c(x,y)>0 just count probability as c(x,y)-d2 / c(x)
            if wordPair in backedOffBigram6:
                normalizingConstant = (series6[wordPair[0]])
                tempProbability = (backedOffBigram6[wordPair])/(normalizingConstant)
                probSentenceGivenL6+=math.log(tempProbability)
            else:
                #otherwise check if c(x)>0 and if it is, count a(x) * (c(x)/ summation of c(v) across all V) as the probability  
                if wordPair[0] in backedOffList6:
                    tempProbability = redistributedBigram6[wordPair[0]] *( backedOffList6[wordPair[0]] / wordCount6)
                    probSentenceGivenL6+=math.log(tempProbability)
                else:
                    #otherwise add the word pair as a key to a list of double backed off values
                    if wordPair in tempUnigramBackoffList6:
                        tempUnigramBackoffList6[wordPair] += 1
                    else:
                        tempUnigramBackoffList6[wordPair] = 1
        
        #redistribute the probabilities that had to be backed off twice.
        print("Now redistributing unigram backoff as necessary.\n")
        
        #First need to count number of oov types
        numOOVTypes1 = len(tempUnigramBackoffList1)
        numOOVTypes2 = len(tempUnigramBackoffList2)
        numOOVTypes3 = len(tempUnigramBackoffList3)
        numOOVTypes4 = len(tempUnigramBackoffList4)
        numOOVTypes5 = len(tempUnigramBackoffList5)
        numOOVTypes6 = len(tempUnigramBackoffList6)
        
        #then need to get b/v values where V is the size of the total vocab of known and previously unknown newly encountered words
        #b is the probability mass set aside for redistribution
        distValueUnigram1 = redistributedUnigram1/(numOOVTypes1+sizeOfVocab1)
        distValueUnigram2 = redistributedUnigram2/(numOOVTypes2+sizeOfVocab2)
        distValueUnigram3 = redistributedUnigram3/(numOOVTypes3+sizeOfVocab3)
        distValueUnigram4 = redistributedUnigram4/(numOOVTypes4+sizeOfVocab4)
        distValueUnigram5 = redistributedUnigram5/(numOOVTypes5+sizeOfVocab5)
        distValueUnigram6 = redistributedUnigram6/(numOOVTypes6+sizeOfVocab6)

        #Now multiply this b/v value by its a(x) and add its log linear probability up as necessary per each occurence/count of double backed off bigram
        for key, value in tempUnigramBackoffList1.items():
            if key[0] in redistributedBigram1:
                tempProbabilitySum = value * math.log(redistributedBigram1[key[0]]*distValueUnigram1)
            else:
                tempProbabilitySum = value * math.log(redistributedBigram1['<UNK>']*distValueUnigram1)
            probSentenceGivenL1+=tempProbabilitySum
            
        #Now multiply this b/v value by its a(x) and add its log linear probability up as necessary per each occurence/count of double backed off bigram
        for key, value in tempUnigramBackoffList2.items():
            if key[0] in redistributedBigram2:
                tempProbabilitySum = value * math.log(redistributedBigram2[key[0]]*distValueUnigram2)
            else:
                tempProbabilitySum = value * math.log(redistributedBigram2['<UNK>']*distValueUnigram2)
            probSentenceGivenL2+=tempProbabilitySum
            
        #Now multiply this b/v value by its a(x) and add its log linear probability up as necessary per each occurence/count of double backed off bigram
        for key, value in tempUnigramBackoffList3.items():
            if key[0] in redistributedBigram3:
                tempProbabilitySum = value * math.log(redistributedBigram3[key[0]]*distValueUnigram3)
            else:
                tempProbabilitySum = value * math.log(redistributedBigram3['<UNK>']*distValueUnigram3)
            probSentenceGivenL3+=tempProbabilitySum
            
            
        #Now multiply this b/v value by its a(x) and add its log linear probability up as necessary per each occurence/count of double backed off bigram
        for key, value in tempUnigramBackoffList4.items():
            if key[0] in redistributedBigram4:
                tempProbabilitySum = value * math.log(redistributedBigram4[key[0]]*distValueUnigram4)
            else:
                tempProbabilitySum = value * math.log(redistributedBigram4['<UNK>']*distValueUnigram4)
            probSentenceGivenL4+=tempProbabilitySum
            
        #Now multiply this b/v value by its a(x) and add its log linear probability up as necessary per each occurence/count of double backed off bigram
        for key, value in tempUnigramBackoffList5.items():
            if key[0] in redistributedBigram5:
                tempProbabilitySum = value * math.log(redistributedBigram5[key[0]]*distValueUnigram5)
            else:
                tempProbabilitySum = value * math.log(redistributedBigram5['<UNK>']*distValueUnigram5)
            probSentenceGivenL5+=tempProbabilitySum
            
        #Now multiply this b/v value by its a(x) and add its log linear probability up as necessary per each occurence/count of double backed off bigram
        for key, value in tempUnigramBackoffList6.items():
            if key[0] in redistributedBigram6:
                tempProbabilitySum = value * math.log(redistributedBigram6[key[0]]*distValueUnigram6)
            else:
                tempProbabilitySum = value * math.log(redistributedBigram6['<UNK>']*distValueUnigram6)
            probSentenceGivenL6+=tempProbabilitySum
        
        
        print("Now predicting language\n")
        #predict which language it is using logs
        logProb1 = probSentenceGivenL1 + math.log(probLang1)
        logProb2 = probSentenceGivenL2 + math.log(probLang2)
        logProb3 = probSentenceGivenL3 + math.log(probLang3)
        logProb4 = probSentenceGivenL4 + math.log(probLang4)
        logProb5 = probSentenceGivenL5 + math.log(probLang5)
        logProb6 = probSentenceGivenL6 + math.log(probLang6)
        
        #store probabilities in dictionary with respective languages as keys
        probDict = {
            "Lang1": logProb1, 
            "Lang2": logProb2, 
            "Lang3": logProb3, 
            "Lang4": logProb4, 
            "Lang5": logProb5, 
            "Lang6": logProb6
            }
        
        #find maximum of these log likelihoods and set that as the predicted language
        Keymax = max(probDict, key=probDict.get)
        predictedLang.append(Keymax)
        
        #append the actual language this dev set is from to actual language list
        actualLang.append('Lang1')
        
        countDevLang1+=1
        #print("Done with %d sentences in dev set for Lang 1\n"%(countDevLang1))
        
        #resetting the temporary list of words per each sentence 
        tempSentence1 = []
        tempSentence2 = []
        tempSentence3 = []
        tempSentence4 = []
        tempSentence5 = []
        tempSentence6 = []
        
        
    tempSentence1 = []
    tempSentence2 = []
    tempSentence3 = []
    tempSentence4 = []
    tempSentence5 = []
    tempSentence6 = []
    
    en_dev1.close()
    """
    #connlu parse and update bigram and unigram counts 
    for tokenlist in parse_incr(en_dev2):
        for token in tokenlist:         
            #If sentence is in entire vocab from all languages, add the word to the sentence. Otherwise add the unknown token to the sentence.
            #Regardless, if the bigram doesn't exist in a specific language one will have laplace + 0 bigram count /unigram count+ laplace + V. 
            #If the unigram doesn't exist, one will have laplace + 0 bigram count / laplace + 0 + V. If the bigram exists one instead has bigram count+laplace
            # divided by unigram count + laplace + V
            if not (token["form"].lower()) in wordList :
                tempSentence1.append('<UNK>')
                tempSentence2.append('<UNK>')
                tempSentence3.append('<UNK>')
                tempSentence4.append('<UNK>')
                tempSentence5.append('<UNK>')
                tempSentence6.append('<UNK>')
            else:
                ##adding to temporary sentence which will be parsed into bigrams
                #making it lower case as a means of preprocessing. words of different case but same spelling are the same type for my purposes
                tempSentence1.append(token["form"].lower())
                tempSentence2.append(token["form"].lower())
                tempSentence3.append(token["form"].lower())
                tempSentence4.append(token["form"].lower())
                tempSentence5.append(token["form"].lower())
                tempSentence6.append(token["form"].lower())


        #now adding eos and bos tags to the sentence
        tempSentence1.insert(0, "<BOS>")
        tempSentence1.append("<EOS>")
        tempSentence2.insert(0, "<BOS>")
        tempSentence2.append("<EOS>")
        tempSentence3.insert(0, "<BOS>")
        tempSentence3.append("<EOS>")
        tempSentence4.insert(0, "<BOS>")
        tempSentence4.append("<EOS>")
        tempSentence5.insert(0, "<BOS>")
        tempSentence5.append("<EOS>")
        tempSentence6.insert(0, "<BOS>")
        tempSentence6.append("<EOS>")
        
        #this is math.log(1) since I am adding log probabilities to avoid multiplication
        probSentenceGivenL1 = 0
        probSentenceGivenL2 = 0
        probSentenceGivenL3 = 0
        probSentenceGivenL4 = 0
        probSentenceGivenL5 = 0
        probSentenceGivenL6 = 0
        
        #for zipping to bigram tuples
        tempBigram1 = zip(tempSentence1, islice(tempSentence1, 1, None))
        tempBigram2 = zip(tempSentence2, islice(tempSentence2, 1, None))
        tempBigram3 = zip(tempSentence3, islice(tempSentence3, 1, None))
        tempBigram4 = zip(tempSentence4, islice(tempSentence4, 1, None))
        tempBigram5 = zip(tempSentence5, islice(tempSentence5, 1, None))
        tempBigram6 = zip(tempSentence6, islice(tempSentence6, 1, None))
        
        #for storing values that will later have redistributed unigram probability
        tempUnigramBackoffList1 = {}
        tempUnigramBackoffList2 = {}
        tempUnigramBackoffList3 = {}
        tempUnigramBackoffList4 = {}
        tempUnigramBackoffList5 = {}
        tempUnigramBackoffList6 = {}
        
        for wordPair in tempBigram1 :
            #if c(x,y)>0 just count probability as c(x,y)-d2 / c(x)
            if wordPair in backedOffBigram1:
                normalizingConstant = (series1[wordPair[0]])
                tempProbability = (backedOffBigram1[wordPair])/(normalizingConstant)
                probSentenceGivenL1+=math.log(tempProbability)
            else:
                #otherwise check if c(x)>0 and if it is, count a(x) * (c(x)/ summation of c(v) across all V) as the probability  
                if wordPair[0] in backedOffList1:
                    tempProbability = redistributedBigram1[wordPair[0]] *( backedOffList1[wordPair[0]] / wordCount1)
                    probSentenceGivenL1+=math.log(tempProbability)
                else:
                    #otherwise add the word pair as a key to a list of double backed off values
                    if wordPair in tempUnigramBackoffList1:
                        tempUnigramBackoffList1[wordPair] += 1
                    else:
                        tempUnigramBackoffList1[wordPair] = 1

        for wordPair in tempBigram2 :
            #if c(x,y)>0 just count probability as c(x,y)-d2 / c(x)
            if wordPair in backedOffBigram2:
                normalizingConstant = (series2[wordPair[0]])
                tempProbability = (backedOffBigram2[wordPair])/(normalizingConstant)
                probSentenceGivenL2+=math.log(tempProbability)
            else:
                #otherwise check if c(x)>0 and if it is, count a(x) * (c(x)/ summation of c(v) across all V) as the probability  
                if wordPair[0] in backedOffList2:
                    tempProbability = redistributedBigram2[wordPair[0]] *( backedOffList2[wordPair[0]] / wordCount2)
                    probSentenceGivenL2+=math.log(tempProbability)
                else:
                    #otherwise add the word pair as a key to a list of double backed off values
                    if wordPair in tempUnigramBackoffList2:
                        tempUnigramBackoffList2[wordPair] += 1
                    else:
                        tempUnigramBackoffList2[wordPair] = 1
                
        for wordPair in tempBigram3 :
            #if c(x,y)>0 just count probability as c(x,y)-d2 / c(x)
            if wordPair in backedOffBigram3:
                normalizingConstant = (series3[wordPair[0]])
                tempProbability = (backedOffBigram3[wordPair])/(normalizingConstant)
                probSentenceGivenL3+=math.log(tempProbability)
            else:
                #otherwise check if c(x)>0 and if it is, count a(x) * (c(x)/ summation of c(v) across all V) as the probability  
                if wordPair[0] in backedOffList3:
                    tempProbability = redistributedBigram3[wordPair[0]] *( backedOffList3[wordPair[0]] / wordCount3)
                    probSentenceGivenL3+=math.log(tempProbability)
                else:
                    #otherwise add the word pair as a key to a list of double backed off values
                    if wordPair in tempUnigramBackoffList3:
                        tempUnigramBackoffList3[wordPair] += 1
                    else:
                        tempUnigramBackoffList3[wordPair] = 1
                
        for wordPair in tempBigram4 :
            #if c(x,y)>0 just count probability as c(x,y)-d2 / c(x)
            if wordPair in backedOffBigram4:
                normalizingConstant = (series4[wordPair[0]])
                tempProbability = (backedOffBigram4[wordPair])/(normalizingConstant)
                probSentenceGivenL4+=math.log(tempProbability)
            else:
                #otherwise check if c(x)>0 and if it is, count a(x) * (c(x)/ summation of c(v) across all V) as the probability  
                if wordPair[0] in backedOffList4:
                    tempProbability = redistributedBigram4[wordPair[0]] *( backedOffList4[wordPair[0]] / wordCount4)
                    probSentenceGivenL4+=math.log(tempProbability)
                else:
                    #otherwise add the word pair as a key to a list of double backed off values
                    if wordPair in tempUnigramBackoffList4:
                        tempUnigramBackoffList4[wordPair] += 1
                    else:
                        tempUnigramBackoffList4[wordPair] = 1
                
        for wordPair in tempBigram2 :
            #if c(x,y)>0 just count probability as c(x,y)-d2 / c(x)
            if wordPair in backedOffBigram5:
                normalizingConstant = (series5[wordPair[0]])
                tempProbability = (backedOffBigram5[wordPair])/(normalizingConstant)
                probSentenceGivenL5+=math.log(tempProbability)
            else:
                #otherwise check if c(x)>0 and if it is, count a(x) * (c(x)/ summation of c(v) across all V) as the probability  
                if wordPair[0] in backedOffList5:
                    tempProbability = redistributedBigram5[wordPair[0]] *( backedOffList5[wordPair[0]] / wordCount5)
                    probSentenceGivenL5+=math.log(tempProbability)
                else:
                    #otherwise add the word pair as a key to a list of double backed off values
                    if wordPair in tempUnigramBackoffList5:
                        tempUnigramBackoffList5[wordPair] += 1
                    else:
                        tempUnigramBackoffList5[wordPair] = 1
                
        for wordPair in tempBigram6 :
            #if c(x,y)>0 just count probability as c(x,y)-d2 / c(x)
            if wordPair in backedOffBigram6:
                normalizingConstant = (series6[wordPair[0]])
                tempProbability = (backedOffBigram6[wordPair])/(normalizingConstant)
                probSentenceGivenL6+=math.log(tempProbability)
            else:
                #otherwise check if c(x)>0 and if it is, count a(x) * (c(x)/ summation of c(v) across all V) as the probability  
                if wordPair[0] in backedOffList6:
                    tempProbability = redistributedBigram6[wordPair[0]] *( backedOffList6[wordPair[0]] / wordCount6)
                    probSentenceGivenL6+=math.log(tempProbability)
                else:
                    #otherwise add the word pair as a key to a list of double backed off values
                    if wordPair in tempUnigramBackoffList6:
                        tempUnigramBackoffList6[wordPair] += 1
                    else:
                        tempUnigramBackoffList6[wordPair] = 1
        
        #redistribute the probabilities that had to be backed off twice.
        print("Now redistributing unigram backoff as necessary.\n")
        
        #First need to count number of oov types
        numOOVTypes1 = len(tempUnigramBackoffList1)
        numOOVTypes2 = len(tempUnigramBackoffList2)
        numOOVTypes3 = len(tempUnigramBackoffList3)
        numOOVTypes4 = len(tempUnigramBackoffList4)
        numOOVTypes5 = len(tempUnigramBackoffList5)
        numOOVTypes6 = len(tempUnigramBackoffList6)
        
        #then need to get b/v values where V is the size of the total vocab of known and previously unknown newly encountered words
        #b is the probability mass set aside for redistribution
        distValueUnigram1 = redistributedUnigram1/(numOOVTypes1+sizeOfVocab1)
        distValueUnigram2 = redistributedUnigram2/(numOOVTypes2+sizeOfVocab2)
        distValueUnigram3 = redistributedUnigram3/(numOOVTypes3+sizeOfVocab3)
        distValueUnigram4 = redistributedUnigram4/(numOOVTypes4+sizeOfVocab4)
        distValueUnigram5 = redistributedUnigram5/(numOOVTypes5+sizeOfVocab5)
        distValueUnigram6 = redistributedUnigram6/(numOOVTypes6+sizeOfVocab6)

        #Now multiply this b/v value by its a(x) and add its log linear probability up as necessary per each occurence/count of double backed off bigram
        for key, value in tempUnigramBackoffList1.items():
            if key[0] in redistributedBigram1:
                tempProbabilitySum = value * math.log(redistributedBigram1[key[0]]*distValueUnigram1)
            else:
                tempProbabilitySum = value * math.log(redistributedBigram1['<UNK>']*distValueUnigram1)
            probSentenceGivenL1+=tempProbabilitySum
            
        #Now multiply this b/v value by its a(x) and add its log linear probability up as necessary per each occurence/count of double backed off bigram
        for key, value in tempUnigramBackoffList2.items():
            if key[0] in redistributedBigram2:
                tempProbabilitySum = value * math.log(redistributedBigram2[key[0]]*distValueUnigram2)
            else:
                tempProbabilitySum = value * math.log(redistributedBigram2['<UNK>']*distValueUnigram2)
            probSentenceGivenL2+=tempProbabilitySum
            
        #Now multiply this b/v value by its a(x) and add its log linear probability up as necessary per each occurence/count of double backed off bigram
        for key, value in tempUnigramBackoffList3.items():
            if key[0] in redistributedBigram3:
                tempProbabilitySum = value * math.log(redistributedBigram3[key[0]]*distValueUnigram3)
            else:
                tempProbabilitySum = value * math.log(redistributedBigram3['<UNK>']*distValueUnigram3)
            probSentenceGivenL1+=tempProbabilitySum
            
            
        #Now multiply this b/v value by its a(x) and add its log linear probability up as necessary per each occurence/count of double backed off bigram
        for key, value in tempUnigramBackoffList4.items():
            if key[0] in redistributedBigram4:
                tempProbabilitySum = value * math.log(redistributedBigram4[key[0]]*distValueUnigram4)
            else:
                tempProbabilitySum = value * math.log(redistributedBigram4['<UNK>']*distValueUnigram4)
            probSentenceGivenL4+=tempProbabilitySum
            
        #Now multiply this b/v value by its a(x) and add its log linear probability up as necessary per each occurence/count of double backed off bigram
        for key, value in tempUnigramBackoffList5.items():
            if key[0] in redistributedBigram5:
                tempProbabilitySum = value * math.log(redistributedBigram5[key[0]]*distValueUnigram5)
            else:
                tempProbabilitySum = value * math.log(redistributedBigram5['<UNK>']*distValueUnigram5)
            probSentenceGivenL5+=tempProbabilitySum
            
        #Now multiply this b/v value by its a(x) and add its log linear probability up as necessary per each occurence/count of double backed off bigram
        for key, value in tempUnigramBackoffList6.items():
            if key[0] in redistributedBigram6:
                tempProbabilitySum = value * math.log(redistributedBigram6[key[0]]*distValueUnigram6)
            else:
                tempProbabilitySum = value * math.log(redistributedBigram6['<UNK>']*distValueUnigram6)
            probSentenceGivenL6+=tempProbabilitySum
        
        
        print("Now predicting language")
        #predict which language it is using logs
        logProb1 = probSentenceGivenL1 + math.log(probLang1)
        logProb2 = probSentenceGivenL2 + math.log(probLang2)
        logProb3 = probSentenceGivenL3 + math.log(probLang3)
        logProb4 = probSentenceGivenL4 + math.log(probLang4)
        logProb5 = probSentenceGivenL5 + math.log(probLang5)
        logProb6 = probSentenceGivenL6 + math.log(probLang6)
        
        #store probabilities in dictionary with respective languages as keys
        probDict = {
            "Lang1": logProb1, 
            "Lang2": logProb2, 
            "Lang3": logProb3, 
            "Lang4": logProb4, 
            "Lang5": logProb5, 
            "Lang6": logProb6
            }
        
        #find maximum of these log likelihoods and set that as the predicted language
        Keymax = max(probDict, key=probDict.get)
        predictedLang.append(Keymax)
        
        #append the actual language this dev set is from to actual language list
        actualLang.append('Lang2')
        
        countDevLang2+=1
        print("Done with %d sentences in dev set for Lang 2"%(countDevLang2))
        
        #resetting the temporary list of words per each sentence 
        tempSentence1 = []
        tempSentence2 = []
        tempSentence3 = []
        tempSentence4 = []
        tempSentence5 = []
        tempSentence6 = []
        
        
    tempSentence1 = []
    tempSentence2 = []
    tempSentence3 = []
    tempSentence4 = []
    tempSentence5 = []
    tempSentence6 = []
    
    en_dev2.close()
    
    #connlu parse and update bigram and unigram counts 
    for tokenlist in parse_incr(en_dev3):
        for token in tokenlist:         
            #If sentence is in entire vocab from all languages, add the word to the sentence. Otherwise add the unknown token to the sentence.
            #Regardless, if the bigram doesn't exist in a specific language one will have laplace + 0 bigram count /unigram count+ laplace + V. 
            #If the unigram doesn't exist, one will have laplace + 0 bigram count / laplace + 0 + V. If the bigram exists one instead has bigram count+laplace
            # divided by unigram count + laplace + V
            if not (token["form"].lower()) in wordList :
                tempSentence1.append('<UNK>')
                tempSentence2.append('<UNK>')
                tempSentence3.append('<UNK>')
                tempSentence4.append('<UNK>')
                tempSentence5.append('<UNK>')
                tempSentence6.append('<UNK>')
            else:
                ##adding to temporary sentence which will be parsed into bigrams
                #making it lower case as a means of preprocessing. words of different case but same spelling are the same type for my purposes
                tempSentence1.append(token["form"].lower())
                tempSentence2.append(token["form"].lower())
                tempSentence3.append(token["form"].lower())
                tempSentence4.append(token["form"].lower())
                tempSentence5.append(token["form"].lower())
                tempSentence6.append(token["form"].lower())


        #now adding eos and bos tags to the sentence
        tempSentence1.insert(0, "<BOS>")
        tempSentence1.append("<EOS>")
        tempSentence2.insert(0, "<BOS>")
        tempSentence2.append("<EOS>")
        tempSentence3.insert(0, "<BOS>")
        tempSentence3.append("<EOS>")
        tempSentence4.insert(0, "<BOS>")
        tempSentence4.append("<EOS>")
        tempSentence5.insert(0, "<BOS>")
        tempSentence5.append("<EOS>")
        tempSentence6.insert(0, "<BOS>")
        tempSentence6.append("<EOS>")
        
        #this is math.log(1) since I am adding log probabilities to avoid multiplication
        probSentenceGivenL1 = 0
        probSentenceGivenL2 = 0
        probSentenceGivenL3 = 0
        probSentenceGivenL4 = 0
        probSentenceGivenL5 = 0
        probSentenceGivenL6 = 0
        
        #for zipping to bigram tuples
        tempBigram1 = zip(tempSentence1, islice(tempSentence1, 1, None))
        tempBigram2 = zip(tempSentence2, islice(tempSentence2, 1, None))
        tempBigram3 = zip(tempSentence3, islice(tempSentence3, 1, None))
        tempBigram4 = zip(tempSentence4, islice(tempSentence4, 1, None))
        tempBigram5 = zip(tempSentence5, islice(tempSentence5, 1, None))
        tempBigram6 = zip(tempSentence6, islice(tempSentence6, 1, None))
        
        #for storing values that will later have redistributed unigram probability
        tempUnigramBackoffList1 = {}
        tempUnigramBackoffList2 = {}
        tempUnigramBackoffList3 = {}
        tempUnigramBackoffList4 = {}
        tempUnigramBackoffList5 = {}
        tempUnigramBackoffList6 = {}
        
        for wordPair in tempBigram1 :
            #if c(x,y)>0 just count probability as c(x,y)-d2 / c(x)
            if wordPair in backedOffBigram1:
                normalizingConstant = (series1[wordPair[0]])
                tempProbability = (backedOffBigram1[wordPair])/(normalizingConstant)
                probSentenceGivenL1+=math.log(tempProbability)
            else:
                #otherwise check if c(x)>0 and if it is, count a(x) * (c(x)/ summation of c(v) across all V) as the probability  
                if wordPair[0] in backedOffList1:
                    tempProbability = redistributedBigram1[wordPair[0]] *( backedOffList1[wordPair[0]] / wordCount1)
                    probSentenceGivenL1+=math.log(tempProbability)
                else:
                    #otherwise add the word pair as a key to a list of double backed off values
                    if wordPair in tempUnigramBackoffList1:
                        tempUnigramBackoffList1[wordPair] += 1
                    else:
                        tempUnigramBackoffList1[wordPair] = 1

        for wordPair in tempBigram2 :
            #if c(x,y)>0 just count probability as c(x,y)-d2 / c(x)
            if wordPair in backedOffBigram2:
                normalizingConstant = (series2[wordPair[0]])
                tempProbability = (backedOffBigram2[wordPair])/(normalizingConstant)
                probSentenceGivenL2+=math.log(tempProbability)
            else:
                #otherwise check if c(x)>0 and if it is, count a(x) * (c(x)/ summation of c(v) across all V) as the probability  
                if wordPair[0] in backedOffList2:
                    tempProbability = redistributedBigram2[wordPair[0]] *( backedOffList2[wordPair[0]] / wordCount2)
                    probSentenceGivenL2+=math.log(tempProbability)
                else:
                    #otherwise add the word pair as a key to a list of double backed off values
                    if wordPair in tempUnigramBackoffList2:
                        tempUnigramBackoffList2[wordPair] += 1
                    else:
                        tempUnigramBackoffList2[wordPair] = 1
                
        for wordPair in tempBigram3 :
            #if c(x,y)>0 just count probability as c(x,y)-d2 / c(x)
            if wordPair in backedOffBigram3:
                normalizingConstant = (series3[wordPair[0]])
                tempProbability = (backedOffBigram3[wordPair])/(normalizingConstant)
                probSentenceGivenL3+=math.log(tempProbability)
            else:
                #otherwise check if c(x)>0 and if it is, count a(x) * (c(x)/ summation of c(v) across all V) as the probability  
                if wordPair[0] in backedOffList3:
                    tempProbability = redistributedBigram3[wordPair[0]] *( backedOffList3[wordPair[0]] / wordCount3)
                    probSentenceGivenL3+=math.log(tempProbability)
                else:
                    #otherwise add the word pair as a key to a list of double backed off values
                    if wordPair in tempUnigramBackoffList3:
                        tempUnigramBackoffList3[wordPair] += 1
                    else:
                        tempUnigramBackoffList3[wordPair] = 1
                
        for wordPair in tempBigram4 :
            #if c(x,y)>0 just count probability as c(x,y)-d2 / c(x)
            if wordPair in backedOffBigram4:
                normalizingConstant = (series4[wordPair[0]])
                tempProbability = (backedOffBigram4[wordPair])/(normalizingConstant)
                probSentenceGivenL4+=math.log(tempProbability)
            else:
                #otherwise check if c(x)>0 and if it is, count a(x) * (c(x)/ summation of c(v) across all V) as the probability  
                if wordPair[0] in backedOffList4:
                    tempProbability = redistributedBigram4[wordPair[0]] *( backedOffList4[wordPair[0]] / wordCount4)
                    probSentenceGivenL4+=math.log(tempProbability)
                else:
                    #otherwise add the word pair as a key to a list of double backed off values
                    if wordPair in tempUnigramBackoffList4:
                        tempUnigramBackoffList4[wordPair] += 1
                    else:
                        tempUnigramBackoffList4[wordPair] = 1
                
        for wordPair in tempBigram2 :
            #if c(x,y)>0 just count probability as c(x,y)-d2 / c(x)
            if wordPair in backedOffBigram5:
                normalizingConstant = (series5[wordPair[0]])
                tempProbability = (backedOffBigram5[wordPair])/(normalizingConstant)
                probSentenceGivenL5+=math.log(tempProbability)
            else:
                #otherwise check if c(x)>0 and if it is, count a(x) * (c(x)/ summation of c(v) across all V) as the probability  
                if wordPair[0] in backedOffList5:
                    tempProbability = redistributedBigram5[wordPair[0]] *( backedOffList5[wordPair[0]] / wordCount5)
                    probSentenceGivenL5+=math.log(tempProbability)
                else:
                    #otherwise add the word pair as a key to a list of double backed off values
                    if wordPair in tempUnigramBackoffList5:
                        tempUnigramBackoffList5[wordPair] += 1
                    else:
                        tempUnigramBackoffList5[wordPair] = 1
                
        for wordPair in tempBigram6 :
            #if c(x,y)>0 just count probability as c(x,y)-d2 / c(x)
            if wordPair in backedOffBigram6:
                normalizingConstant = (series6[wordPair[0]])
                tempProbability = (backedOffBigram6[wordPair])/(normalizingConstant)
                probSentenceGivenL6+=math.log(tempProbability)
            else:
                #otherwise check if c(x)>0 and if it is, count a(x) * (c(x)/ summation of c(v) across all V) as the probability  
                if wordPair[0] in backedOffList6:
                    tempProbability = redistributedBigram6[wordPair[0]] *( backedOffList6[wordPair[0]] / wordCount6)
                    probSentenceGivenL6+=math.log(tempProbability)
                else:
                    #otherwise add the word pair as a key to a list of double backed off values
                    if wordPair in tempUnigramBackoffList6:
                        tempUnigramBackoffList6[wordPair] += 1
                    else:
                        tempUnigramBackoffList6[wordPair] = 1
        
        #redistribute the probabilities that had to be backed off twice.
        print("Now redistributing unigram backoff as necessary.\n")
        
        #First need to count number of oov types
        numOOVTypes1 = len(tempUnigramBackoffList1)
        numOOVTypes2 = len(tempUnigramBackoffList2)
        numOOVTypes3 = len(tempUnigramBackoffList3)
        numOOVTypes4 = len(tempUnigramBackoffList4)
        numOOVTypes5 = len(tempUnigramBackoffList5)
        numOOVTypes6 = len(tempUnigramBackoffList6)
        
        #then need to get b/v values where V is the size of the total vocab of known and previously unknown newly encountered words
        #b is the probability mass set aside for redistribution
        distValueUnigram1 = redistributedUnigram1/(numOOVTypes1+sizeOfVocab1)
        distValueUnigram2 = redistributedUnigram2/(numOOVTypes2+sizeOfVocab2)
        distValueUnigram3 = redistributedUnigram3/(numOOVTypes3+sizeOfVocab3)
        distValueUnigram4 = redistributedUnigram4/(numOOVTypes4+sizeOfVocab4)
        distValueUnigram5 = redistributedUnigram5/(numOOVTypes5+sizeOfVocab5)
        distValueUnigram6 = redistributedUnigram6/(numOOVTypes6+sizeOfVocab6)

        #Now multiply this b/v value by its a(x) and add its log linear probability up as necessary per each occurence/count of double backed off bigram
        for key, value in tempUnigramBackoffList1.items():
            if key[0] in redistributedBigram1:
                tempProbabilitySum = value * math.log(redistributedBigram1[key[0]]*distValueUnigram1)
            else:
                tempProbabilitySum = value * math.log(redistributedBigram1['<UNK>']*distValueUnigram1)
            probSentenceGivenL1+=tempProbabilitySum
            
        #Now multiply this b/v value by its a(x) and add its log linear probability up as necessary per each occurence/count of double backed off bigram
        for key, value in tempUnigramBackoffList2.items():
            if key[0] in redistributedBigram2:
                tempProbabilitySum = value * math.log(redistributedBigram2[key[0]]*distValueUnigram2)
            else:
                tempProbabilitySum = value * math.log(redistributedBigram2['<UNK>']*distValueUnigram2)
            probSentenceGivenL2+=tempProbabilitySum
            
        #Now multiply this b/v value by its a(x) and add its log linear probability up as necessary per each occurence/count of double backed off bigram
        for key, value in tempUnigramBackoffList3.items():
            if key[0] in redistributedBigram3:
                tempProbabilitySum = value * math.log(redistributedBigram3[key[0]]*distValueUnigram3)
            else:
                tempProbabilitySum = value * math.log(redistributedBigram3['<UNK>']*distValueUnigram3)
            probSentenceGivenL1+=tempProbabilitySum
            
            
        #Now multiply this b/v value by its a(x) and add its log linear probability up as necessary per each occurence/count of double backed off bigram
        for key, value in tempUnigramBackoffList4.items():
            if key[0] in redistributedBigram4:
                tempProbabilitySum = value * math.log(redistributedBigram4[key[0]]*distValueUnigram4)
            else:
                tempProbabilitySum = value * math.log(redistributedBigram4['<UNK>']*distValueUnigram4)
            probSentenceGivenL4+=tempProbabilitySum
            
        #Now multiply this b/v value by its a(x) and add its log linear probability up as necessary per each occurence/count of double backed off bigram
        for key, value in tempUnigramBackoffList5.items():
            if key[0] in redistributedBigram5:
                tempProbabilitySum = value * math.log(redistributedBigram5[key[0]]*distValueUnigram5)
            else:
                tempProbabilitySum = value * math.log(redistributedBigram5['<UNK>']*distValueUnigram5)
            probSentenceGivenL5+=tempProbabilitySum
            
        #Now multiply this b/v value by its a(x) and add its log linear probability up as necessary per each occurence/count of double backed off bigram
        for key, value in tempUnigramBackoffList6.items():
            if key[0] in redistributedBigram6:
                tempProbabilitySum = value * math.log(redistributedBigram6[key[0]]*distValueUnigram6)
            else:
                tempProbabilitySum = value * math.log(redistributedBigram6['<UNK>']*distValueUnigram6)
            probSentenceGivenL6+=tempProbabilitySum
        
        
        print("Now predicting language")
        #predict which language it is using logs
        logProb1 = probSentenceGivenL1 + math.log(probLang1)
        logProb2 = probSentenceGivenL2 + math.log(probLang2)
        logProb3 = probSentenceGivenL3 + math.log(probLang3)
        logProb4 = probSentenceGivenL4 + math.log(probLang4)
        logProb5 = probSentenceGivenL5 + math.log(probLang5)
        logProb6 = probSentenceGivenL6 + math.log(probLang6)
        
        #store probabilities in dictionary with respective languages as keys
        probDict = {
            "Lang1": logProb1, 
            "Lang2": logProb2, 
            "Lang3": logProb3, 
            "Lang4": logProb4, 
            "Lang5": logProb5, 
            "Lang6": logProb6
            }
        
        #find maximum of these log likelihoods and set that as the predicted language
        Keymax = max(probDict, key=probDict.get)
        predictedLang.append(Keymax)
        
        #append the actual language this dev set is from to actual language list
        actualLang.append('Lang3')
        
        countDevLang3+=1
        print("Done with %d sentences in dev set for Lang 3"%(countDevLang3))
        
        #resetting the temporary list of words per each sentence 
        tempSentence1 = []
        tempSentence2 = []
        tempSentence3 = []
        tempSentence4 = []
        tempSentence5 = []
        tempSentence6 = []
        
        
    tempSentence1 = []
    tempSentence2 = []
    tempSentence3 = []
    tempSentence4 = []
    tempSentence5 = []
    tempSentence6 = []
    
    en_dev3.close()
    
    #connlu parse and update bigram and unigram counts 
    for tokenlist in parse_incr(en_dev4):
        for token in tokenlist:         
            #If sentence is in entire vocab from all languages, add the word to the sentence. Otherwise add the unknown token to the sentence.
            #Regardless, if the bigram doesn't exist in a specific language one will have laplace + 0 bigram count /unigram count+ laplace + V. 
            #If the unigram doesn't exist, one will have laplace + 0 bigram count / laplace + 0 + V. If the bigram exists one instead has bigram count+laplace
            # divided by unigram count + laplace + V
            if not (token["form"].lower()) in wordList :
                tempSentence1.append('<UNK>')
                tempSentence2.append('<UNK>')
                tempSentence3.append('<UNK>')
                tempSentence4.append('<UNK>')
                tempSentence5.append('<UNK>')
                tempSentence6.append('<UNK>')
            else:
                ##adding to temporary sentence which will be parsed into bigrams
                #making it lower case as a means of preprocessing. words of different case but same spelling are the same type for my purposes
                tempSentence1.append(token["form"].lower())
                tempSentence2.append(token["form"].lower())
                tempSentence3.append(token["form"].lower())
                tempSentence4.append(token["form"].lower())
                tempSentence5.append(token["form"].lower())
                tempSentence6.append(token["form"].lower())


        #now adding eos and bos tags to the sentence
        tempSentence1.insert(0, "<BOS>")
        tempSentence1.append("<EOS>")
        tempSentence2.insert(0, "<BOS>")
        tempSentence2.append("<EOS>")
        tempSentence3.insert(0, "<BOS>")
        tempSentence3.append("<EOS>")
        tempSentence4.insert(0, "<BOS>")
        tempSentence4.append("<EOS>")
        tempSentence5.insert(0, "<BOS>")
        tempSentence5.append("<EOS>")
        tempSentence6.insert(0, "<BOS>")
        tempSentence6.append("<EOS>")
        
        #this is math.log(1) since I am adding log probabilities to avoid multiplication
        probSentenceGivenL1 = 0
        probSentenceGivenL2 = 0
        probSentenceGivenL3 = 0
        probSentenceGivenL4 = 0
        probSentenceGivenL5 = 0
        probSentenceGivenL6 = 0
        
        #for zipping to bigram tuples
        tempBigram1 = zip(tempSentence1, islice(tempSentence1, 1, None))
        tempBigram2 = zip(tempSentence2, islice(tempSentence2, 1, None))
        tempBigram3 = zip(tempSentence3, islice(tempSentence3, 1, None))
        tempBigram4 = zip(tempSentence4, islice(tempSentence4, 1, None))
        tempBigram5 = zip(tempSentence5, islice(tempSentence5, 1, None))
        tempBigram6 = zip(tempSentence6, islice(tempSentence6, 1, None))
        
        #for storing values that will later have redistributed unigram probability
        tempUnigramBackoffList1 = {}
        tempUnigramBackoffList2 = {}
        tempUnigramBackoffList3 = {}
        tempUnigramBackoffList4 = {}
        tempUnigramBackoffList5 = {}
        tempUnigramBackoffList6 = {}
        
        for wordPair in tempBigram1 :
            #if c(x,y)>0 just count probability as c(x,y)-d2 / c(x)
            if wordPair in backedOffBigram1:
                normalizingConstant = (series1[wordPair[0]])
                tempProbability = (backedOffBigram1[wordPair])/(normalizingConstant)
                probSentenceGivenL1+=math.log(tempProbability)
            else:
                #otherwise check if c(x)>0 and if it is, count a(x) * (c(x)/ summation of c(v) across all V) as the probability  
                if wordPair[0] in backedOffList1:
                    tempProbability = redistributedBigram1[wordPair[0]] *( backedOffList1[wordPair[0]] / wordCount1)
                    probSentenceGivenL1+=math.log(tempProbability)
                else:
                    #otherwise add the word pair as a key to a list of double backed off values
                    if wordPair in tempUnigramBackoffList1:
                        tempUnigramBackoffList1[wordPair] += 1
                    else:
                        tempUnigramBackoffList1[wordPair] = 1

        for wordPair in tempBigram2 :
            #if c(x,y)>0 just count probability as c(x,y)-d2 / c(x)
            if wordPair in backedOffBigram2:
                normalizingConstant = (series2[wordPair[0]])
                tempProbability = (backedOffBigram2[wordPair])/(normalizingConstant)
                probSentenceGivenL2+=math.log(tempProbability)
            else:
                #otherwise check if c(x)>0 and if it is, count a(x) * (c(x)/ summation of c(v) across all V) as the probability  
                if wordPair[0] in backedOffList2:
                    tempProbability = redistributedBigram2[wordPair[0]] *( backedOffList2[wordPair[0]] / wordCount2)
                    probSentenceGivenL2+=math.log(tempProbability)
                else:
                    #otherwise add the word pair as a key to a list of double backed off values
                    if wordPair in tempUnigramBackoffList2:
                        tempUnigramBackoffList2[wordPair] += 1
                    else:
                        tempUnigramBackoffList2[wordPair] = 1
                
        for wordPair in tempBigram3 :
            #if c(x,y)>0 just count probability as c(x,y)-d2 / c(x)
            if wordPair in backedOffBigram3:
                normalizingConstant = (series3[wordPair[0]])
                tempProbability = (backedOffBigram3[wordPair])/(normalizingConstant)
                probSentenceGivenL3+=math.log(tempProbability)
            else:
                #otherwise check if c(x)>0 and if it is, count a(x) * (c(x)/ summation of c(v) across all V) as the probability  
                if wordPair[0] in backedOffList3:
                    tempProbability = redistributedBigram3[wordPair[0]] *( backedOffList3[wordPair[0]] / wordCount3)
                    probSentenceGivenL3+=math.log(tempProbability)
                else:
                    #otherwise add the word pair as a key to a list of double backed off values
                    if wordPair in tempUnigramBackoffList3:
                        tempUnigramBackoffList3[wordPair] += 1
                    else:
                        tempUnigramBackoffList3[wordPair] = 1
                
        for wordPair in tempBigram4 :
            #if c(x,y)>0 just count probability as c(x,y)-d2 / c(x)
            if wordPair in backedOffBigram4:
                normalizingConstant = (series4[wordPair[0]])
                tempProbability = (backedOffBigram4[wordPair])/(normalizingConstant)
                probSentenceGivenL4+=math.log(tempProbability)
            else:
                #otherwise check if c(x)>0 and if it is, count a(x) * (c(x)/ summation of c(v) across all V) as the probability  
                if wordPair[0] in backedOffList4:
                    tempProbability = redistributedBigram4[wordPair[0]] *( backedOffList4[wordPair[0]] / wordCount4)
                    probSentenceGivenL4+=math.log(tempProbability)
                else:
                    #otherwise add the word pair as a key to a list of double backed off values
                    if wordPair in tempUnigramBackoffList4:
                        tempUnigramBackoffList4[wordPair] += 1
                    else:
                        tempUnigramBackoffList4[wordPair] = 1
                
        for wordPair in tempBigram2 :
            #if c(x,y)>0 just count probability as c(x,y)-d2 / c(x)
            if wordPair in backedOffBigram5:
                normalizingConstant = (series5[wordPair[0]])
                tempProbability = (backedOffBigram5[wordPair])/(normalizingConstant)
                probSentenceGivenL5+=math.log(tempProbability)
            else:
                #otherwise check if c(x)>0 and if it is, count a(x) * (c(x)/ summation of c(v) across all V) as the probability  
                if wordPair[0] in backedOffList5:
                    tempProbability = redistributedBigram5[wordPair[0]] *( backedOffList5[wordPair[0]] / wordCount5)
                    probSentenceGivenL5+=math.log(tempProbability)
                else:
                    #otherwise add the word pair as a key to a list of double backed off values
                    if wordPair in tempUnigramBackoffList5:
                        tempUnigramBackoffList5[wordPair] += 1
                    else:
                        tempUnigramBackoffList5[wordPair] = 1
                
        for wordPair in tempBigram6 :
            #if c(x,y)>0 just count probability as c(x,y)-d2 / c(x)
            if wordPair in backedOffBigram6:
                normalizingConstant = (series6[wordPair[0]])
                tempProbability = (backedOffBigram6[wordPair])/(normalizingConstant)
                probSentenceGivenL6+=math.log(tempProbability)
            else:
                #otherwise check if c(x)>0 and if it is, count a(x) * (c(x)/ summation of c(v) across all V) as the probability  
                if wordPair[0] in backedOffList6:
                    tempProbability = redistributedBigram6[wordPair[0]] *( backedOffList6[wordPair[0]] / wordCount6)
                    probSentenceGivenL6+=math.log(tempProbability)
                else:
                    #otherwise add the word pair as a key to a list of double backed off values
                    if wordPair in tempUnigramBackoffList6:
                        tempUnigramBackoffList6[wordPair] += 1
                    else:
                        tempUnigramBackoffList6[wordPair] = 1
        
        #redistribute the probabilities that had to be backed off twice.
        print("Now redistributing unigram backoff as necessary.\n")
        
        #First need to count number of oov types
        numOOVTypes1 = len(tempUnigramBackoffList1)
        numOOVTypes2 = len(tempUnigramBackoffList2)
        numOOVTypes3 = len(tempUnigramBackoffList3)
        numOOVTypes4 = len(tempUnigramBackoffList4)
        numOOVTypes5 = len(tempUnigramBackoffList5)
        numOOVTypes6 = len(tempUnigramBackoffList6)
        
        #then need to get b/v values where V is the size of the total vocab of known and previously unknown newly encountered words
        #b is the probability mass set aside for redistribution
        distValueUnigram1 = redistributedUnigram1/(numOOVTypes1+sizeOfVocab1)
        distValueUnigram2 = redistributedUnigram2/(numOOVTypes2+sizeOfVocab2)
        distValueUnigram3 = redistributedUnigram3/(numOOVTypes3+sizeOfVocab3)
        distValueUnigram4 = redistributedUnigram4/(numOOVTypes4+sizeOfVocab4)
        distValueUnigram5 = redistributedUnigram5/(numOOVTypes5+sizeOfVocab5)
        distValueUnigram6 = redistributedUnigram6/(numOOVTypes6+sizeOfVocab6)

        #Now multiply this b/v value by its a(x) and add its log linear probability up as necessary per each occurence/count of double backed off bigram
        for key, value in tempUnigramBackoffList1.items():
            if key[0] in redistributedBigram1:
                tempProbabilitySum = value * math.log(redistributedBigram1[key[0]]*distValueUnigram1)
            else:
                tempProbabilitySum = value * math.log(redistributedBigram1['<UNK>']*distValueUnigram1)
            probSentenceGivenL1+=tempProbabilitySum
            
        #Now multiply this b/v value by its a(x) and add its log linear probability up as necessary per each occurence/count of double backed off bigram
        for key, value in tempUnigramBackoffList2.items():
            if key[0] in redistributedBigram2:
                tempProbabilitySum = value * math.log(redistributedBigram2[key[0]]*distValueUnigram2)
            else:
                tempProbabilitySum = value * math.log(redistributedBigram2['<UNK>']*distValueUnigram2)
            probSentenceGivenL2+=tempProbabilitySum
            
        #Now multiply this b/v value by its a(x) and add its log linear probability up as necessary per each occurence/count of double backed off bigram
        for key, value in tempUnigramBackoffList3.items():
            if key[0] in redistributedBigram3:
                tempProbabilitySum = value * math.log(redistributedBigram3[key[0]]*distValueUnigram3)
            else:
                tempProbabilitySum = value * math.log(redistributedBigram3['<UNK>']*distValueUnigram3)
            probSentenceGivenL1+=tempProbabilitySum
            
            
        #Now multiply this b/v value by its a(x) and add its log linear probability up as necessary per each occurence/count of double backed off bigram
        for key, value in tempUnigramBackoffList4.items():
            if key[0] in redistributedBigram4:
                tempProbabilitySum = value * math.log(redistributedBigram4[key[0]]*distValueUnigram4)
            else:
                tempProbabilitySum = value * math.log(redistributedBigram4['<UNK>']*distValueUnigram4)
            probSentenceGivenL4+=tempProbabilitySum
            
        #Now multiply this b/v value by its a(x) and add its log linear probability up as necessary per each occurence/count of double backed off bigram
        for key, value in tempUnigramBackoffList5.items():
            if key[0] in redistributedBigram5:
                tempProbabilitySum = value * math.log(redistributedBigram5[key[0]]*distValueUnigram5)
            else:
                tempProbabilitySum = value * math.log(redistributedBigram5['<UNK>']*distValueUnigram5)
            probSentenceGivenL5+=tempProbabilitySum
            
        #Now multiply this b/v value by its a(x) and add its log linear probability up as necessary per each occurence/count of double backed off bigram
        for key, value in tempUnigramBackoffList6.items():
            if key[0] in redistributedBigram6:
                tempProbabilitySum = value * math.log(redistributedBigram6[key[0]]*distValueUnigram6)
            else:
                tempProbabilitySum = value * math.log(redistributedBigram6['<UNK>']*distValueUnigram6)
            probSentenceGivenL6+=tempProbabilitySum
        
        
        print("Now predicting language")
        #predict which language it is using logs
        logProb1 = probSentenceGivenL1 + math.log(probLang1)
        logProb2 = probSentenceGivenL2 + math.log(probLang2)
        logProb3 = probSentenceGivenL3 + math.log(probLang3)
        logProb4 = probSentenceGivenL4 + math.log(probLang4)
        logProb5 = probSentenceGivenL5 + math.log(probLang5)
        logProb6 = probSentenceGivenL6 + math.log(probLang6)
        
        #store probabilities in dictionary with respective languages as keys
        probDict = {
            "Lang1": logProb1, 
            "Lang2": logProb2, 
            "Lang3": logProb3, 
            "Lang4": logProb4, 
            "Lang5": logProb5, 
            "Lang6": logProb6
            }
        
        #find maximum of these log likelihoods and set that as the predicted language
        Keymax = max(probDict, key=probDict.get)
        predictedLang.append(Keymax)
        
        #append the actual language this dev set is from to actual language list
        actualLang.append('Lang4')
        
        countDevLang4+=1
        print("Done with %d sentences in dev set for Lang 4"%(countDevLang4))
        
        #resetting the temporary list of words per each sentence 
        tempSentence1 = []
        tempSentence2 = []
        tempSentence3 = []
        tempSentence4 = []
        tempSentence5 = []
        tempSentence6 = []
        
        
    tempSentence1 = []
    tempSentence2 = []
    tempSentence3 = []
    tempSentence4 = []
    tempSentence5 = []
    tempSentence6 = []
    
    en_dev4.close()
    
    #connlu parse and update bigram and unigram counts 
    for tokenlist in parse_incr(en_dev5):
        for token in tokenlist:         
            #If sentence is in entire vocab from all languages, add the word to the sentence. Otherwise add the unknown token to the sentence.
            #Regardless, if the bigram doesn't exist in a specific language one will have laplace + 0 bigram count /unigram count+ laplace + V. 
            #If the unigram doesn't exist, one will have laplace + 0 bigram count / laplace + 0 + V. If the bigram exists one instead has bigram count+laplace
            # divided by unigram count + laplace + V
            if not (token["form"].lower()) in wordList :
                tempSentence1.append('<UNK>')
                tempSentence2.append('<UNK>')
                tempSentence3.append('<UNK>')
                tempSentence4.append('<UNK>')
                tempSentence5.append('<UNK>')
                tempSentence6.append('<UNK>')
            else:
                ##adding to temporary sentence which will be parsed into bigrams
                #making it lower case as a means of preprocessing. words of different case but same spelling are the same type for my purposes
                tempSentence1.append(token["form"].lower())
                tempSentence2.append(token["form"].lower())
                tempSentence3.append(token["form"].lower())
                tempSentence4.append(token["form"].lower())
                tempSentence5.append(token["form"].lower())
                tempSentence6.append(token["form"].lower())


        #now adding eos and bos tags to the sentence
        tempSentence1.insert(0, "<BOS>")
        tempSentence1.append("<EOS>")
        tempSentence2.insert(0, "<BOS>")
        tempSentence2.append("<EOS>")
        tempSentence3.insert(0, "<BOS>")
        tempSentence3.append("<EOS>")
        tempSentence4.insert(0, "<BOS>")
        tempSentence4.append("<EOS>")
        tempSentence5.insert(0, "<BOS>")
        tempSentence5.append("<EOS>")
        tempSentence6.insert(0, "<BOS>")
        tempSentence6.append("<EOS>")
        
        #this is math.log(1) since I am adding log probabilities to avoid multiplication
        probSentenceGivenL1 = 0
        probSentenceGivenL2 = 0
        probSentenceGivenL3 = 0
        probSentenceGivenL4 = 0
        probSentenceGivenL5 = 0
        probSentenceGivenL6 = 0
        
        #for zipping to bigram tuples
        tempBigram1 = zip(tempSentence1, islice(tempSentence1, 1, None))
        tempBigram2 = zip(tempSentence2, islice(tempSentence2, 1, None))
        tempBigram3 = zip(tempSentence3, islice(tempSentence3, 1, None))
        tempBigram4 = zip(tempSentence4, islice(tempSentence4, 1, None))
        tempBigram5 = zip(tempSentence5, islice(tempSentence5, 1, None))
        tempBigram6 = zip(tempSentence6, islice(tempSentence6, 1, None))
        
        #for storing values that will later have redistributed unigram probability
        tempUnigramBackoffList1 = {}
        tempUnigramBackoffList2 = {}
        tempUnigramBackoffList3 = {}
        tempUnigramBackoffList4 = {}
        tempUnigramBackoffList5 = {}
        tempUnigramBackoffList6 = {}
        
        for wordPair in tempBigram1 :
            #if c(x,y)>0 just count probability as c(x,y)-d2 / c(x)
            if wordPair in backedOffBigram1:
                normalizingConstant = (series1[wordPair[0]])
                tempProbability = (backedOffBigram1[wordPair])/(normalizingConstant)
                probSentenceGivenL1+=math.log(tempProbability)
            else:
                #otherwise check if c(x)>0 and if it is, count a(x) * (c(x)/ summation of c(v) across all V) as the probability  
                if wordPair[0] in backedOffList1:
                    tempProbability = redistributedBigram1[wordPair[0]] *( backedOffList1[wordPair[0]] / wordCount1)
                    probSentenceGivenL1+=math.log(tempProbability)
                else:
                    #otherwise add the word pair as a key to a list of double backed off values
                    if wordPair in tempUnigramBackoffList1:
                        tempUnigramBackoffList1[wordPair] += 1
                    else:
                        tempUnigramBackoffList1[wordPair] = 1

        for wordPair in tempBigram2 :
            #if c(x,y)>0 just count probability as c(x,y)-d2 / c(x)
            if wordPair in backedOffBigram2:
                normalizingConstant = (series2[wordPair[0]])
                tempProbability = (backedOffBigram2[wordPair])/(normalizingConstant)
                probSentenceGivenL2+=math.log(tempProbability)
            else:
                #otherwise check if c(x)>0 and if it is, count a(x) * (c(x)/ summation of c(v) across all V) as the probability  
                if wordPair[0] in backedOffList2:
                    tempProbability = redistributedBigram2[wordPair[0]] *( backedOffList2[wordPair[0]] / wordCount2)
                    probSentenceGivenL2+=math.log(tempProbability)
                else:
                    #otherwise add the word pair as a key to a list of double backed off values
                    if wordPair in tempUnigramBackoffList2:
                        tempUnigramBackoffList2[wordPair] += 1
                    else:
                        tempUnigramBackoffList2[wordPair] = 1
                
        for wordPair in tempBigram3 :
            #if c(x,y)>0 just count probability as c(x,y)-d2 / c(x)
            if wordPair in backedOffBigram3:
                normalizingConstant = (series3[wordPair[0]])
                tempProbability = (backedOffBigram3[wordPair])/(normalizingConstant)
                probSentenceGivenL3+=math.log(tempProbability)
            else:
                #otherwise check if c(x)>0 and if it is, count a(x) * (c(x)/ summation of c(v) across all V) as the probability  
                if wordPair[0] in backedOffList3:
                    tempProbability = redistributedBigram3[wordPair[0]] *( backedOffList3[wordPair[0]] / wordCount3)
                    probSentenceGivenL3+=math.log(tempProbability)
                else:
                    #otherwise add the word pair as a key to a list of double backed off values
                    if wordPair in tempUnigramBackoffList3:
                        tempUnigramBackoffList3[wordPair] += 1
                    else:
                        tempUnigramBackoffList3[wordPair] = 1
                
        for wordPair in tempBigram4 :
            #if c(x,y)>0 just count probability as c(x,y)-d2 / c(x)
            if wordPair in backedOffBigram4:
                normalizingConstant = (series4[wordPair[0]])
                tempProbability = (backedOffBigram4[wordPair])/(normalizingConstant)
                probSentenceGivenL4+=math.log(tempProbability)
            else:
                #otherwise check if c(x)>0 and if it is, count a(x) * (c(x)/ summation of c(v) across all V) as the probability  
                if wordPair[0] in backedOffList4:
                    tempProbability = redistributedBigram4[wordPair[0]] *( backedOffList4[wordPair[0]] / wordCount4)
                    probSentenceGivenL4+=math.log(tempProbability)
                else:
                    #otherwise add the word pair as a key to a list of double backed off values
                    if wordPair in tempUnigramBackoffList4:
                        tempUnigramBackoffList4[wordPair] += 1
                    else:
                        tempUnigramBackoffList4[wordPair] = 1
                
        for wordPair in tempBigram2 :
            #if c(x,y)>0 just count probability as c(x,y)-d2 / c(x)
            if wordPair in backedOffBigram5:
                normalizingConstant = (series5[wordPair[0]])
                tempProbability = (backedOffBigram5[wordPair])/(normalizingConstant)
                probSentenceGivenL5+=math.log(tempProbability)
            else:
                #otherwise check if c(x)>0 and if it is, count a(x) * (c(x)/ summation of c(v) across all V) as the probability  
                if wordPair[0] in backedOffList5:
                    tempProbability = redistributedBigram5[wordPair[0]] *( backedOffList5[wordPair[0]] / wordCount5)
                    probSentenceGivenL5+=math.log(tempProbability)
                else:
                    #otherwise add the word pair as a key to a list of double backed off values
                    if wordPair in tempUnigramBackoffList5:
                        tempUnigramBackoffList5[wordPair] += 1
                    else:
                        tempUnigramBackoffList5[wordPair] = 1
                
        for wordPair in tempBigram6 :
            #if c(x,y)>0 just count probability as c(x,y)-d2 / c(x)
            if wordPair in backedOffBigram6:
                normalizingConstant = (series6[wordPair[0]])
                tempProbability = (backedOffBigram6[wordPair])/(normalizingConstant)
                probSentenceGivenL6+=math.log(tempProbability)
            else:
                #otherwise check if c(x)>0 and if it is, count a(x) * (c(x)/ summation of c(v) across all V) as the probability  
                if wordPair[0] in backedOffList6:
                    tempProbability = redistributedBigram6[wordPair[0]] *( backedOffList6[wordPair[0]] / wordCount6)
                    probSentenceGivenL6+=math.log(tempProbability)
                else:
                    #otherwise add the word pair as a key to a list of double backed off values
                    if wordPair in tempUnigramBackoffList6:
                        tempUnigramBackoffList6[wordPair] += 1
                    else:
                        tempUnigramBackoffList6[wordPair] = 1
        
        #redistribute the probabilities that had to be backed off twice.
        print("Now redistributing unigram backoff as necessary.\n")
        
        #First need to count number of oov types
        numOOVTypes1 = len(tempUnigramBackoffList1)
        numOOVTypes2 = len(tempUnigramBackoffList2)
        numOOVTypes3 = len(tempUnigramBackoffList3)
        numOOVTypes4 = len(tempUnigramBackoffList4)
        numOOVTypes5 = len(tempUnigramBackoffList5)
        numOOVTypes6 = len(tempUnigramBackoffList6)
        
        #then need to get b/v values where V is the size of the total vocab of known and previously unknown newly encountered words
        #b is the probability mass set aside for redistribution
        distValueUnigram1 = redistributedUnigram1/(numOOVTypes1+sizeOfVocab1)
        distValueUnigram2 = redistributedUnigram2/(numOOVTypes2+sizeOfVocab2)
        distValueUnigram3 = redistributedUnigram3/(numOOVTypes3+sizeOfVocab3)
        distValueUnigram4 = redistributedUnigram4/(numOOVTypes4+sizeOfVocab4)
        distValueUnigram5 = redistributedUnigram5/(numOOVTypes5+sizeOfVocab5)
        distValueUnigram6 = redistributedUnigram6/(numOOVTypes6+sizeOfVocab6)

        #Now multiply this b/v value by its a(x) and add its log linear probability up as necessary per each occurence/count of double backed off bigram
        for key, value in tempUnigramBackoffList1.items():
            if key[0] in redistributedBigram1:
                tempProbabilitySum = value * math.log(redistributedBigram1[key[0]]*distValueUnigram1)
            else:
                tempProbabilitySum = value * math.log(redistributedBigram1['<UNK>']*distValueUnigram1)
            probSentenceGivenL1+=tempProbabilitySum
            
        #Now multiply this b/v value by its a(x) and add its log linear probability up as necessary per each occurence/count of double backed off bigram
        for key, value in tempUnigramBackoffList2.items():
            if key[0] in redistributedBigram2:
                tempProbabilitySum = value * math.log(redistributedBigram2[key[0]]*distValueUnigram2)
            else:
                tempProbabilitySum = value * math.log(redistributedBigram2['<UNK>']*distValueUnigram2)
            probSentenceGivenL2+=tempProbabilitySum
            
        #Now multiply this b/v value by its a(x) and add its log linear probability up as necessary per each occurence/count of double backed off bigram
        for key, value in tempUnigramBackoffList3.items():
            if key[0] in redistributedBigram3:
                tempProbabilitySum = value * math.log(redistributedBigram3[key[0]]*distValueUnigram3)
            else:
                tempProbabilitySum = value * math.log(redistributedBigram3['<UNK>']*distValueUnigram3)
            probSentenceGivenL1+=tempProbabilitySum
            
            
        #Now multiply this b/v value by its a(x) and add its log linear probability up as necessary per each occurence/count of double backed off bigram
        for key, value in tempUnigramBackoffList4.items():
            if key[0] in redistributedBigram4:
                tempProbabilitySum = value * math.log(redistributedBigram4[key[0]]*distValueUnigram4)
            else:
                tempProbabilitySum = value * math.log(redistributedBigram4['<UNK>']*distValueUnigram4)
            probSentenceGivenL4+=tempProbabilitySum
            
        #Now multiply this b/v value by its a(x) and add its log linear probability up as necessary per each occurence/count of double backed off bigram
        for key, value in tempUnigramBackoffList5.items():
            if key[0] in redistributedBigram5:
                tempProbabilitySum = value * math.log(redistributedBigram5[key[0]]*distValueUnigram5)
            else:
                tempProbabilitySum = value * math.log(redistributedBigram5['<UNK>']*distValueUnigram5)
            probSentenceGivenL5+=tempProbabilitySum
            
        #Now multiply this b/v value by its a(x) and add its log linear probability up as necessary per each occurence/count of double backed off bigram
        for key, value in tempUnigramBackoffList6.items():
            if key[0] in redistributedBigram6:
                tempProbabilitySum = value * math.log(redistributedBigram6[key[0]]*distValueUnigram6)
            else:
                tempProbabilitySum = value * math.log(redistributedBigram6['<UNK>']*distValueUnigram6)
            probSentenceGivenL6+=tempProbabilitySum
        
        
        print("Now predicting language")
        #predict which language it is using logs
        logProb1 = probSentenceGivenL1 + math.log(probLang1)
        logProb2 = probSentenceGivenL2 + math.log(probLang2)
        logProb3 = probSentenceGivenL3 + math.log(probLang3)
        logProb4 = probSentenceGivenL4 + math.log(probLang4)
        logProb5 = probSentenceGivenL5 + math.log(probLang5)
        logProb6 = probSentenceGivenL6 + math.log(probLang6)
        
        #store probabilities in dictionary with respective languages as keys
        probDict = {
            "Lang1": logProb1, 
            "Lang2": logProb2, 
            "Lang3": logProb3, 
            "Lang4": logProb4, 
            "Lang5": logProb5, 
            "Lang6": logProb6
            }
        
        #find maximum of these log likelihoods and set that as the predicted language
        Keymax = max(probDict, key=probDict.get)
        predictedLang.append(Keymax)
        
        #append the actual language this dev set is from to actual language list
        actualLang.append('Lang5')
        
        countDevLang5+=1
        print("Done with %d sentences in dev set for Lang 5"%(countDevLang5))
        
        #resetting the temporary list of words per each sentence 
        tempSentence1 = []
        tempSentence2 = []
        tempSentence3 = []
        tempSentence4 = []
        tempSentence5 = []
        tempSentence6 = []
        
        
    tempSentence1 = []
    tempSentence2 = []
    tempSentence3 = []
    tempSentence4 = []
    tempSentence5 = []
    tempSentence6 = []
    
    en_dev5.close()
    
    #connlu parse and update bigram and unigram counts 
    for tokenlist in parse_incr(en_dev6):
        for token in tokenlist:         
            #If sentence is in entire vocab from all languages, add the word to the sentence. Otherwise add the unknown token to the sentence.
            #Regardless, if the bigram doesn't exist in a specific language one will have laplace + 0 bigram count /unigram count+ laplace + V. 
            #If the unigram doesn't exist, one will have laplace + 0 bigram count / laplace + 0 + V. If the bigram exists one instead has bigram count+laplace
            # divided by unigram count + laplace + V
            if not (token["form"].lower()) in wordList :
                tempSentence1.append('<UNK>')
                tempSentence2.append('<UNK>')
                tempSentence3.append('<UNK>')
                tempSentence4.append('<UNK>')
                tempSentence5.append('<UNK>')
                tempSentence6.append('<UNK>')
            else:
                ##adding to temporary sentence which will be parsed into bigrams
                #making it lower case as a means of preprocessing. words of different case but same spelling are the same type for my purposes
                tempSentence1.append(token["form"].lower())
                tempSentence2.append(token["form"].lower())
                tempSentence3.append(token["form"].lower())
                tempSentence4.append(token["form"].lower())
                tempSentence5.append(token["form"].lower())
                tempSentence6.append(token["form"].lower())


        #now adding eos and bos tags to the sentence
        tempSentence1.insert(0, "<BOS>")
        tempSentence1.append("<EOS>")
        tempSentence2.insert(0, "<BOS>")
        tempSentence2.append("<EOS>")
        tempSentence3.insert(0, "<BOS>")
        tempSentence3.append("<EOS>")
        tempSentence4.insert(0, "<BOS>")
        tempSentence4.append("<EOS>")
        tempSentence5.insert(0, "<BOS>")
        tempSentence5.append("<EOS>")
        tempSentence6.insert(0, "<BOS>")
        tempSentence6.append("<EOS>")
        
        #this is math.log(1) since I am adding log probabilities to avoid multiplication
        probSentenceGivenL1 = 0
        probSentenceGivenL2 = 0
        probSentenceGivenL3 = 0
        probSentenceGivenL4 = 0
        probSentenceGivenL5 = 0
        probSentenceGivenL6 = 0
        
        #for zipping to bigram tuples
        tempBigram1 = zip(tempSentence1, islice(tempSentence1, 1, None))
        tempBigram2 = zip(tempSentence2, islice(tempSentence2, 1, None))
        tempBigram3 = zip(tempSentence3, islice(tempSentence3, 1, None))
        tempBigram4 = zip(tempSentence4, islice(tempSentence4, 1, None))
        tempBigram5 = zip(tempSentence5, islice(tempSentence5, 1, None))
        tempBigram6 = zip(tempSentence6, islice(tempSentence6, 1, None))
        
        #for storing values that will later have redistributed unigram probability
        tempUnigramBackoffList1 = {}
        tempUnigramBackoffList2 = {}
        tempUnigramBackoffList3 = {}
        tempUnigramBackoffList4 = {}
        tempUnigramBackoffList5 = {}
        tempUnigramBackoffList6 = {}
        
        for wordPair in tempBigram1 :
            #if c(x,y)>0 just count probability as c(x,y)-d2 / c(x)
            if wordPair in backedOffBigram1:
                normalizingConstant = (series1[wordPair[0]])
                tempProbability = (backedOffBigram1[wordPair])/(normalizingConstant)
                probSentenceGivenL1+=math.log(tempProbability)
            else:
                #otherwise check if c(x)>0 and if it is, count a(x) * (c(x)/ summation of c(v) across all V) as the probability  
                if wordPair[0] in backedOffList1:
                    tempProbability = redistributedBigram1[wordPair[0]] *( backedOffList1[wordPair[0]] / wordCount1)
                    probSentenceGivenL1+=math.log(tempProbability)
                else:
                    #otherwise add the word pair as a key to a list of double backed off values
                    if wordPair in tempUnigramBackoffList1:
                        tempUnigramBackoffList1[wordPair] += 1
                    else:
                        tempUnigramBackoffList1[wordPair] = 1

        for wordPair in tempBigram2 :
            #if c(x,y)>0 just count probability as c(x,y)-d2 / c(x)
            if wordPair in backedOffBigram2:
                normalizingConstant = (series2[wordPair[0]])
                tempProbability = (backedOffBigram2[wordPair])/(normalizingConstant)
                probSentenceGivenL2+=math.log(tempProbability)
            else:
                #otherwise check if c(x)>0 and if it is, count a(x) * (c(x)/ summation of c(v) across all V) as the probability  
                if wordPair[0] in backedOffList2:
                    tempProbability = redistributedBigram2[wordPair[0]] *( backedOffList2[wordPair[0]] / wordCount2)
                    probSentenceGivenL2+=math.log(tempProbability)
                else:
                    #otherwise add the word pair as a key to a list of double backed off values
                    if wordPair in tempUnigramBackoffList2:
                        tempUnigramBackoffList2[wordPair] += 1
                    else:
                        tempUnigramBackoffList2[wordPair] = 1
                
        for wordPair in tempBigram3 :
            #if c(x,y)>0 just count probability as c(x,y)-d2 / c(x)
            if wordPair in backedOffBigram3:
                normalizingConstant = (series3[wordPair[0]])
                tempProbability = (backedOffBigram3[wordPair])/(normalizingConstant)
                probSentenceGivenL3+=math.log(tempProbability)
            else:
                #otherwise check if c(x)>0 and if it is, count a(x) * (c(x)/ summation of c(v) across all V) as the probability  
                if wordPair[0] in backedOffList3:
                    tempProbability = redistributedBigram3[wordPair[0]] *( backedOffList3[wordPair[0]] / wordCount3)
                    probSentenceGivenL3+=math.log(tempProbability)
                else:
                    #otherwise add the word pair as a key to a list of double backed off values
                    if wordPair in tempUnigramBackoffList3:
                        tempUnigramBackoffList3[wordPair] += 1
                    else:
                        tempUnigramBackoffList3[wordPair] = 1
                
        for wordPair in tempBigram4 :
            #if c(x,y)>0 just count probability as c(x,y)-d2 / c(x)
            if wordPair in backedOffBigram4:
                normalizingConstant = (series4[wordPair[0]])
                tempProbability = (backedOffBigram4[wordPair])/(normalizingConstant)
                probSentenceGivenL4+=math.log(tempProbability)
            else:
                #otherwise check if c(x)>0 and if it is, count a(x) * (c(x)/ summation of c(v) across all V) as the probability  
                if wordPair[0] in backedOffList4:
                    tempProbability = redistributedBigram4[wordPair[0]] *( backedOffList4[wordPair[0]] / wordCount4)
                    probSentenceGivenL4+=math.log(tempProbability)
                else:
                    #otherwise add the word pair as a key to a list of double backed off values
                    if wordPair in tempUnigramBackoffList4:
                        tempUnigramBackoffList4[wordPair] += 1
                    else:
                        tempUnigramBackoffList4[wordPair] = 1
                
        for wordPair in tempBigram2 :
            #if c(x,y)>0 just count probability as c(x,y)-d2 / c(x)
            if wordPair in backedOffBigram5:
                normalizingConstant = (series5[wordPair[0]])
                tempProbability = (backedOffBigram5[wordPair])/(normalizingConstant)
                probSentenceGivenL5+=math.log(tempProbability)
            else:
                #otherwise check if c(x)>0 and if it is, count a(x) * (c(x)/ summation of c(v) across all V) as the probability  
                if wordPair[0] in backedOffList5:
                    tempProbability = redistributedBigram5[wordPair[0]] *( backedOffList5[wordPair[0]] / wordCount5)
                    probSentenceGivenL5+=math.log(tempProbability)
                else:
                    #otherwise add the word pair as a key to a list of double backed off values
                    if wordPair in tempUnigramBackoffList5:
                        tempUnigramBackoffList5[wordPair] += 1
                    else:
                        tempUnigramBackoffList5[wordPair] = 1
                
        for wordPair in tempBigram6 :
            #if c(x,y)>0 just count probability as c(x,y)-d2 / c(x)
            if wordPair in backedOffBigram6:
                normalizingConstant = (series6[wordPair[0]])
                tempProbability = (backedOffBigram6[wordPair])/(normalizingConstant)
                probSentenceGivenL6+=math.log(tempProbability)
            else:
                #otherwise check if c(x)>0 and if it is, count a(x) * (c(x)/ summation of c(v) across all V) as the probability  
                if wordPair[0] in backedOffList6:
                    tempProbability = redistributedBigram6[wordPair[0]] *( backedOffList6[wordPair[0]] / wordCount6)
                    probSentenceGivenL6+=math.log(tempProbability)
                else:
                    #otherwise add the word pair as a key to a list of double backed off values
                    if wordPair in tempUnigramBackoffList6:
                        tempUnigramBackoffList6[wordPair] += 1
                    else:
                        tempUnigramBackoffList6[wordPair] = 1
        
        #redistribute the probabilities that had to be backed off twice.
        print("Now redistributing unigram backoff as necessary.\n")
        
        #First need to count number of oov types
        numOOVTypes1 = len(tempUnigramBackoffList1)
        numOOVTypes2 = len(tempUnigramBackoffList2)
        numOOVTypes3 = len(tempUnigramBackoffList3)
        numOOVTypes4 = len(tempUnigramBackoffList4)
        numOOVTypes5 = len(tempUnigramBackoffList5)
        numOOVTypes6 = len(tempUnigramBackoffList6)
        
        #then need to get b/v values where V is the size of the total vocab of known and previously unknown newly encountered words
        #b is the probability mass set aside for redistribution
        distValueUnigram1 = redistributedUnigram1/(numOOVTypes1+sizeOfVocab1)
        distValueUnigram2 = redistributedUnigram2/(numOOVTypes2+sizeOfVocab2)
        distValueUnigram3 = redistributedUnigram3/(numOOVTypes3+sizeOfVocab3)
        distValueUnigram4 = redistributedUnigram4/(numOOVTypes4+sizeOfVocab4)
        distValueUnigram5 = redistributedUnigram5/(numOOVTypes5+sizeOfVocab5)
        distValueUnigram6 = redistributedUnigram6/(numOOVTypes6+sizeOfVocab6)

        #Now multiply this b/v value by its a(x) and add its log linear probability up as necessary per each occurence/count of double backed off bigram
        for key, value in tempUnigramBackoffList1.items():
            if key[0] in redistributedBigram1:
                tempProbabilitySum = value * math.log(redistributedBigram1[key[0]]*distValueUnigram1)
            else:
                tempProbabilitySum = value * math.log(redistributedBigram1['<UNK>']*distValueUnigram1)
            probSentenceGivenL1+=tempProbabilitySum
            
        #Now multiply this b/v value by its a(x) and add its log linear probability up as necessary per each occurence/count of double backed off bigram
        for key, value in tempUnigramBackoffList2.items():
            if key[0] in redistributedBigram2:
                tempProbabilitySum = value * math.log(redistributedBigram2[key[0]]*distValueUnigram2)
            else:
                tempProbabilitySum = value * math.log(redistributedBigram2['<UNK>']*distValueUnigram2)
            probSentenceGivenL2+=tempProbabilitySum
            
        #Now multiply this b/v value by its a(x) and add its log linear probability up as necessary per each occurence/count of double backed off bigram
        for key, value in tempUnigramBackoffList3.items():
            if key[0] in redistributedBigram3:
                tempProbabilitySum = value * math.log(redistributedBigram3[key[0]]*distValueUnigram3)
            else:
                tempProbabilitySum = value * math.log(redistributedBigram3['<UNK>']*distValueUnigram3)
            probSentenceGivenL1+=tempProbabilitySum
            
            
        #Now multiply this b/v value by its a(x) and add its log linear probability up as necessary per each occurence/count of double backed off bigram
        for key, value in tempUnigramBackoffList4.items():
            if key[0] in redistributedBigram4:
                tempProbabilitySum = value * math.log(redistributedBigram4[key[0]]*distValueUnigram4)
            else:
                tempProbabilitySum = value * math.log(redistributedBigram4['<UNK>']*distValueUnigram4)
            probSentenceGivenL4+=tempProbabilitySum
            
        #Now multiply this b/v value by its a(x) and add its log linear probability up as necessary per each occurence/count of double backed off bigram
        for key, value in tempUnigramBackoffList5.items():
            if key[0] in redistributedBigram5:
                tempProbabilitySum = value * math.log(redistributedBigram5[key[0]]*distValueUnigram5)
            else:
                tempProbabilitySum = value * math.log(redistributedBigram5['<UNK>']*distValueUnigram5)
            probSentenceGivenL5+=tempProbabilitySum
            
        #Now multiply this b/v value by its a(x) and add its log linear probability up as necessary per each occurence/count of double backed off bigram
        for key, value in tempUnigramBackoffList6.items():
            if key[0] in redistributedBigram6:
                tempProbabilitySum = value * math.log(redistributedBigram6[key[0]]*distValueUnigram6)
            else:
                tempProbabilitySum = value * math.log(redistributedBigram6['<UNK>']*distValueUnigram6)
            probSentenceGivenL6+=tempProbabilitySum
        
        print("Now predicting language")
        #predict which language it is using logs
        logProb1 = probSentenceGivenL1 + math.log(probLang1)
        logProb2 = probSentenceGivenL2 + math.log(probLang2)
        logProb3 = probSentenceGivenL3 + math.log(probLang3)
        logProb4 = probSentenceGivenL4 + math.log(probLang4)
        logProb5 = probSentenceGivenL5 + math.log(probLang5)
        logProb6 = probSentenceGivenL6 + math.log(probLang6)
        
        #store probabilities in dictionary with respective languages as keys
        probDict = {
            "Lang1": logProb1, 
            "Lang2": logProb2, 
            "Lang3": logProb3, 
            "Lang4": logProb4, 
            "Lang5": logProb5, 
            "Lang6": logProb6
            }
        
        #find maximum of these log likelihoods and set that as the predicted language
        Keymax = max(probDict, key=probDict.get)
        predictedLang.append(Keymax)
        
        #append the actual language this dev set is from to actual language list
        actualLang.append('Lang6')
        
        countDevLang6+=1
        print("Done with %d sentences in dev set for Lang 6"%(countDevLang6))
        
        #resetting the temporary list of words per each sentence 
        tempSentence1 = []
        tempSentence2 = []
        tempSentence3 = []
        tempSentence4 = []
        tempSentence5 = []
        tempSentence6 = []
        
        
    tempSentence1 = []
    tempSentence2 = []
    tempSentence3 = []
    tempSentence4 = []
    tempSentence5 = []
    tempSentence6 = []
    
    en_dev6.close()
    """
    """
    #connlu parse and update bigram and unigram counts 
    for tokenlist in parse_incr(en_dev2):
        for token in tokenlist:         
            #If sentence is in entire vocab from all languages, add the word to the sentence. Otherwise add the unknown token to the sentence.
            #Regardless, if the bigram doesn't exist in a specific language one will have laplace + 0 bigram count /unigram count+ laplace + V. 
            #If the unigram doesn't exist, one will have laplace + 0 bigram count / laplace + 0 + V. If the bigram exists one instead has bigram count+laplace
            # divided by unigram count + laplace + V
            if not (token["form"].lower()) in wordList :
                tempSentence1.append('<UNK>')
                tempSentence2.append('<UNK>')
                tempSentence3.append('<UNK>')
                tempSentence4.append('<UNK>')
                tempSentence5.append('<UNK>')
                tempSentence6.append('<UNK>')
            else:
                ##adding to temporary sentence which will be parsed into bigrams
                #making it lower case as a means of preprocessing. words of different case but same spelling are the same type for my purposes
                tempSentence1.append(token["form"].lower())
                tempSentence2.append(token["form"].lower())
                tempSentence3.append(token["form"].lower())
                tempSentence4.append(token["form"].lower())
                tempSentence5.append(token["form"].lower())
                tempSentence6.append(token["form"].lower())


        #now adding eos and bos tags to the sentence
        tempSentence1.insert(0, "<BOS>")
        tempSentence1.append("<EOS>")
        tempSentence2.insert(0, "<BOS>")
        tempSentence2.append("<EOS>")
        tempSentence3.insert(0, "<BOS>")
        tempSentence3.append("<EOS>")
        tempSentence4.insert(0, "<BOS>")
        tempSentence4.append("<EOS>")
        tempSentence5.insert(0, "<BOS>")
        tempSentence5.append("<EOS>")
        tempSentence6.insert(0, "<BOS>")
        tempSentence6.append("<EOS>")
        
        #this is math.log(1) since I am adding log probabilities to avoid multiplication
        probSentenceGivenL1 = 0
        probSentenceGivenL2 = 0
        probSentenceGivenL3 = 0
        probSentenceGivenL4 = 0
        probSentenceGivenL5 = 0
        probSentenceGivenL6 = 0
        
        tempBigram1 = zip(tempSentence1, islice(tempSentence1, 1, None))
        tempBigram2 = zip(tempSentence2, islice(tempSentence2, 1, None))
        tempBigram3 = zip(tempSentence3, islice(tempSentence3, 1, None))
        tempBigram4 = zip(tempSentence4, islice(tempSentence4, 1, None))
        tempBigram5 = zip(tempSentence5, islice(tempSentence5, 1, None))
        tempBigram6 = zip(tempSentence6, islice(tempSentence6, 1, None))
        
        for wordPair in tempBigram1 :
            if wordPair in bigramCounts1:
                normalizingConstant = (series1[wordPair[0]]+V)
                tempProbability = (bigramCounts1[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL1+=math.log(tempProbability)
            else:
                if not wordPair[0] in series1:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (series1[wordPair[0]]+V)
                ##normalizingConstant = (series1[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL1+=(math.log(tempProbability))
                    
        for wordPair in tempBigram2 :
            #####changing to use dictionary
            if wordPair in bigramCounts2:
                normalizingConstant = (series2[wordPair[0]]+V)
                tempProbability = (bigramCounts2[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL2+=math.log(tempProbability)
            else:
                if not wordPair[0] in series2:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (series2[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL2+=(math.log(tempProbability))
                
        for wordPair in tempBigram3 :
            #####changing to use dictionary
            if wordPair in bigramCounts3:
                normalizingConstant = (series3[wordPair[0]]+V)
                tempProbability = (bigramCounts3[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL3+=math.log(tempProbability)
            else:
                if not wordPair[0] in series3:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (series3[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL3+=(math.log(tempProbability))
                
        for wordPair in tempBigram4 :
            #####changing to use dictionary
            if wordPair in bigramCounts4:
                normalizingConstant = (series4[wordPair[0]]+V)
                tempProbability = (bigramCounts4[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL4+=math.log(tempProbability)
            else:
                if not wordPair[0] in series4:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (series4[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL4+=(math.log(tempProbability))
                
        for wordPair in tempBigram5 :
            #####changing to use dictionary
            if wordPair in bigramCounts5:
                normalizingConstant = (series5[wordPair[0]]+V)
                tempProbability = (bigramCounts5[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL5+=math.log(tempProbability)
            else:
                if not wordPair[0] in series5:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (series5[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL5+=(math.log(tempProbability))
                
        for wordPair in tempBigram6 :
            #####changing to use dictionary
            if wordPair in bigramCounts6:
                normalizingConstant = (series6[wordPair[0]]+V)
                tempProbability = (bigramCounts6[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL6+=math.log(tempProbability)
            else:
                if not wordPair[0] in series6:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (series6[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL6+=(math.log(tempProbability))
        
        
        
        #predict which language it is using logs
        logProb1 = probSentenceGivenL1 + math.log(probLang1)
        logProb2 = probSentenceGivenL2 + math.log(probLang2)
        logProb3 = probSentenceGivenL3 + math.log(probLang3)
        logProb4 = probSentenceGivenL4 + math.log(probLang4)
        logProb5 = probSentenceGivenL5 + math.log(probLang5)
        logProb6 = probSentenceGivenL6 + math.log(probLang6)
        
        #store probabilities in dictionary with respective languages as keys
        probDict = {
            "Lang1": logProb1, 
            "Lang2": logProb2, 
            "Lang3": logProb3, 
            "Lang4": logProb4, 
            "Lang5": logProb5, 
            "Lang6": logProb6
            }
        
        #find maximum of these log likelihoods and set that as the predicted language
        Keymax = max(probDict, key=probDict.get)
        predictedLang.append(Keymax)
        
        #append the actual language this dev set is from to actual language list
        actualLang.append('Lang2')
        
        countDevLang2+=1
        print("Done with %d sentences in dev set for Lang 2"%(countDevLang2))
        
        #resetting the temporary list of words per each sentence 
        tempSentence1 = []
        tempSentence2 = []
        tempSentence3 = []
        tempSentence4 = []
        tempSentence5 = []
        tempSentence6 = []
        
        
    tempSentence1 = []
    tempSentence2 = []
    tempSentence3 = []
    tempSentence4 = []
    tempSentence5 = []
    tempSentence6 = []
    
    en_dev2.close()
    
    #connlu parse and update bigram and unigram counts 
    for tokenlist in parse_incr(en_dev3):
        for token in tokenlist:         
            #If sentence is in entire vocab from all languages, add the word to the sentence. Otherwise add the unknown token to the sentence.
            #Regardless, if the bigram doesn't exist in a specific language one will have laplace + 0 bigram count /unigram count+ laplace + V. 
            #If the unigram doesn't exist, one will have laplace + 0 bigram count / laplace + 0 + V. If the bigram exists one instead has bigram count+laplace
            # divided by unigram count + laplace + V
            if not (token["form"].lower()) in wordList :
                tempSentence1.append('<UNK>')
                tempSentence2.append('<UNK>')
                tempSentence3.append('<UNK>')
                tempSentence4.append('<UNK>')
                tempSentence5.append('<UNK>')
                tempSentence6.append('<UNK>')
            else:
                ##adding to temporary sentence which will be parsed into bigrams
                #making it lower case as a means of preprocessing. words of different case but same spelling are the same type for my purposes
                tempSentence1.append(token["form"].lower())
                tempSentence2.append(token["form"].lower())
                tempSentence3.append(token["form"].lower())
                tempSentence4.append(token["form"].lower())
                tempSentence5.append(token["form"].lower())
                tempSentence6.append(token["form"].lower())


        #now adding eos and bos tags to the sentence
        tempSentence1.insert(0, "<BOS>")
        tempSentence1.append("<EOS>")
        tempSentence2.insert(0, "<BOS>")
        tempSentence2.append("<EOS>")
        tempSentence3.insert(0, "<BOS>")
        tempSentence3.append("<EOS>")
        tempSentence4.insert(0, "<BOS>")
        tempSentence4.append("<EOS>")
        tempSentence5.insert(0, "<BOS>")
        tempSentence5.append("<EOS>")
        tempSentence6.insert(0, "<BOS>")
        tempSentence6.append("<EOS>")
        
        #this is math.log(1) since I am adding log probabilities to avoid multiplication
        probSentenceGivenL1 = 0
        probSentenceGivenL2 = 0
        probSentenceGivenL3 = 0
        probSentenceGivenL4 = 0
        probSentenceGivenL5 = 0
        probSentenceGivenL6 = 0
        
        tempBigram1 = zip(tempSentence1, islice(tempSentence1, 1, None))
        tempBigram2 = zip(tempSentence2, islice(tempSentence2, 1, None))
        tempBigram3 = zip(tempSentence3, islice(tempSentence3, 1, None))
        tempBigram4 = zip(tempSentence4, islice(tempSentence4, 1, None))
        tempBigram5 = zip(tempSentence5, islice(tempSentence5, 1, None))
        tempBigram6 = zip(tempSentence6, islice(tempSentence6, 1, None))
        
        for wordPair in tempBigram1 :
            if wordPair in bigramCounts1:
                normalizingConstant = (series1[wordPair[0]]+V)
                tempProbability = (bigramCounts1[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL1+=math.log(tempProbability)
            else:
                if not wordPair[0] in series1:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (series1[wordPair[0]]+V)
                ##normalizingConstant = (series1[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL1+=(math.log(tempProbability))
                    
        for wordPair in tempBigram2 :
            #####changing to use dictionary
            if wordPair in bigramCounts2:
                normalizingConstant = (series2[wordPair[0]]+V)
                tempProbability = (bigramCounts2[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL2+=math.log(tempProbability)
            else:
                if not wordPair[0] in series2:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (series2[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL2+=(math.log(tempProbability))
                
        for wordPair in tempBigram3 :
            #####changing to use dictionary
            if wordPair in bigramCounts3:
                normalizingConstant = (series3[wordPair[0]]+V)
                tempProbability = (bigramCounts3[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL3+=math.log(tempProbability)
            else:
                if not wordPair[0] in series3:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (series3[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL3+=(math.log(tempProbability))
                
        for wordPair in tempBigram4 :
            #####changing to use dictionary
            if wordPair in bigramCounts4:
                normalizingConstant = (series4[wordPair[0]]+V)
                tempProbability = (bigramCounts4[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL4+=math.log(tempProbability)
            else:
                if not wordPair[0] in series4:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (series4[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL4+=(math.log(tempProbability))
                
        for wordPair in tempBigram5 :
            #####changing to use dictionary
            if wordPair in bigramCounts5:
                normalizingConstant = (series5[wordPair[0]]+V)
                tempProbability = (bigramCounts5[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL5+=math.log(tempProbability)
            else:
                if not wordPair[0] in series5:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (series5[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL5+=(math.log(tempProbability))
                
        for wordPair in tempBigram6 :
            #####changing to use dictionary
            if wordPair in bigramCounts6:
                normalizingConstant = (series6[wordPair[0]]+V)
                tempProbability = (bigramCounts6[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL6+=math.log(tempProbability)
            else:
                if not wordPair[0] in series6:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (series6[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL6+=(math.log(tempProbability))
        
        
        
        #predict which language it is using logs
        logProb1 = probSentenceGivenL1 + math.log(probLang1)
        logProb2 = probSentenceGivenL2 + math.log(probLang2)
        logProb3 = probSentenceGivenL3 + math.log(probLang3)
        logProb4 = probSentenceGivenL4 + math.log(probLang4)
        logProb5 = probSentenceGivenL5 + math.log(probLang5)
        logProb6 = probSentenceGivenL6 + math.log(probLang6)
        
        #store probabilities in dictionary with respective languages as keys
        probDict = {
            "Lang1": logProb1, 
            "Lang2": logProb2, 
            "Lang3": logProb3, 
            "Lang4": logProb4, 
            "Lang5": logProb5, 
            "Lang6": logProb6
        }
        
        #find maximum of these log likelihoods and set that as the predicted language
        Keymax = max(probDict, key=probDict.get)
        predictedLang.append(Keymax)
        
        #append the actual language this dev set is from to actual language list
        actualLang.append('Lang3')
        
        countDevLang3+=1
        print("Done with %d sentences in dev set for Lang 3"%(countDevLang3))
        
        #resetting the temporary list of words per each sentence 
        tempSentence1 = []
        tempSentence2 = []
        tempSentence3 = []
        tempSentence4 = []
        tempSentence5 = []
        tempSentence6 = []
        
        
    tempSentence1 = []
    tempSentence2 = []
    tempSentence3 = []
    tempSentence4 = []
    tempSentence5 = []
    tempSentence6 = []
    
    en_dev3.close()
    
    #connlu parse and update bigram and unigram counts 
    for tokenlist in parse_incr(en_dev4):
        for token in tokenlist:         
            #If sentence is in entire vocab from all languages, add the word to the sentence. Otherwise add the unknown token to the sentence.
            #Regardless, if the bigram doesn't exist in a specific language one will have laplace + 0 bigram count /unigram count+ laplace + V. 
            #If the unigram doesn't exist, one will have laplace + 0 bigram count / laplace + 0 + V. If the bigram exists one instead has bigram count+laplace
            # divided by unigram count + laplace + V
            if not (token["form"].lower()) in wordList :
                tempSentence1.append('<UNK>')
                tempSentence2.append('<UNK>')
                tempSentence3.append('<UNK>')
                tempSentence4.append('<UNK>')
                tempSentence5.append('<UNK>')
                tempSentence6.append('<UNK>')
            else:
                ##adding to temporary sentence which will be parsed into bigrams
                #making it lower case as a means of preprocessing. words of different case but same spelling are the same type for my purposes
                tempSentence1.append(token["form"].lower())
                tempSentence2.append(token["form"].lower())
                tempSentence3.append(token["form"].lower())
                tempSentence4.append(token["form"].lower())
                tempSentence5.append(token["form"].lower())
                tempSentence6.append(token["form"].lower())


        #now adding eos and bos tags to the sentence
        tempSentence1.insert(0, "<BOS>")
        tempSentence1.append("<EOS>")
        tempSentence2.insert(0, "<BOS>")
        tempSentence2.append("<EOS>")
        tempSentence3.insert(0, "<BOS>")
        tempSentence3.append("<EOS>")
        tempSentence4.insert(0, "<BOS>")
        tempSentence4.append("<EOS>")
        tempSentence5.insert(0, "<BOS>")
        tempSentence5.append("<EOS>")
        tempSentence6.insert(0, "<BOS>")
        tempSentence6.append("<EOS>")
        
        #this is math.log(1) since I am adding log probabilities to avoid multiplication
        probSentenceGivenL1 = 0
        probSentenceGivenL2 = 0
        probSentenceGivenL3 = 0
        probSentenceGivenL4 = 0
        probSentenceGivenL5 = 0
        probSentenceGivenL6 = 0
        
        tempBigram1 = zip(tempSentence1, islice(tempSentence1, 1, None))
        tempBigram2 = zip(tempSentence2, islice(tempSentence2, 1, None))
        tempBigram3 = zip(tempSentence3, islice(tempSentence3, 1, None))
        tempBigram4 = zip(tempSentence4, islice(tempSentence4, 1, None))
        tempBigram5 = zip(tempSentence5, islice(tempSentence5, 1, None))
        tempBigram6 = zip(tempSentence6, islice(tempSentence6, 1, None))
        
        for wordPair in tempBigram1 :
            if wordPair in bigramCounts1:
                normalizingConstant = (series1[wordPair[0]]+V)
                tempProbability = (bigramCounts1[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL1+=math.log(tempProbability)
            else:
                if not wordPair[0] in series1:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (series1[wordPair[0]]+V)
                ##normalizingConstant = (series1[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL1+=(math.log(tempProbability))
                    
        for wordPair in tempBigram2 :
            #####changing to use dictionary
            if wordPair in bigramCounts2:
                normalizingConstant = (series2[wordPair[0]]+V)
                tempProbability = (bigramCounts2[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL2+=math.log(tempProbability)
            else:
                if not wordPair[0] in series2:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (series2[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL2+=(math.log(tempProbability))
                
        for wordPair in tempBigram3 :
            #####changing to use dictionary
            if wordPair in bigramCounts3:
                normalizingConstant = (series3[wordPair[0]]+V)
                tempProbability = (bigramCounts3[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL3+=math.log(tempProbability)
            else:
                if not wordPair[0] in series3:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (series3[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL3+=(math.log(tempProbability))
                
        for wordPair in tempBigram4 :
            #####changing to use dictionary
            if wordPair in bigramCounts4:
                normalizingConstant = (series4[wordPair[0]]+V)
                tempProbability = (bigramCounts4[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL4+=math.log(tempProbability)
            else:
                if not wordPair[0] in series4:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (series4[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL4+=(math.log(tempProbability))
                
        for wordPair in tempBigram5 :
            #####changing to use dictionary
            if wordPair in bigramCounts5:
                normalizingConstant = (series5[wordPair[0]]+V)
                tempProbability = (bigramCounts5[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL5+=math.log(tempProbability)
            else:
                if not wordPair[0] in series5:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (series5[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL5+=(math.log(tempProbability))
                
        for wordPair in tempBigram6 :
            #####changing to use dictionary
            if wordPair in bigramCounts6:
                normalizingConstant = (series6[wordPair[0]]+V)
                tempProbability = (bigramCounts6[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL6+=math.log(tempProbability)
            else:
                if not wordPair[0] in series6:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (series6[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL6+=(math.log(tempProbability))
        
        
        
        #predict which language it is using logs
        logProb1 = probSentenceGivenL1 + math.log(probLang1)
        logProb2 = probSentenceGivenL2 + math.log(probLang2)
        logProb3 = probSentenceGivenL3 + math.log(probLang3)
        logProb4 = probSentenceGivenL4 + math.log(probLang4)
        logProb5 = probSentenceGivenL5 + math.log(probLang5)
        logProb6 = probSentenceGivenL6 + math.log(probLang6)
        
        #store probabilities in dictionary with respective languages as keys
        probDict = {
            "Lang1": logProb1, 
            "Lang2": logProb2, 
            "Lang3": logProb3, 
            "Lang4": logProb4, 
            "Lang5": logProb5, 
            "Lang6": logProb6
            }
        
        #find maximum of these log likelihoods and set that as the predicted language
        Keymax = max(probDict, key=probDict.get)
        predictedLang.append(Keymax)
        
        #append the actual language this dev set is from to actual language list
        actualLang.append('Lang4')
        
        countDevLang4+=1
        print("Done with %d sentences in dev set for Lang 4"%(countDevLang4))
        
        #resetting the temporary list of words per each sentence 
        tempSentence1 = []
        tempSentence2 = []
        tempSentence3 = []
        tempSentence4 = []
        tempSentence5 = []
        tempSentence6 = []
        
        
    tempSentence1 = []
    tempSentence2 = []
    tempSentence3 = []
    tempSentence4 = []
    tempSentence5 = []
    tempSentence6 = []
    
    en_dev4.close()
    
    #connlu parse and update bigram and unigram counts 
    for tokenlist in parse_incr(en_dev5):
        for token in tokenlist:         
            #If sentence is in entire vocab from all languages, add the word to the sentence. Otherwise add the unknown token to the sentence.
            #Regardless, if the bigram doesn't exist in a specific language one will have laplace + 0 bigram count /unigram count+ laplace + V. 
            #If the unigram doesn't exist, one will have laplace + 0 bigram count / laplace + 0 + V. If the bigram exists one instead has bigram count+laplace
            # divided by unigram count + laplace + V
            if not (token["form"].lower()) in wordList :
                tempSentence1.append('<UNK>')
                tempSentence2.append('<UNK>')
                tempSentence3.append('<UNK>')
                tempSentence4.append('<UNK>')
                tempSentence5.append('<UNK>')
                tempSentence6.append('<UNK>')
            else:
                ##adding to temporary sentence which will be parsed into bigrams
                #making it lower case as a means of preprocessing. words of different case but same spelling are the same type for my purposes
                tempSentence1.append(token["form"].lower())
                tempSentence2.append(token["form"].lower())
                tempSentence3.append(token["form"].lower())
                tempSentence4.append(token["form"].lower())
                tempSentence5.append(token["form"].lower())
                tempSentence6.append(token["form"].lower())


        #now adding eos and bos tags to the sentence
        tempSentence1.insert(0, "<BOS>")
        tempSentence1.append("<EOS>")
        tempSentence2.insert(0, "<BOS>")
        tempSentence2.append("<EOS>")
        tempSentence3.insert(0, "<BOS>")
        tempSentence3.append("<EOS>")
        tempSentence4.insert(0, "<BOS>")
        tempSentence4.append("<EOS>")
        tempSentence5.insert(0, "<BOS>")
        tempSentence5.append("<EOS>")
        tempSentence6.insert(0, "<BOS>")
        tempSentence6.append("<EOS>")
        
        #this is math.log(1) since I am adding log probabilities to avoid multiplication
        probSentenceGivenL1 = 0
        probSentenceGivenL2 = 0
        probSentenceGivenL3 = 0
        probSentenceGivenL4 = 0
        probSentenceGivenL5 = 0
        probSentenceGivenL6 = 0
        
        tempBigram1 = zip(tempSentence1, islice(tempSentence1, 1, None))
        tempBigram2 = zip(tempSentence2, islice(tempSentence2, 1, None))
        tempBigram3 = zip(tempSentence3, islice(tempSentence3, 1, None))
        tempBigram4 = zip(tempSentence4, islice(tempSentence4, 1, None))
        tempBigram5 = zip(tempSentence5, islice(tempSentence5, 1, None))
        tempBigram6 = zip(tempSentence6, islice(tempSentence6, 1, None))
        
        for wordPair in tempBigram1 :
            if wordPair in bigramCounts1:
                normalizingConstant = (series1[wordPair[0]]+V)
                tempProbability = (bigramCounts1[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL1+=math.log(tempProbability)
            else:
                if not wordPair[0] in series1:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (series1[wordPair[0]]+V)
                ##normalizingConstant = (series1[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL1+=(math.log(tempProbability))
                    
        for wordPair in tempBigram2 :
            #####changing to use dictionary
            if wordPair in bigramCounts2:
                normalizingConstant = (series2[wordPair[0]]+V)
                tempProbability = (bigramCounts2[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL2+=math.log(tempProbability)
            else:
                if not wordPair[0] in series2:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (series2[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL2+=(math.log(tempProbability))
                
        for wordPair in tempBigram3 :
            #####changing to use dictionary
            if wordPair in bigramCounts3:
                normalizingConstant = (series3[wordPair[0]]+V)
                tempProbability = (bigramCounts3[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL3+=math.log(tempProbability)
            else:
                if not wordPair[0] in series3:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (series3[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL3+=(math.log(tempProbability))
                
        for wordPair in tempBigram4 :
            #####changing to use dictionary
            if wordPair in bigramCounts4:
                normalizingConstant = (series4[wordPair[0]]+V)
                tempProbability = (bigramCounts4[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL4+=math.log(tempProbability)
            else:
                if not wordPair[0] in series4:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (series4[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL4+=(math.log(tempProbability))
                
        for wordPair in tempBigram5 :
            #####changing to use dictionary
            if wordPair in bigramCounts5:
                normalizingConstant = (series5[wordPair[0]]+V)
                tempProbability = (bigramCounts5[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL5+=math.log(tempProbability)
            else:
                if not wordPair[0] in series5:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (series5[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL5+=(math.log(tempProbability))
                
        for wordPair in tempBigram6 :
            #####changing to use dictionary
            if wordPair in bigramCounts6:
                normalizingConstant = (series6[wordPair[0]]+V)
                tempProbability = (bigramCounts6[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL6+=math.log(tempProbability)
            else:
                if not wordPair[0] in series6:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (series6[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL6+=(math.log(tempProbability))
        
        
        
        #predict which language it is using logs
        logProb1 = probSentenceGivenL1 + math.log(probLang1)
        logProb2 = probSentenceGivenL2 + math.log(probLang2)
        logProb3 = probSentenceGivenL3 + math.log(probLang3)
        logProb4 = probSentenceGivenL4 + math.log(probLang4)
        logProb5 = probSentenceGivenL5 + math.log(probLang5)
        logProb6 = probSentenceGivenL6 + math.log(probLang6)
        
        #store probabilities in dictionary with respective languages as keys
        probDict = {
            "Lang1": logProb1, 
            "Lang2": logProb2, 
            "Lang3": logProb3, 
            "Lang4": logProb4, 
            "Lang5": logProb5, 
            "Lang6": logProb6
            }
        
        #find maximum of these log likelihoods and set that as the predicted language
        Keymax = max(probDict, key=probDict.get)
        predictedLang.append(Keymax)
        
        #append the actual language this dev set is from to actual language list
        actualLang.append('Lang5')
        
        countDevLang5+=1
        print("Done with %d sentences in dev set for Lang 5"%(countDevLang5))
        
        #resetting the temporary list of words per each sentence 
        tempSentence1 = []
        tempSentence2 = []
        tempSentence3 = []
        tempSentence4 = []
        tempSentence5 = []
        tempSentence6 = []
        
        
    tempSentence1 = []
    tempSentence2 = []
    tempSentence3 = []
    tempSentence4 = []
    tempSentence5 = []
    tempSentence6 = []
    
    en_dev5.close()
    
    #connlu parse and update bigram and unigram counts 
    for tokenlist in parse_incr(en_dev6):
        for token in tokenlist:         
            #If sentence is in entire vocab from all languages, add the word to the sentence. Otherwise add the unknown token to the sentence.
            #Regardless, if the bigram doesn't exist in a specific language one will have laplace + 0 bigram count /unigram count+ laplace + V. 
            #If the unigram doesn't exist, one will have laplace + 0 bigram count / laplace + 0 + V. If the bigram exists one instead has bigram count+laplace
            # divided by unigram count + laplace + V
            if not (token["form"].lower()) in wordList :
                tempSentence1.append('<UNK>')
                tempSentence2.append('<UNK>')
                tempSentence3.append('<UNK>')
                tempSentence4.append('<UNK>')
                tempSentence5.append('<UNK>')
                tempSentence6.append('<UNK>')
            else:
                ##adding to temporary sentence which will be parsed into bigrams
                #making it lower case as a means of preprocessing. words of different case but same spelling are the same type for my purposes
                tempSentence1.append(token["form"].lower())
                tempSentence2.append(token["form"].lower())
                tempSentence3.append(token["form"].lower())
                tempSentence4.append(token["form"].lower())
                tempSentence5.append(token["form"].lower())
                tempSentence6.append(token["form"].lower())


        #now adding eos and bos tags to the sentence
        tempSentence1.insert(0, "<BOS>")
        tempSentence1.append("<EOS>")
        tempSentence2.insert(0, "<BOS>")
        tempSentence2.append("<EOS>")
        tempSentence3.insert(0, "<BOS>")
        tempSentence3.append("<EOS>")
        tempSentence4.insert(0, "<BOS>")
        tempSentence4.append("<EOS>")
        tempSentence5.insert(0, "<BOS>")
        tempSentence5.append("<EOS>")
        tempSentence6.insert(0, "<BOS>")
        tempSentence6.append("<EOS>")
        
        #this is math.log(1) since I am adding log probabilities to avoid multiplication
        probSentenceGivenL1 = 0
        probSentenceGivenL2 = 0
        probSentenceGivenL3 = 0
        probSentenceGivenL4 = 0
        probSentenceGivenL5 = 0
        probSentenceGivenL6 = 0
        
        tempBigram1 = zip(tempSentence1, islice(tempSentence1, 1, None))
        tempBigram2 = zip(tempSentence2, islice(tempSentence2, 1, None))
        tempBigram3 = zip(tempSentence3, islice(tempSentence3, 1, None))
        tempBigram4 = zip(tempSentence4, islice(tempSentence4, 1, None))
        tempBigram5 = zip(tempSentence5, islice(tempSentence5, 1, None))
        tempBigram6 = zip(tempSentence6, islice(tempSentence6, 1, None))
        
        for wordPair in tempBigram1 :
            if wordPair in bigramCounts1:
                normalizingConstant = (series1[wordPair[0]]+V)
                tempProbability = (bigramCounts1[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL1+=math.log(tempProbability)
            else:
                if not wordPair[0] in series1:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (series1[wordPair[0]]+V)
                ##normalizingConstant = (series1[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL1+=(math.log(tempProbability))
                    
        for wordPair in tempBigram2 :
            #####changing to use dictionary
            if wordPair in bigramCounts2:
                normalizingConstant = (series2[wordPair[0]]+V)
                tempProbability = (bigramCounts2[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL2+=math.log(tempProbability)
            else:
                if not wordPair[0] in series2:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (series2[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL2+=(math.log(tempProbability))
                
        for wordPair in tempBigram3 :
            #####changing to use dictionary
            if wordPair in bigramCounts3:
                normalizingConstant = (series3[wordPair[0]]+V)
                tempProbability = (bigramCounts3[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL3+=math.log(tempProbability)
            else:
                if not wordPair[0] in series3:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (series3[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL3+=(math.log(tempProbability))
                
        for wordPair in tempBigram4 :
            #####changing to use dictionary
            if wordPair in bigramCounts4:
                normalizingConstant = (series4[wordPair[0]]+V)
                tempProbability = (bigramCounts4[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL4+=math.log(tempProbability)
            else:
                if not wordPair[0] in series4:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (series4[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL4+=(math.log(tempProbability))
                
        for wordPair in tempBigram5 :
            #####changing to use dictionary
            if wordPair in bigramCounts5:
                normalizingConstant = (series5[wordPair[0]]+V)
                tempProbability = (bigramCounts5[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL5+=math.log(tempProbability)
            else:
                if not wordPair[0] in series5:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (series5[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL5+=(math.log(tempProbability))
                
        for wordPair in tempBigram6 :
            #####changing to use dictionary
            if wordPair in bigramCounts6:
                normalizingConstant = (series6[wordPair[0]]+V)
                tempProbability = (bigramCounts6[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL6+=math.log(tempProbability)
            else:
                if not wordPair[0] in series6:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (series6[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL6+=(math.log(tempProbability))
        
        
        
        #predict which language it is using logs
        logProb1 = probSentenceGivenL1 + math.log(probLang1)
        logProb2 = probSentenceGivenL2 + math.log(probLang2)
        logProb3 = probSentenceGivenL3 + math.log(probLang3)
        logProb4 = probSentenceGivenL4 + math.log(probLang4)
        logProb5 = probSentenceGivenL5 + math.log(probLang5)
        logProb6 = probSentenceGivenL6 + math.log(probLang6)
        
        #store probabilities in dictionary with respective languages as keys
        probDict = {
            "Lang1": logProb1, 
            "Lang2": logProb2, 
            "Lang3": logProb3, 
            "Lang4": logProb4, 
            "Lang5": logProb5, 
            "Lang6": logProb6
            }
        
        #find maximum of these log likelihoods and set that as the predicted language
        Keymax = max(probDict, key=probDict.get)
        predictedLang.append(Keymax)
        
        #append the actual language this dev set is from to actual language list
        actualLang.append('Lang6')
        
        countDevLang6+=1
        print("Done with %d sentences in dev set for Lang 6"%(countDevLang6))
        
        #resetting the temporary list of words per each sentence 
        tempSentence1 = []
        tempSentence2 = []
        tempSentence3 = []
        tempSentence4 = []
        tempSentence5 = []
        tempSentence6 = []
        
        
    tempSentence1 = []
    tempSentence2 = []
    tempSentence3 = []
    tempSentence4 = []
    tempSentence5 = []
    tempSentence6 = []
    
    en_dev6.close() 
    """
    
    print("Now calculating precision recall and f1 scores\n")
    
    #calculate precision and recall using scikit python module
    precision = precision_score(actualLang, predictedLang,average = "macro")
    
    recall = recall_score(actualLang, predictedLang,average = "macro")
    
    print(
        "\nTotal number of sentences in dev set 1 is %d, in dev set 2 is %d"
        ", in dev set 3 is %d, in dev set 4 is %d, in dev set 5 is %d, and in dev set 6 is %d."
        %(countDevLang1,countDevLang2,countDevLang3,countDevLang4,countDevLang5,countDevLang6)
    )
    
    print("\nPrecision is:")
    print(precision)
    
    print("Recall is:")
    print(recall)

    f1Score = (2*precision*recall)/(precision+recall)
    print("F1Score is:")
    print(f1Score)
    
    
#check if correct number of arguments 
if (len(sys.argv) <13  ) :
    print("Incorrect number of arguments for the script")

else:
    if len(sys.argv) >= 15:
        bigramLaplace(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],sys.argv[6], sys.argv[7],
            sys.argv[8] , sys.argv[9], sys.argv[10], sys.argv[11], sys.argv[12], sys.argv[13],sys.argv[14])
    elif len(sys.argv) == 15: 
        bigramLaplace(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],sys.argv[6], sys.argv[7],
            sys.argv[8] , sys.argv[9], sys.argv[10], sys.argv[11], sys.argv[12], sys.argv[13])
    else : 
        bigramLaplace(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],sys.argv[6], sys.argv[7],
            sys.argv[8] , sys.argv[9], sys.argv[10], sys.argv[11], sys.argv[12])