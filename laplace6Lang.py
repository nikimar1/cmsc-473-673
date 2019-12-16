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
    , pathToTuneLang1, pathToTuneLang2, pathToTuneLang3, pathToTuneLang4, pathToTuneLang5, pathToTuneLang6, laplaceConstant = .1):
    
    #removed oov constant implementation 
    #I am populating some unknown token list with a minimal training set of tokens with frequency one. 
    #This is because having no probability for unknown tokens is bad form in my opinion and this distributes at least a small probability to <UNK>
    #if int(oovConstant)>=1:
    #    oovFrequency = int(oovConstant)
    #else:
    #    oovFrequency = 1
    
    #get laplace constant from hyperparamters. cannot be 0 or less. should probably be less than 1 for better performance
    if float(laplaceConstant)> 0:
        laplace = float (laplaceConstant)
    else:
        laplace = .1

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
    
    #connlu parse and update bigram and unigram counts 
    for tokenlist in parse_incr(train3):
        for token in tokenlist:
            ##adding to temporary sentence which will be parsed into bigrams
            #making it lower case as a means of preprocessing. words of different case but same spelling are the same type for my purposes
            tempSentence.append(token["form"].lower())
            #adding to list of tokens. this will later be used to get unigram counts 
            list3.append(token["form"].lower())
            
        #now adding eos and bos tags to the sentence
        tempSentence.insert(0, "<BOS>")
        tempSentence.append("<EOS>")
        
        #now adding them to unigram token list 
        list3.append("<BOS>")
        list3.append("<EOS>")

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
            
        #now adding eos and bos tags to the sentence
        tempSentence.insert(0, "<BOS>")
        tempSentence.append("<EOS>")
        
        #now adding them to unigram token list 
        list4.append("<BOS>")
        list4.append("<EOS>")

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
            
        #now adding eos and bos tags to the sentence
        tempSentence.insert(0, "<BOS>")
        tempSentence.append("<EOS>")
        
        #now adding them to unigram token list 
        list5.append("<BOS>")
        list5.append("<EOS>")

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
            
        #now adding eos and bos tags to the sentence
        tempSentence.insert(0, "<BOS>")
        tempSentence.append("<EOS>")
        
        #now adding them to unigram token list 
        list6.append("<BOS>")
        list6.append("<EOS>")

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
    
    #left in previous code for oov cutoff. using 0 instead of oovfreqency cutoff variable
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
    filteredList1.at['<UNK>'] = laplace

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
    filteredList2.at['<UNK>'] = laplace

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
    filteredList3.at['<UNK>'] = laplace
    
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
    filteredList4.at['<UNK>'] = laplace
    
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
    filteredList5.at['<UNK>'] = laplace
    
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
    filteredList6.at['<UNK>'] = laplace

    #for storing list of words (types) in the vocab which will be used for indexing the rows and columns of the dataframe
    wordList1 = filteredList1.keys().tolist()
    wordList2 = filteredList2.keys().tolist()
    wordList3 = filteredList3.keys().tolist()
    wordList4 = filteredList4.keys().tolist()
    wordList5 = filteredList5.keys().tolist()
    wordList6 = filteredList6.keys().tolist()

    #get number of types
    sizeOfVocab1=len(wordList1)
    sizeOfVocab2=len(wordList2)
    sizeOfVocab3=len(wordList3)
    sizeOfVocab4=len(wordList4)
    sizeOfVocab5=len(wordList5)
    sizeOfVocab6=len(wordList6)

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
                bigramCounts3[(wordPair[0], wordPair[1])] = 1
                
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
                bigramCounts4[(wordPair[0], wordPair[1])] = 1
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
                bigramCounts5[(wordPair[0], wordPair[1])] = 1

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
                bigramCounts6[(wordPair[0], wordPair[1])] = 1
    
    #get entire vocab and V value if laplace is assigned to each of these words
    wordList = wordList1+wordList2+wordList3+wordList4+wordList5+wordList6
    V = laplace * (sizeOfVocab1 + sizeOfVocab2 + sizeOfVocab3 + sizeOfVocab4 + sizeOfVocab5 + sizeOfVocab6)

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
        
        tempBigram1 = zip(tempSentence1, islice(tempSentence1, 1, None))
        tempBigram2 = zip(tempSentence2, islice(tempSentence2, 1, None))
        tempBigram3 = zip(tempSentence3, islice(tempSentence3, 1, None))
        tempBigram4 = zip(tempSentence4, islice(tempSentence4, 1, None))
        tempBigram5 = zip(tempSentence5, islice(tempSentence5, 1, None))
        tempBigram6 = zip(tempSentence6, islice(tempSentence6, 1, None))
        
        for wordPair in tempBigram1 :
            if wordPair in bigramCounts1:
                normalizingConstant = (filteredList1[wordPair[0]]+V)
                tempProbability = (bigramCounts1[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL1+=math.log(tempProbability)
            else:
                if not wordPair[0] in filteredList1:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (filteredList1[wordPair[0]]+V)
                ##normalizingConstant = (filteredList1[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL1+=(math.log(tempProbability))
                    
        for wordPair in tempBigram2 :
            #####changing to use dictionary
            if wordPair in bigramCounts2:
                normalizingConstant = (filteredList2[wordPair[0]]+V)
                tempProbability = (bigramCounts2[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL2+=math.log(tempProbability)
            else:
                if not wordPair[0] in filteredList2:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (filteredList2[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL2+=(math.log(tempProbability))
                
        for wordPair in tempBigram3 :
            #####changing to use dictionary
            if wordPair in bigramCounts3:
                normalizingConstant = (filteredList3[wordPair[0]]+V)
                tempProbability = (bigramCounts3[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL3+=math.log(tempProbability)
            else:
                if not wordPair[0] in filteredList3:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (filteredList3[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL3+=(math.log(tempProbability))
                
        for wordPair in tempBigram4 :
            #####changing to use dictionary
            if wordPair in bigramCounts4:
                normalizingConstant = (filteredList4[wordPair[0]]+V)
                tempProbability = (bigramCounts4[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL4+=math.log(tempProbability)
            else:
                if not wordPair[0] in filteredList4:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (filteredList4[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL4+=(math.log(tempProbability))
                
        for wordPair in tempBigram5 :
            #####changing to use dictionary
            if wordPair in bigramCounts5:
                normalizingConstant = (filteredList5[wordPair[0]]+V)
                tempProbability = (bigramCounts5[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL5+=math.log(tempProbability)
            else:
                if not wordPair[0] in filteredList5:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (filteredList5[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL5+=(math.log(tempProbability))
                
        for wordPair in tempBigram6 :
            #####changing to use dictionary
            if wordPair in bigramCounts6:
                normalizingConstant = (filteredList6[wordPair[0]]+V)
                tempProbability = (bigramCounts6[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL6+=math.log(tempProbability)
            else:
                if not wordPair[0] in filteredList6:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (filteredList6[wordPair[0]]+V)
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
        actualLang.append('Lang1')
        
        countDevLang1+=1
        print("Done with %d sentences in dev set for Lang 1"%(countDevLang1))
        
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
                normalizingConstant = (filteredList1[wordPair[0]]+V)
                tempProbability = (bigramCounts1[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL1+=math.log(tempProbability)
            else:
                if not wordPair[0] in filteredList1:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (filteredList1[wordPair[0]]+V)
                ##normalizingConstant = (filteredList1[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL1+=(math.log(tempProbability))
                    
        for wordPair in tempBigram2 :
            #####changing to use dictionary
            if wordPair in bigramCounts2:
                normalizingConstant = (filteredList2[wordPair[0]]+V)
                tempProbability = (bigramCounts2[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL2+=math.log(tempProbability)
            else:
                if not wordPair[0] in filteredList2:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (filteredList2[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL2+=(math.log(tempProbability))
                
        for wordPair in tempBigram3 :
            #####changing to use dictionary
            if wordPair in bigramCounts3:
                normalizingConstant = (filteredList3[wordPair[0]]+V)
                tempProbability = (bigramCounts3[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL3+=math.log(tempProbability)
            else:
                if not wordPair[0] in filteredList3:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (filteredList3[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL3+=(math.log(tempProbability))
                
        for wordPair in tempBigram4 :
            #####changing to use dictionary
            if wordPair in bigramCounts4:
                normalizingConstant = (filteredList4[wordPair[0]]+V)
                tempProbability = (bigramCounts4[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL4+=math.log(tempProbability)
            else:
                if not wordPair[0] in filteredList4:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (filteredList4[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL4+=(math.log(tempProbability))
                
        for wordPair in tempBigram5 :
            #####changing to use dictionary
            if wordPair in bigramCounts5:
                normalizingConstant = (filteredList5[wordPair[0]]+V)
                tempProbability = (bigramCounts5[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL5+=math.log(tempProbability)
            else:
                if not wordPair[0] in filteredList5:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (filteredList5[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL5+=(math.log(tempProbability))
                
        for wordPair in tempBigram6 :
            #####changing to use dictionary
            if wordPair in bigramCounts6:
                normalizingConstant = (filteredList6[wordPair[0]]+V)
                tempProbability = (bigramCounts6[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL6+=math.log(tempProbability)
            else:
                if not wordPair[0] in filteredList6:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (filteredList6[wordPair[0]]+V)
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
                normalizingConstant = (filteredList1[wordPair[0]]+V)
                tempProbability = (bigramCounts1[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL1+=math.log(tempProbability)
            else:
                if not wordPair[0] in filteredList1:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (filteredList1[wordPair[0]]+V)
                ##normalizingConstant = (filteredList1[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL1+=(math.log(tempProbability))
                    
        for wordPair in tempBigram2 :
            #####changing to use dictionary
            if wordPair in bigramCounts2:
                normalizingConstant = (filteredList2[wordPair[0]]+V)
                tempProbability = (bigramCounts2[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL2+=math.log(tempProbability)
            else:
                if not wordPair[0] in filteredList2:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (filteredList2[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL2+=(math.log(tempProbability))
                
        for wordPair in tempBigram3 :
            #####changing to use dictionary
            if wordPair in bigramCounts3:
                normalizingConstant = (filteredList3[wordPair[0]]+V)
                tempProbability = (bigramCounts3[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL3+=math.log(tempProbability)
            else:
                if not wordPair[0] in filteredList3:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (filteredList3[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL3+=(math.log(tempProbability))
                
        for wordPair in tempBigram4 :
            #####changing to use dictionary
            if wordPair in bigramCounts4:
                normalizingConstant = (filteredList4[wordPair[0]]+V)
                tempProbability = (bigramCounts4[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL4+=math.log(tempProbability)
            else:
                if not wordPair[0] in filteredList4:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (filteredList4[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL4+=(math.log(tempProbability))
                
        for wordPair in tempBigram5 :
            #####changing to use dictionary
            if wordPair in bigramCounts5:
                normalizingConstant = (filteredList5[wordPair[0]]+V)
                tempProbability = (bigramCounts5[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL5+=math.log(tempProbability)
            else:
                if not wordPair[0] in filteredList5:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (filteredList5[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL5+=(math.log(tempProbability))
                
        for wordPair in tempBigram6 :
            #####changing to use dictionary
            if wordPair in bigramCounts6:
                normalizingConstant = (filteredList6[wordPair[0]]+V)
                tempProbability = (bigramCounts6[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL6+=math.log(tempProbability)
            else:
                if not wordPair[0] in filteredList6:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (filteredList6[wordPair[0]]+V)
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
                normalizingConstant = (filteredList1[wordPair[0]]+V)
                tempProbability = (bigramCounts1[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL1+=math.log(tempProbability)
            else:
                if not wordPair[0] in filteredList1:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (filteredList1[wordPair[0]]+V)
                ##normalizingConstant = (filteredList1[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL1+=(math.log(tempProbability))
                    
        for wordPair in tempBigram2 :
            #####changing to use dictionary
            if wordPair in bigramCounts2:
                normalizingConstant = (filteredList2[wordPair[0]]+V)
                tempProbability = (bigramCounts2[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL2+=math.log(tempProbability)
            else:
                if not wordPair[0] in filteredList2:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (filteredList2[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL2+=(math.log(tempProbability))
                
        for wordPair in tempBigram3 :
            #####changing to use dictionary
            if wordPair in bigramCounts3:
                normalizingConstant = (filteredList3[wordPair[0]]+V)
                tempProbability = (bigramCounts3[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL3+=math.log(tempProbability)
            else:
                if not wordPair[0] in filteredList3:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (filteredList3[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL3+=(math.log(tempProbability))
                
        for wordPair in tempBigram4 :
            #####changing to use dictionary
            if wordPair in bigramCounts4:
                normalizingConstant = (filteredList4[wordPair[0]]+V)
                tempProbability = (bigramCounts4[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL4+=math.log(tempProbability)
            else:
                if not wordPair[0] in filteredList4:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (filteredList4[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL4+=(math.log(tempProbability))
                
        for wordPair in tempBigram5 :
            #####changing to use dictionary
            if wordPair in bigramCounts5:
                normalizingConstant = (filteredList5[wordPair[0]]+V)
                tempProbability = (bigramCounts5[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL5+=math.log(tempProbability)
            else:
                if not wordPair[0] in filteredList5:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (filteredList5[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL5+=(math.log(tempProbability))
                
        for wordPair in tempBigram6 :
            #####changing to use dictionary
            if wordPair in bigramCounts6:
                normalizingConstant = (filteredList6[wordPair[0]]+V)
                tempProbability = (bigramCounts6[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL6+=math.log(tempProbability)
            else:
                if not wordPair[0] in filteredList6:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (filteredList6[wordPair[0]]+V)
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
                normalizingConstant = (filteredList1[wordPair[0]]+V)
                tempProbability = (bigramCounts1[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL1+=math.log(tempProbability)
            else:
                if not wordPair[0] in filteredList1:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (filteredList1[wordPair[0]]+V)
                ##normalizingConstant = (filteredList1[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL1+=(math.log(tempProbability))
                    
        for wordPair in tempBigram2 :
            #####changing to use dictionary
            if wordPair in bigramCounts2:
                normalizingConstant = (filteredList2[wordPair[0]]+V)
                tempProbability = (bigramCounts2[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL2+=math.log(tempProbability)
            else:
                if not wordPair[0] in filteredList2:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (filteredList2[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL2+=(math.log(tempProbability))
                
        for wordPair in tempBigram3 :
            #####changing to use dictionary
            if wordPair in bigramCounts3:
                normalizingConstant = (filteredList3[wordPair[0]]+V)
                tempProbability = (bigramCounts3[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL3+=math.log(tempProbability)
            else:
                if not wordPair[0] in filteredList3:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (filteredList3[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL3+=(math.log(tempProbability))
                
        for wordPair in tempBigram4 :
            #####changing to use dictionary
            if wordPair in bigramCounts4:
                normalizingConstant = (filteredList4[wordPair[0]]+V)
                tempProbability = (bigramCounts4[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL4+=math.log(tempProbability)
            else:
                if not wordPair[0] in filteredList4:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (filteredList4[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL4+=(math.log(tempProbability))
                
        for wordPair in tempBigram5 :
            #####changing to use dictionary
            if wordPair in bigramCounts5:
                normalizingConstant = (filteredList5[wordPair[0]]+V)
                tempProbability = (bigramCounts5[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL5+=math.log(tempProbability)
            else:
                if not wordPair[0] in filteredList5:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (filteredList5[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL5+=(math.log(tempProbability))
                
        for wordPair in tempBigram6 :
            #####changing to use dictionary
            if wordPair in bigramCounts6:
                normalizingConstant = (filteredList6[wordPair[0]]+V)
                tempProbability = (bigramCounts6[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL6+=math.log(tempProbability)
            else:
                if not wordPair[0] in filteredList6:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (filteredList6[wordPair[0]]+V)
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
                normalizingConstant = (filteredList1[wordPair[0]]+V)
                tempProbability = (bigramCounts1[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL1+=math.log(tempProbability)
            else:
                if not wordPair[0] in filteredList1:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (filteredList1[wordPair[0]]+V)
                ##normalizingConstant = (filteredList1[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL1+=(math.log(tempProbability))
                    
        for wordPair in tempBigram2 :
            #####changing to use dictionary
            if wordPair in bigramCounts2:
                normalizingConstant = (filteredList2[wordPair[0]]+V)
                tempProbability = (bigramCounts2[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL2+=math.log(tempProbability)
            else:
                if not wordPair[0] in filteredList2:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (filteredList2[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL2+=(math.log(tempProbability))
                
        for wordPair in tempBigram3 :
            #####changing to use dictionary
            if wordPair in bigramCounts3:
                normalizingConstant = (filteredList3[wordPair[0]]+V)
                tempProbability = (bigramCounts3[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL3+=math.log(tempProbability)
            else:
                if not wordPair[0] in filteredList3:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (filteredList3[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL3+=(math.log(tempProbability))
                
        for wordPair in tempBigram4 :
            #####changing to use dictionary
            if wordPair in bigramCounts4:
                normalizingConstant = (filteredList4[wordPair[0]]+V)
                tempProbability = (bigramCounts4[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL4+=math.log(tempProbability)
            else:
                if not wordPair[0] in filteredList4:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (filteredList4[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL4+=(math.log(tempProbability))
                
        for wordPair in tempBigram5 :
            #####changing to use dictionary
            if wordPair in bigramCounts5:
                normalizingConstant = (filteredList5[wordPair[0]]+V)
                tempProbability = (bigramCounts5[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL5+=math.log(tempProbability)
            else:
                if not wordPair[0] in filteredList5:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (filteredList5[wordPair[0]]+V)
                tempProbability = (laplace)/(normalizingConstant)
                probSentenceGivenL5+=(math.log(tempProbability))
                
        for wordPair in tempBigram6 :
            #####changing to use dictionary
            if wordPair in bigramCounts6:
                normalizingConstant = (filteredList6[wordPair[0]]+V)
                tempProbability = (bigramCounts6[wordPair]+laplace)/(normalizingConstant)
                probSentenceGivenL6+=math.log(tempProbability)
            else:
                if not wordPair[0] in filteredList6:
                    normalizingConstant = laplace + V
                else:
                    normalizingConstant = (filteredList6[wordPair[0]]+V)
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
    if len(sys.argv) >= 14:
        bigramLaplace(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],sys.argv[6], sys.argv[7],
            sys.argv[8] , sys.argv[9], sys.argv[10], sys.argv[11], sys.argv[12], sys.argv[13])
    else : 
        bigramLaplace(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],sys.argv[6], sys.argv[7],
            sys.argv[8] , sys.argv[9], sys.argv[10], sys.argv[11], sys.argv[12])