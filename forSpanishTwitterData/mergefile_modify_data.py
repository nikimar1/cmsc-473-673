import io
import fnmatch
#had to pip install wordsegment
from wordsegment import load, segment
from wordsegment import clean

def main():
    #list for sentences
    sentences = []
    file_name = "mergefile.txt"
    file = open(file_name, "r", encoding="utf-8")
    data = file.readlines()
    for i in data:
        i = i.strip('\n').replace('["',"").replace('"]',"")
        sentence = i.split('","')[1]	
        # delete one record in data "485501831406026752"
        words = sentence.split(" ")
        #add list of words to sentence list
        sentences.append(words)
     
    #for loading word segment     
    load()
      
    #for storing cleaned and segmented/spaced out words (for hashtag seperation)
    cleanedWords = []
    #for storing resultant sentences
    cleanedSentences = []
    
    for sentence in sentences:
        for word in sentence:
        
            #finds hashtags by using # and wildcard along with fnmatch module
            filtered = fnmatch.filter(word, '#*')
               
            #if no hashtag in the current word, append cleaned version of it which removes some punctuation, lower cases, and otherwise preprocesses
            if not filtered:
                cleanedWords.append(clean(word))
                
            #otherwise use segment to try to break it up a hashtag into distinct words
            else:
                cleanedWords.extend(segment(word))
                
        #add resultant list of words to cleaned sentence structure as well as resetting container for word lists
        cleanedSentences.append(cleanedWords)
        cleanedWords= []
              
    #iterate over and print newly cleaned up sentences
    for sentenceClean in cleanedSentences:
        print (sentenceClean)

if __name__ == '__main__':
    main()