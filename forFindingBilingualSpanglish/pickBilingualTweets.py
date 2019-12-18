import csv

#This program is intended to provide a command line ui for picking out bilingual tweets 
#from the two csv tables we made after cleaning up UNK and ES data.

#For list storing supposed bilingual tweets
biList = []

print("The following file will go over two tables to query you for bilingual sentence recognition.\n"\
    "You are able to quit out from each table at any time to get partial results since the tables are long.\n"\
    "Our program will update how many bilingual strings you have found so far so try to get\na one hundred at least"\
    "in total over both tables.\n")

#input for starting row from table 1 to look at (from 0 to length of table1 - 2)
startTableOne = int(input("\nInput what row from table one you wish to start with as an integer (0 for first row):"))

#newline           
print()

#input for starting row from table 2 to look at (from 0 to length of table2 - 2)
startTableTwo = int(input("\nInput what row from table two you wish to start with as an integer (0 for first row):"))

print("\nStarting with table one of es labelled tweets.")
#for checking if you want to stop iterating over this table
timeToQuit = False
#for checking if you have iterated to the correct starting row
rowFound = False

#for counting number of rows
with open("dataFromEs.csv", 'r') as f:  #opens data file
    reader = csv.reader(f)
    row_count = row_max = sum(1 for row in reader)
    
#for iterating over data sentence by sentence and prompting user if the sentence is spanglish or if they want to quit this table.
with open("dataFromEs.csv", 'r') as f:  #opens data file
    reader = csv.reader(f)
    for row in reader:
        #if time to quit, stop looping
        if timeToQuit:
            break
        #if not at starting row keep looping
        if not (row_count == (row_max - startTableOne) or rowFound):
            row_count-=1
        else:
            #from now on will not skip and loop to next row
            rowFound = True
            #print some information about your progress so far
            print("\nYour word list size so far is %d"%len(biList))
            print("There are %d rows left in this table"%row_count)
            print("You are on row %d"% (row_max - row_count))
            print(row)
            #query if sentence above is spanglish or if you want to quit
            tempAnswer = input("\nIs the above list of words bilingual Spanish and English. Input lowercase y for yes,"\
               " lowercase q to stop searching\nfor bilingual words, and anything else (such as just pressing enter) for no:")
            row_count-=1
            #make decisions based on input
            if tempAnswer == 'y':
                biList.append(row)
            if tempAnswer == 'q':
                timeToQuit = True

print("\nStarting with table two of UNK labelled tweets.")
#for checking if you want to stop iterating over this table
timeToQuit = False
#for checking if you have iterated to the correct starting row
rowFound = False

#for counting number of rows
with open("dataFromUNK.csv", 'r') as f:  #opens data file
    reader = csv.reader(f)
    row_count = row_max = sum(1 for row in reader)
    
#for iterating over data sentence by sentence and prompting user if the sentence is spanglish or if they want to quit this table.
with open("dataFromUNK.csv", 'r') as f:  #opens data file
    reader = csv.reader(f)
    for row in reader:
        #if time to quit, stop looping
        if timeToQuit:
            break
        #if not at starting row keep looping
        if not (row_count == (row_max - startTableOne) or rowFound):
            row_count-=1
        else:
            #from now on will not skip and loop to next row
            rowFound = True
            #print some information about your progress so far
            print("\nYour word list size so far is %d"%len(biList))
            print("There are %d rows left in this table"%row_count)
            print("You are on row %d"% (row_max - row_count))
            print(row)
            #query if sentence above is spanglish or if you want to quit
            tempAnswer = input("\nIs the above list of words bilingual Spanish and English. Input lowercase y for yes,"\
               " lowercase q to stop searching\nfor bilingual words, and anything else (such as just pressing enter) for no:")
            row_count-=1
            #make decisions based on input
            if tempAnswer == 'y':
                biList.append(row)
            if tempAnswer == 'q':
                timeToQuit = True

    
#Print some information for the user before printing the sentences you found after you press enter.
input("\nYour results below will be written to biList.csv.\nRename the file if you want to make multiple lists.\n"\
    "Take note of what rows you stopped at in each table.\nPress enter to continue to print results and write csv file.")
print()    
for sentence in biList:
    print(sentence)

#output sentences you found to csv named "biList.csv"
#Remember to rename this file in order to avoid overwriting it in the future if it was incomplete but you want to retain the old record
with open("biList.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(biList)