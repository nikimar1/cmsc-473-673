## Experimental Twitter Data Parse

#### This repository contains an incomplete fetch from twitter api. We used it in order to test if my parsing and twitter fetch code worked

#### The result was that I was able to fetch from twitter using twurl which was installed via ruby gems. 
I had to install ruby, ruby gems, and twurl. I then had to authenticate my commandline with twitter developers api after applying for an account
I was then able to fetch files and output them as multiple txt files. I did this by running: python fetchTwitterUsingTwurl.py

#### After cutting off the fetch prematurely before it finished iterating, I cleaned out empty files
To delete 0 kb files from fetching tweets that no longer existed, I ran: python removeEmptyFiles.py

#### After removing empty files in this folder, I merged all the smaller txt files to one using a commandline argument via a python program
I ran 'python merge.py' without quotes but you can also just run 'copy *.txt mergefile.txt' in the command line without quotes

Note that I had also manually added some text files with different formatting by copy pasting one or two bilingual tweets I found online
along with their url without using twitter api. However, I deleted these entries from the merged file after creating it.

#### We are now figuring out how to parse the data into sentences and tokens for natural language processing 