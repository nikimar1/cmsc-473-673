import pandas as pd
import io
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Creating a dataframe whose colmuns are label and sentence
def build_label_sent(lang, no_of_file,index_start):

	datas = list()
	data_dict = dict()

	for i in range(1,no_of_file+1):
		file_name = lang+"/"+lang + ' (' + str(i) + ').txt'
		try:
			file = open(file_name, "r", encoding="utf-8")
			datas += file.readlines()
		finally:
			file.close()

	for i,data in enumerate(datas):
		sent_dict = dict()
		sent_dict["LABEL"] = lang
		sent_dict["SENTENCE"] = data.strip("\n").split("\t")[1]
		#print(sent_dict["SENTENCE"])
		data_dict[i+index_start] = sent_dict

	return pd.DataFrame(data_dict).transpose() 

#Choosing the label based on threshold value
def get_label(sentences, threshold):
	
	count_label = dict()

	for sentence in sentences:
		words = sentence.split(" ")
		for word in words:
			if word not in count_label.keys():
				count_label[word] = 1
			else:
				count_label[word] += 1

	labels = list()

	for label, count in count_label.items():
		if count > threshold:
			labels.append(label)
	labels.append("<UNK>")
	return labels

#Generating a dataset
def get_dataset(sentences , columns):

	no_of_sentences = sentences.index[-1] + 1
	#print(sentences.loc[0,"LABEL"]) 
	dataset_dict = dict()
	for i in range(no_of_sentences):
		temp_dict = dict()
		temp_dict["LABEL"] = sentences.loc[i,"LABEL"]
		for column in columns:
			temp_dict[column] = 0
		words = sentences.loc[i,"SENTENCE"].split(" ")

		for word in words:
			if word in temp_dict.keys():

				temp_dict[word] += 1
			else:
				temp_dict['<UNK>'] += 1
		dataset_dict[i] = temp_dict

	dataset = pd.DataFrame(dataset_dict)

	return dataset.transpose()
	
#Main function
def main():
	files_in_english = 1	#Number of files in English folder		range: 1 to 10
	files_in_german = 1		#Number of files in German folder		range: 1 to 21
	files_in_french = 1		#Number of files in French folder		range: 1 to 4
	files_in_italian = 1	#Number of files in Italian folder		range: 1 to 6

	english = build_label_sent("English", files_in_english, 0)
	german = build_label_sent("German", files_in_german, english.index[-1]+1)
	french = build_label_sent("French", files_in_french, german.index[-1]+1)
	italian = build_label_sent("Italian", files_in_italian, french.index[-1]+1)
	frames = pd.concat([english,german,french,italian])
	
	threshold = 20			#Threshold value
	labels = get_label(frames["SENTENCE"], threshold)
	dataset = get_dataset(frames , labels)

	X = dataset.drop("LABEL", axis=1)
	Y = dataset["LABEL"]
	x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


	#Laplace model
	laplace = 1.5

	MultiNB = MultinomialNB(alpha=laplace)
	MultiNB.fit(x_train,y_train)

	y_pred = MultiNB.predict(x_test)
	ans = accuracy_score(y_test,y_pred)
	print("Laplace:",laplace,"\tAccuracy Score:",ans)
	print(classification_report(y_test,y_pred))

	#MaxEnt model
	c2 = 5.1

	LogReg = LogisticRegression(C=c2,penalty='l2')
	LogReg.fit(x_train,y_train)

	y_pred = LogReg.predict(x_test)
	accuracy = accuracy_score(y_test, y_pred)
	print('Accuracy value:',accuracy)
	print(classification_report(y_test,y_pred))

if __name__ == '__main__':
	main()