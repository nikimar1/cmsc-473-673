import io

def main():
	file_name = "mergefile.txt";
	file = open(file_name, "r", encoding="utf-8")
	data = file.readlines()
	for i in data:
		i = i.strip('\n').replace('["',"").replace('"]',"")
		sentence = i.split('","')[1]	
		# delete one record in data "485501831406026752"
		words = sentence.split(" ")
		print(words)

if __name__ == '__main__':
	main()