import os

directory = os.fsencode("./")

for file in os.listdir(directory):
    if os.path.getsize(file) ==0:
        os.remove(file)
        continue
    else:
        continue