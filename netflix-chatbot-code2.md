
# Netflix Chatbot: Building & Running Chatbot

## Stephanie Fissel

# PART 3: Running the bot to answer questions

## Import Packages


```python
import pymongo
from pymongo import MongoClient
```


```python
# pip install tensorflow
```


```python
import nltk 
import discord
nltk.download('punkt')
```

    [nltk_data] Downloading package punkt to
    [nltk_data]     /Users/stephaniefissel/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!





    True




```python
from nltk import word_tokenize,sent_tokenize
```


```python
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np 
import tflearn
import tensorflow as tf
import random
import json
import pickle
```

    WARNING:tensorflow:From /Users/stephaniefissel/anaconda3/lib/python3.7/site-packages/tensorflow/python/compat/v2_compat.py:111: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
    Instructions for updating:
    non-resource variables are not supported in the long term


### - Connect to db and read in intent collection
### - Run chatbot with MongoDB queries to answer questions matched to patterns ini intents.json


```python
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["Netflix"]
mycol = mydb["collection"]

with open("/Users/stephaniefissel/Desktop/ds2002/intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle","rb") as f:
        words, labels, training, output = pickle.load(f)

except:
    words = []
    labels = []
    docs_x = []
    docs_y = []
    for intent in ai_intents.find():
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
            
        if intent["tag"] not in labels:
            labels.append(intent["tag"])


    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
               bag.append(1)
            else:
              bag.append(0)
    
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        
        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)
    
    with open("data.pickle","wb") as f:
        pickle.dump((words, labels, training, output), f)



net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    
    return np.array(bag)



def chat():
    print("Start talking with the bot! (type quit to stop)")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        result = model.predict([bag_of_words(inp, words)])[0]
        result_index = np.argmax(result)
        tag = labels[result_index]

        if result[result_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == question1:
                    query1 = mycol.find({"MAIN_GENRE": "romance", "MOVIE_DUMMY": "1"}).sort("SCORE",-1).limit(3)
            print("The top three rated romance movies on Netflix are:")
            for x in query1:
                print(x)
            
        if result[result_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == question2:
                    query2 = mycol.find({"MOVIE_DUMMY": "1"}).sort("DURATION", -1).limit(1)
            print("The longest top movie on Netflix is:")
            for x in query2:
                print(x)
        
        if result[result_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == question3:
                    query3 = mycol.find({"RELEASE_YEAR": "2001"}).sort("SCORE",-1).limit(1)
            print("The genre of the top movie in 2001 is:")
            for x in query3:
                print(x)
            
        if result[result_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == question4:
                    query4 = mycol.find({"MOVIE_DUMMY":"1", "MAIN_GENRE": "drama", "SCORE":{"$gte":"8"}})
            print("These were the drama movies with a score greater than or equal to 8:")
            for x in query4:
                print(x)
            
        if result[result_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == question5:
                    query5 = mycol.find({"MOVIE_DUMMY":"1", "MAIN_PRODUCTION": "GB"})
            print("These were the movies that were mainly produced in GB:")
            for x in query5:
                print(x)
            
        if result[result_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == question6:
                    query6 = mycol.find({"SHOW_DUMMY":"1", "RELEASE_YEAR": {"$lt":"2000"}})
            print("These were the top shows on Netflix produced before 2000:")
            for x in query6:
                print(x)
            
        if result[result_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == question7:
                    query7 = mycol.find({"SHOW_DUMMY":"1", "RELEASE_YEAR": "2017"}).sort("SCORE", -1).limit(1)
            print("This was the top show on Netflix 5 years ago:")
            for x in query7:
                print(x)
        
        if result[result_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == question8:
                    query8 = mycol.find({"SHOW_DUMMY":"1"}).sort("NUMBER_OF_SEASONS", -1).limit(1)
            print("This was the top show on Netflix with the most seasons:")
            for x in query8:
                print(x)
                
        if result[result_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == question9:
                    query9 = mycol.find({"SHOW_DUMMY":"1"}).sort("NUMBER_OF_VOTES", -1).limit(1)
            print("This was the show that received the most votes:")
            for x in query9:
                print(x)
                
        if result[result_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == question10:
                    query10 = mycol.find({"MOVIE_DUMMY":"1"}).sort("SCORE", -1).limit(1)
            print("This is the top movie on Netflix with the highest score:")
            for x in query10:
                print(x)

        else:
            print("I didnt get that. Try again with one of these 10 questions: What were the top three rated romance movies on Netflix of all time?, What was the longest top movie on Netflix?, What was the genre of the top movie in 2001?, What drama movies have a score greater than or equal to 8?, What movies were mainly produced in GB?, What top shows on Netflix were produced before 2000?, What was the top show on Netflix 5 years ago?, What show has the most seasons?, What show has received the most votes? What is the top movie on Netflix with the highest score?")
chat()
```

    WARNING:tensorflow:From /Users/stephaniefissel/anaconda3/lib/python3.7/site-packages/tflearn/initializations.py:165: calling TruncatedNormal.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
    Instructions for updating:
    Call initializer instance with the dtype argument instead of passing it to the constructor


    2022-12-16 22:42:46.814047: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.


    ---------------------------------
    Run id: ZTWYMT
    Log directory: /tmp/tflearn_logs/
    INFO:tensorflow:Summary name Accuracy/ (raw) is illegal; using Accuracy/__raw_ instead.
    ---------------------------------
    Training samples: 1
    Validation samples: 0
    --
    Training Step: 1  | time: 0.065s
    | Adam | epoch: 001 | loss: 0.00000 - acc: 0.0000 -- iter: 1/1
    --
    Training Step: 2  | total loss: [1m[32m3.09064[0m[0m | time: 0.002s
    | Adam | epoch: 002 | loss: 3.09064 - acc: 0.0000 -- iter: 1/1
    --
    Training Step: 3  | total loss: [1m[32m3.36986[0m[0m | time: 0.002s
    | Adam | epoch: 003 | loss: 3.36986 - acc: 0.8182 -- iter: 1/1
    --
    Training Step: 4  | total loss: [1m[32m3.41479[0m[0m | time: 0.002s
    | Adam | epoch: 004 | loss: 3.41479 - acc: 0.9545 -- iter: 1/1
    --
    Training Step: 5  | total loss: [1m[32m3.42365[0m[0m | time: 0.002s
    | Adam | epoch: 005 | loss: 3.42365 - acc: 0.9860 -- iter: 1/1
    --
    Training Step: 6  | total loss: [1m[32m3.42477[0m[0m | time: 0.002s
    | Adam | epoch: 006 | loss: 3.42477 - acc: 0.9950 -- iter: 1/1
    --
    Training Step: 7  | total loss: [1m[32m3.42379[0m[0m | time: 0.002s
    | Adam | epoch: 007 | loss: 3.42379 - acc: 0.9980 -- iter: 1/1
    --
    Training Step: 8  | total loss: [1m[32m3.42212[0m[0m | time: 0.002s
    | Adam | epoch: 008 | loss: 3.42212 - acc: 0.9991 -- iter: 1/1
    --
    Training Step: 9  | total loss: [1m[32m3.42018[0m[0m | time: 0.002s
    | Adam | epoch: 009 | loss: 3.42018 - acc: 0.9996 -- iter: 1/1
    --
    Training Step: 10  | total loss: [1m[32m3.41811[0m[0m | time: 0.002s
    | Adam | epoch: 010 | loss: 3.41811 - acc: 0.9998 -- iter: 1/1
    --
    Training Step: 11  | total loss: [1m[32m3.41594[0m[0m | time: 0.002s
    | Adam | epoch: 011 | loss: 3.41594 - acc: 0.9999 -- iter: 1/1
    --
    Training Step: 12  | total loss: [1m[32m3.41371[0m[0m | time: 0.002s
    | Adam | epoch: 012 | loss: 3.41371 - acc: 0.9999 -- iter: 1/1
    --
    Training Step: 13  | total loss: [1m[32m3.41140[0m[0m | time: 0.002s
    | Adam | epoch: 013 | loss: 3.41140 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 14  | total loss: [1m[32m3.40903[0m[0m | time: 0.002s
    | Adam | epoch: 014 | loss: 3.40903 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 15  | total loss: [1m[32m3.40658[0m[0m | time: 0.002s
    | Adam | epoch: 015 | loss: 3.40658 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 16  | total loss: [1m[32m3.40405[0m[0m | time: 0.002s
    | Adam | epoch: 016 | loss: 3.40405 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 17  | total loss: [1m[32m3.40144[0m[0m | time: 0.002s
    | Adam | epoch: 017 | loss: 3.40144 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 18  | total loss: [1m[32m3.39874[0m[0m | time: 0.002s
    | Adam | epoch: 018 | loss: 3.39874 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 19  | total loss: [1m[32m3.39594[0m[0m | time: 0.002s
    | Adam | epoch: 019 | loss: 3.39594 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 20  | total loss: [1m[32m3.39305[0m[0m | time: 0.002s
    | Adam | epoch: 020 | loss: 3.39305 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 21  | total loss: [1m[32m3.39005[0m[0m | time: 0.002s
    | Adam | epoch: 021 | loss: 3.39005 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 22  | total loss: [1m[32m3.38693[0m[0m | time: 0.002s
    | Adam | epoch: 022 | loss: 3.38693 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 23  | total loss: [1m[32m3.38369[0m[0m | time: 0.002s
    | Adam | epoch: 023 | loss: 3.38369 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 24  | total loss: [1m[32m3.38032[0m[0m | time: 0.002s
    | Adam | epoch: 024 | loss: 3.38032 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 25  | total loss: [1m[32m3.37681[0m[0m | time: 0.002s
    | Adam | epoch: 025 | loss: 3.37681 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 26  | total loss: [1m[32m3.37316[0m[0m | time: 0.002s
    | Adam | epoch: 026 | loss: 3.37316 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 27  | total loss: [1m[32m3.36936[0m[0m | time: 0.002s
    | Adam | epoch: 027 | loss: 3.36936 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 28  | total loss: [1m[32m3.36539[0m[0m | time: 0.002s
    | Adam | epoch: 028 | loss: 3.36539 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 29  | total loss: [1m[32m3.36124[0m[0m | time: 0.002s
    | Adam | epoch: 029 | loss: 3.36124 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 30  | total loss: [1m[32m3.35691[0m[0m | time: 0.002s
    | Adam | epoch: 030 | loss: 3.35691 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 31  | total loss: [1m[32m3.35239[0m[0m | time: 0.002s
    | Adam | epoch: 031 | loss: 3.35239 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 32  | total loss: [1m[32m3.34766[0m[0m | time: 0.002s
    | Adam | epoch: 032 | loss: 3.34766 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 33  | total loss: [1m[32m3.34271[0m[0m | time: 0.002s
    | Adam | epoch: 033 | loss: 3.34271 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 34  | total loss: [1m[32m3.33753[0m[0m | time: 0.002s
    | Adam | epoch: 034 | loss: 3.33753 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 35  | total loss: [1m[32m3.33211[0m[0m | time: 0.002s
    | Adam | epoch: 035 | loss: 3.33211 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 36  | total loss: [1m[32m3.32644[0m[0m | time: 0.002s
    | Adam | epoch: 036 | loss: 3.32644 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 37  | total loss: [1m[32m3.32050[0m[0m | time: 0.002s
    | Adam | epoch: 037 | loss: 3.32050 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 38  | total loss: [1m[32m3.31429[0m[0m | time: 0.002s
    | Adam | epoch: 038 | loss: 3.31429 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 39  | total loss: [1m[32m3.30778[0m[0m | time: 0.002s
    | Adam | epoch: 039 | loss: 3.30778 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 40  | total loss: [1m[32m3.30097[0m[0m | time: 0.002s
    | Adam | epoch: 040 | loss: 3.30097 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 41  | total loss: [1m[32m3.29384[0m[0m | time: 0.002s
    | Adam | epoch: 041 | loss: 3.29384 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 42  | total loss: [1m[32m3.28638[0m[0m | time: 0.002s
    | Adam | epoch: 042 | loss: 3.28638 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 43  | total loss: [1m[32m3.27857[0m[0m | time: 0.002s
    | Adam | epoch: 043 | loss: 3.27857 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 44  | total loss: [1m[32m3.27040[0m[0m | time: 0.002s
    | Adam | epoch: 044 | loss: 3.27040 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 45  | total loss: [1m[32m3.26185[0m[0m | time: 0.002s
    | Adam | epoch: 045 | loss: 3.26185 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 46  | total loss: [1m[32m3.25292[0m[0m | time: 0.002s
    | Adam | epoch: 046 | loss: 3.25292 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 47  | total loss: [1m[32m3.24357[0m[0m | time: 0.002s
    | Adam | epoch: 047 | loss: 3.24357 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 48  | total loss: [1m[32m3.23380[0m[0m | time: 0.002s
    | Adam | epoch: 048 | loss: 3.23380 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 49  | total loss: [1m[32m3.22360[0m[0m | time: 0.002s
    | Adam | epoch: 049 | loss: 3.22360 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 50  | total loss: [1m[32m3.21294[0m[0m | time: 0.002s
    | Adam | epoch: 050 | loss: 3.21294 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 51  | total loss: [1m[32m3.20181[0m[0m | time: 0.002s
    | Adam | epoch: 051 | loss: 3.20181 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 52  | total loss: [1m[32m3.19019[0m[0m | time: 0.002s
    | Adam | epoch: 052 | loss: 3.19019 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 53  | total loss: [1m[32m3.17807[0m[0m | time: 0.002s
    | Adam | epoch: 053 | loss: 3.17807 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 54  | total loss: [1m[32m3.16542[0m[0m | time: 0.002s
    | Adam | epoch: 054 | loss: 3.16542 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 55  | total loss: [1m[32m3.15224[0m[0m | time: 0.002s
    | Adam | epoch: 055 | loss: 3.15224 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 56  | total loss: [1m[32m3.13851[0m[0m | time: 0.002s
    | Adam | epoch: 056 | loss: 3.13851 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 57  | total loss: [1m[32m3.12421[0m[0m | time: 0.002s
    | Adam | epoch: 057 | loss: 3.12421 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 58  | total loss: [1m[32m3.10931[0m[0m | time: 0.002s
    | Adam | epoch: 058 | loss: 3.10931 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 59  | total loss: [1m[32m3.09381[0m[0m | time: 0.002s
    | Adam | epoch: 059 | loss: 3.09381 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 60  | total loss: [1m[32m3.07769[0m[0m | time: 0.002s
    | Adam | epoch: 060 | loss: 3.07769 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 61  | total loss: [1m[32m3.06093[0m[0m | time: 0.002s
    | Adam | epoch: 061 | loss: 3.06093 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 62  | total loss: [1m[32m3.04350[0m[0m | time: 0.003s
    | Adam | epoch: 062 | loss: 3.04350 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 63  | total loss: [1m[32m3.02541[0m[0m | time: 0.002s
    | Adam | epoch: 063 | loss: 3.02541 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 64  | total loss: [1m[32m3.00662[0m[0m | time: 0.002s
    | Adam | epoch: 064 | loss: 3.00662 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 65  | total loss: [1m[32m2.98712[0m[0m | time: 0.002s
    | Adam | epoch: 065 | loss: 2.98712 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 66  | total loss: [1m[32m2.96690[0m[0m | time: 0.002s
    | Adam | epoch: 066 | loss: 2.96690 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 67  | total loss: [1m[32m2.94594[0m[0m | time: 0.002s
    | Adam | epoch: 067 | loss: 2.94594 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 68  | total loss: [1m[32m2.92421[0m[0m | time: 0.002s
    | Adam | epoch: 068 | loss: 2.92421 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 69  | total loss: [1m[32m2.90171[0m[0m | time: 0.002s
    | Adam | epoch: 069 | loss: 2.90171 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 70  | total loss: [1m[32m2.87842[0m[0m | time: 0.002s
    | Adam | epoch: 070 | loss: 2.87842 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 71  | total loss: [1m[32m2.85433[0m[0m | time: 0.002s
    | Adam | epoch: 071 | loss: 2.85433 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 72  | total loss: [1m[32m2.82942[0m[0m | time: 0.002s
    | Adam | epoch: 072 | loss: 2.82942 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 73  | total loss: [1m[32m2.80367[0m[0m | time: 0.002s
    | Adam | epoch: 073 | loss: 2.80367 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 74  | total loss: [1m[32m2.77707[0m[0m | time: 0.002s
    | Adam | epoch: 074 | loss: 2.77707 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 75  | total loss: [1m[32m2.74961[0m[0m | time: 0.002s
    | Adam | epoch: 075 | loss: 2.74961 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 76  | total loss: [1m[32m2.72128[0m[0m | time: 0.002s
    | Adam | epoch: 076 | loss: 2.72128 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 77  | total loss: [1m[32m2.69206[0m[0m | time: 0.002s
    | Adam | epoch: 077 | loss: 2.69206 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 78  | total loss: [1m[32m2.66195[0m[0m | time: 0.002s
    | Adam | epoch: 078 | loss: 2.66195 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 79  | total loss: [1m[32m2.63093[0m[0m | time: 0.002s
    | Adam | epoch: 079 | loss: 2.63093 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 80  | total loss: [1m[32m2.59901[0m[0m | time: 0.002s
    | Adam | epoch: 080 | loss: 2.59901 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 81  | total loss: [1m[32m2.56616[0m[0m | time: 0.003s
    | Adam | epoch: 081 | loss: 2.56616 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 82  | total loss: [1m[32m2.53240[0m[0m | time: 0.002s
    | Adam | epoch: 082 | loss: 2.53240 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 83  | total loss: [1m[32m2.49732[0m[0m | time: 0.002s
    | Adam | epoch: 083 | loss: 2.49732 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 84  | total loss: [1m[32m2.46095[0m[0m | time: 0.022s
    | Adam | epoch: 084 | loss: 2.46095 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 85  | total loss: [1m[32m2.42330[0m[0m | time: 0.019s
    | Adam | epoch: 085 | loss: 2.42330 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 86  | total loss: [1m[32m2.38440[0m[0m | time: 0.002s
    | Adam | epoch: 086 | loss: 2.38440 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 87  | total loss: [1m[32m2.34425[0m[0m | time: 0.002s
    | Adam | epoch: 087 | loss: 2.34425 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 88  | total loss: [1m[32m2.30289[0m[0m | time: 0.006s
    | Adam | epoch: 088 | loss: 2.30289 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 89  | total loss: [1m[32m2.26033[0m[0m | time: 0.006s
    | Adam | epoch: 089 | loss: 2.26033 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 90  | total loss: [1m[32m2.21661[0m[0m | time: 0.002s
    | Adam | epoch: 090 | loss: 2.21661 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 91  | total loss: [1m[32m2.17175[0m[0m | time: 0.002s
    | Adam | epoch: 091 | loss: 2.17175 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 92  | total loss: [1m[32m2.12578[0m[0m | time: 0.002s
    | Adam | epoch: 092 | loss: 2.12578 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 93  | total loss: [1m[32m2.07876[0m[0m | time: 0.002s
    | Adam | epoch: 093 | loss: 2.07876 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 94  | total loss: [1m[32m2.03070[0m[0m | time: 0.002s
    | Adam | epoch: 094 | loss: 2.03070 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 95  | total loss: [1m[32m1.98168[0m[0m | time: 0.002s
    | Adam | epoch: 095 | loss: 1.98168 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 96  | total loss: [1m[32m1.93174[0m[0m | time: 0.002s
    | Adam | epoch: 096 | loss: 1.93174 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 97  | total loss: [1m[32m1.88094[0m[0m | time: 0.002s
    | Adam | epoch: 097 | loss: 1.88094 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 98  | total loss: [1m[32m1.82935[0m[0m | time: 0.002s
    | Adam | epoch: 098 | loss: 1.82935 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 99  | total loss: [1m[32m1.77704[0m[0m | time: 0.002s
    | Adam | epoch: 099 | loss: 1.77704 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 100  | total loss: [1m[32m1.72410[0m[0m | time: 0.002s
    | Adam | epoch: 100 | loss: 1.72410 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 101  | total loss: [1m[32m1.67062[0m[0m | time: 0.002s
    | Adam | epoch: 101 | loss: 1.67062 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 102  | total loss: [1m[32m1.61669[0m[0m | time: 0.002s
    | Adam | epoch: 102 | loss: 1.61669 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 103  | total loss: [1m[32m1.56242[0m[0m | time: 0.002s
    | Adam | epoch: 103 | loss: 1.56242 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 104  | total loss: [1m[32m1.50792[0m[0m | time: 0.002s
    | Adam | epoch: 104 | loss: 1.50792 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 105  | total loss: [1m[32m1.45330[0m[0m | time: 0.002s
    | Adam | epoch: 105 | loss: 1.45330 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 106  | total loss: [1m[32m1.39870[0m[0m | time: 0.003s
    | Adam | epoch: 106 | loss: 1.39870 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 107  | total loss: [1m[32m1.34424[0m[0m | time: 0.002s
    | Adam | epoch: 107 | loss: 1.34424 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 108  | total loss: [1m[32m1.29006[0m[0m | time: 0.002s
    | Adam | epoch: 108 | loss: 1.29006 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 109  | total loss: [1m[32m1.23628[0m[0m | time: 0.002s
    | Adam | epoch: 109 | loss: 1.23628 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 110  | total loss: [1m[32m1.18305[0m[0m | time: 0.002s
    | Adam | epoch: 110 | loss: 1.18305 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 111  | total loss: [1m[32m1.13050[0m[0m | time: 0.002s
    | Adam | epoch: 111 | loss: 1.13050 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 112  | total loss: [1m[32m1.07877[0m[0m | time: 0.002s
    | Adam | epoch: 112 | loss: 1.07877 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 113  | total loss: [1m[32m1.02797[0m[0m | time: 0.002s
    | Adam | epoch: 113 | loss: 1.02797 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 114  | total loss: [1m[32m0.97824[0m[0m | time: 0.002s
    | Adam | epoch: 114 | loss: 0.97824 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 115  | total loss: [1m[32m0.92970[0m[0m | time: 0.002s
    | Adam | epoch: 115 | loss: 0.92970 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 116  | total loss: [1m[32m0.88243[0m[0m | time: 0.002s
    | Adam | epoch: 116 | loss: 0.88243 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 117  | total loss: [1m[32m0.83654[0m[0m | time: 0.002s
    | Adam | epoch: 117 | loss: 0.83654 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 118  | total loss: [1m[32m0.79212[0m[0m | time: 0.002s
    | Adam | epoch: 118 | loss: 0.79212 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 119  | total loss: [1m[32m0.74922[0m[0m | time: 0.002s
    | Adam | epoch: 119 | loss: 0.74922 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 120  | total loss: [1m[32m0.70791[0m[0m | time: 0.002s
    | Adam | epoch: 120 | loss: 0.70791 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 121  | total loss: [1m[32m0.66823[0m[0m | time: 0.002s
    | Adam | epoch: 121 | loss: 0.66823 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 122  | total loss: [1m[32m0.63021[0m[0m | time: 0.002s
    | Adam | epoch: 122 | loss: 0.63021 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 123  | total loss: [1m[32m0.59386[0m[0m | time: 0.002s
    | Adam | epoch: 123 | loss: 0.59386 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 124  | total loss: [1m[32m0.55919[0m[0m | time: 0.003s
    | Adam | epoch: 124 | loss: 0.55919 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 125  | total loss: [1m[32m0.52620[0m[0m | time: 0.002s
    | Adam | epoch: 125 | loss: 0.52620 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 126  | total loss: [1m[32m0.49486[0m[0m | time: 0.002s
    | Adam | epoch: 126 | loss: 0.49486 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 127  | total loss: [1m[32m0.46515[0m[0m | time: 0.002s
    | Adam | epoch: 127 | loss: 0.46515 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 128  | total loss: [1m[32m0.43704[0m[0m | time: 0.002s
    | Adam | epoch: 128 | loss: 0.43704 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 129  | total loss: [1m[32m0.41049[0m[0m | time: 0.002s
    | Adam | epoch: 129 | loss: 0.41049 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 130  | total loss: [1m[32m0.38544[0m[0m | time: 0.002s
    | Adam | epoch: 130 | loss: 0.38544 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 131  | total loss: [1m[32m0.36185[0m[0m | time: 0.002s
    | Adam | epoch: 131 | loss: 0.36185 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 132  | total loss: [1m[32m0.33967[0m[0m | time: 0.002s
    | Adam | epoch: 132 | loss: 0.33967 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 133  | total loss: [1m[32m0.31883[0m[0m | time: 0.002s
    | Adam | epoch: 133 | loss: 0.31883 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 134  | total loss: [1m[32m0.29928[0m[0m | time: 0.002s
    | Adam | epoch: 134 | loss: 0.29928 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 135  | total loss: [1m[32m0.28095[0m[0m | time: 0.001s
    | Adam | epoch: 135 | loss: 0.28095 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 136  | total loss: [1m[32m0.26379[0m[0m | time: 0.002s
    | Adam | epoch: 136 | loss: 0.26379 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 137  | total loss: [1m[32m0.24773[0m[0m | time: 0.002s
    | Adam | epoch: 137 | loss: 0.24773 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 138  | total loss: [1m[32m0.23272[0m[0m | time: 0.003s
    | Adam | epoch: 138 | loss: 0.23272 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 139  | total loss: [1m[32m0.21870[0m[0m | time: 0.002s
    | Adam | epoch: 139 | loss: 0.21870 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 140  | total loss: [1m[32m0.20561[0m[0m | time: 0.002s
    | Adam | epoch: 140 | loss: 0.20561 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 141  | total loss: [1m[32m0.19339[0m[0m | time: 0.002s
    | Adam | epoch: 141 | loss: 0.19339 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 142  | total loss: [1m[32m0.18199[0m[0m | time: 0.002s
    | Adam | epoch: 142 | loss: 0.18199 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 143  | total loss: [1m[32m0.17137[0m[0m | time: 0.002s
    | Adam | epoch: 143 | loss: 0.17137 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 144  | total loss: [1m[32m0.16146[0m[0m | time: 0.002s
    | Adam | epoch: 144 | loss: 0.16146 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 145  | total loss: [1m[32m0.15223[0m[0m | time: 0.003s
    | Adam | epoch: 145 | loss: 0.15223 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 146  | total loss: [1m[32m0.14364[0m[0m | time: 0.002s
    | Adam | epoch: 146 | loss: 0.14364 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 147  | total loss: [1m[32m0.13563[0m[0m | time: 0.002s
    | Adam | epoch: 147 | loss: 0.13563 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 148  | total loss: [1m[32m0.12817[0m[0m | time: 0.002s
    | Adam | epoch: 148 | loss: 0.12817 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 149  | total loss: [1m[32m0.12122[0m[0m | time: 0.002s
    | Adam | epoch: 149 | loss: 0.12122 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 150  | total loss: [1m[32m0.11475[0m[0m | time: 0.002s
    | Adam | epoch: 150 | loss: 0.11475 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 151  | total loss: [1m[32m0.10872[0m[0m | time: 0.002s
    | Adam | epoch: 151 | loss: 0.10872 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 152  | total loss: [1m[32m0.10311[0m[0m | time: 0.003s
    | Adam | epoch: 152 | loss: 0.10311 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 153  | total loss: [1m[32m0.09788[0m[0m | time: 0.002s
    | Adam | epoch: 153 | loss: 0.09788 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 154  | total loss: [1m[32m0.09300[0m[0m | time: 0.002s
    | Adam | epoch: 154 | loss: 0.09300 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 155  | total loss: [1m[32m0.08846[0m[0m | time: 0.002s
    | Adam | epoch: 155 | loss: 0.08846 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 156  | total loss: [1m[32m0.08422[0m[0m | time: 0.002s
    | Adam | epoch: 156 | loss: 0.08422 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 157  | total loss: [1m[32m0.08027[0m[0m | time: 0.002s
    | Adam | epoch: 157 | loss: 0.08027 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 158  | total loss: [1m[32m0.07658[0m[0m | time: 0.003s
    | Adam | epoch: 158 | loss: 0.07658 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 159  | total loss: [1m[32m0.07314[0m[0m | time: 0.002s
    | Adam | epoch: 159 | loss: 0.07314 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 160  | total loss: [1m[32m0.06992[0m[0m | time: 0.002s
    | Adam | epoch: 160 | loss: 0.06992 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 161  | total loss: [1m[32m0.06692[0m[0m | time: 0.002s
    | Adam | epoch: 161 | loss: 0.06692 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 162  | total loss: [1m[32m0.06411[0m[0m | time: 0.002s
    | Adam | epoch: 162 | loss: 0.06411 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 163  | total loss: [1m[32m0.06149[0m[0m | time: 0.002s
    | Adam | epoch: 163 | loss: 0.06149 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 164  | total loss: [1m[32m0.05903[0m[0m | time: 0.003s
    | Adam | epoch: 164 | loss: 0.05903 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 165  | total loss: [1m[32m0.05673[0m[0m | time: 0.002s
    | Adam | epoch: 165 | loss: 0.05673 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 166  | total loss: [1m[32m0.05457[0m[0m | time: 0.002s
    | Adam | epoch: 166 | loss: 0.05457 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 167  | total loss: [1m[32m0.05255[0m[0m | time: 0.002s
    | Adam | epoch: 167 | loss: 0.05255 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 168  | total loss: [1m[32m0.05065[0m[0m | time: 0.002s
    | Adam | epoch: 168 | loss: 0.05065 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 169  | total loss: [1m[32m0.04886[0m[0m | time: 0.002s
    | Adam | epoch: 169 | loss: 0.04886 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 170  | total loss: [1m[32m0.04719[0m[0m | time: 0.003s
    | Adam | epoch: 170 | loss: 0.04719 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 171  | total loss: [1m[32m0.04561[0m[0m | time: 0.002s
    | Adam | epoch: 171 | loss: 0.04561 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 172  | total loss: [1m[32m0.04413[0m[0m | time: 0.002s
    | Adam | epoch: 172 | loss: 0.04413 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 173  | total loss: [1m[32m0.04273[0m[0m | time: 0.002s
    | Adam | epoch: 173 | loss: 0.04273 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 174  | total loss: [1m[32m0.04141[0m[0m | time: 0.002s
    | Adam | epoch: 174 | loss: 0.04141 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 175  | total loss: [1m[32m0.04016[0m[0m | time: 0.002s
    | Adam | epoch: 175 | loss: 0.04016 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 176  | total loss: [1m[32m0.03899[0m[0m | time: 0.002s
    | Adam | epoch: 176 | loss: 0.03899 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 177  | total loss: [1m[32m0.03787[0m[0m | time: 0.002s
    | Adam | epoch: 177 | loss: 0.03787 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 178  | total loss: [1m[32m0.03682[0m[0m | time: 0.002s
    | Adam | epoch: 178 | loss: 0.03682 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 179  | total loss: [1m[32m0.03582[0m[0m | time: 0.002s
    | Adam | epoch: 179 | loss: 0.03582 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 180  | total loss: [1m[32m0.03488[0m[0m | time: 0.002s
    | Adam | epoch: 180 | loss: 0.03488 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 181  | total loss: [1m[32m0.03398[0m[0m | time: 0.002s
    | Adam | epoch: 181 | loss: 0.03398 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 182  | total loss: [1m[32m0.03313[0m[0m | time: 0.002s
    | Adam | epoch: 182 | loss: 0.03313 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 183  | total loss: [1m[32m0.03232[0m[0m | time: 0.002s
    | Adam | epoch: 183 | loss: 0.03232 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 184  | total loss: [1m[32m0.03154[0m[0m | time: 0.002s
    | Adam | epoch: 184 | loss: 0.03154 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 185  | total loss: [1m[32m0.03081[0m[0m | time: 0.002s
    | Adam | epoch: 185 | loss: 0.03081 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 186  | total loss: [1m[32m0.03011[0m[0m | time: 0.002s
    | Adam | epoch: 186 | loss: 0.03011 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 187  | total loss: [1m[32m0.02944[0m[0m | time: 0.001s
    | Adam | epoch: 187 | loss: 0.02944 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 188  | total loss: [1m[32m0.02818[0m[0m | time: 0.037s
    | Adam | epoch: 188 | loss: 0.02818 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 189  | total loss: [1m[32m0.02818[0m[0m | time: 0.002s
    | Adam | epoch: 189 | loss: 0.02818 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 190  | total loss: [1m[32m0.02760[0m[0m | time: 0.002s
    | Adam | epoch: 190 | loss: 0.02760 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 191  | total loss: [1m[32m0.02704[0m[0m | time: 0.003s
    | Adam | epoch: 191 | loss: 0.02704 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 192  | total loss: [1m[32m0.02650[0m[0m | time: 0.007s
    | Adam | epoch: 192 | loss: 0.02650 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 193  | total loss: [1m[32m0.02598[0m[0m | time: 0.003s
    | Adam | epoch: 193 | loss: 0.02598 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 194  | total loss: [1m[32m0.02548[0m[0m | time: 0.002s
    | Adam | epoch: 194 | loss: 0.02548 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 195  | total loss: [1m[32m0.02501[0m[0m | time: 0.002s
    | Adam | epoch: 195 | loss: 0.02501 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 196  | total loss: [1m[32m0.02455[0m[0m | time: 0.002s
    | Adam | epoch: 196 | loss: 0.02455 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 197  | total loss: [1m[32m0.02410[0m[0m | time: 0.002s
    | Adam | epoch: 197 | loss: 0.02410 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 198  | total loss: [1m[32m0.02368[0m[0m | time: 0.002s
    | Adam | epoch: 198 | loss: 0.02368 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 199  | total loss: [1m[32m0.02327[0m[0m | time: 0.002s
    | Adam | epoch: 199 | loss: 0.02327 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 200  | total loss: [1m[32m0.02287[0m[0m | time: 0.002s
    | Adam | epoch: 200 | loss: 0.02287 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 201  | total loss: [1m[32m0.02249[0m[0m | time: 0.002s
    | Adam | epoch: 201 | loss: 0.02249 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 202  | total loss: [1m[32m0.02211[0m[0m | time: 0.003s
    | Adam | epoch: 202 | loss: 0.02211 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 203  | total loss: [1m[32m0.02176[0m[0m | time: 0.002s
    | Adam | epoch: 203 | loss: 0.02176 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 204  | total loss: [1m[32m0.02141[0m[0m | time: 0.002s
    | Adam | epoch: 204 | loss: 0.02141 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 205  | total loss: [1m[32m0.02107[0m[0m | time: 0.002s
    | Adam | epoch: 205 | loss: 0.02107 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 206  | total loss: [1m[32m0.02074[0m[0m | time: 0.002s
    | Adam | epoch: 206 | loss: 0.02074 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 207  | total loss: [1m[32m0.02043[0m[0m | time: 0.002s
    | Adam | epoch: 207 | loss: 0.02043 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 208  | total loss: [1m[32m0.02012[0m[0m | time: 0.002s
    | Adam | epoch: 208 | loss: 0.02012 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 209  | total loss: [1m[32m0.01982[0m[0m | time: 0.002s
    | Adam | epoch: 209 | loss: 0.01982 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 210  | total loss: [1m[32m0.01953[0m[0m | time: 0.002s
    | Adam | epoch: 210 | loss: 0.01953 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 211  | total loss: [1m[32m0.01925[0m[0m | time: 0.002s
    | Adam | epoch: 211 | loss: 0.01925 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 212  | total loss: [1m[32m0.01898[0m[0m | time: 0.002s
    | Adam | epoch: 212 | loss: 0.01898 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 213  | total loss: [1m[32m0.01845[0m[0m | time: 0.011s
    | Adam | epoch: 213 | loss: 0.01845 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 214  | total loss: [1m[32m0.01820[0m[0m | time: 0.009s
    | Adam | epoch: 214 | loss: 0.01820 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 215  | total loss: [1m[32m0.01820[0m[0m | time: 0.002s
    | Adam | epoch: 215 | loss: 0.01820 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 216  | total loss: [1m[32m0.01795[0m[0m | time: 0.002s
    | Adam | epoch: 216 | loss: 0.01795 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 217  | total loss: [1m[32m0.01771[0m[0m | time: 0.002s
    | Adam | epoch: 217 | loss: 0.01771 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 218  | total loss: [1m[32m0.01748[0m[0m | time: 0.002s
    | Adam | epoch: 218 | loss: 0.01748 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 219  | total loss: [1m[32m0.01725[0m[0m | time: 0.003s
    | Adam | epoch: 219 | loss: 0.01725 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 220  | total loss: [1m[32m0.01703[0m[0m | time: 0.002s
    | Adam | epoch: 220 | loss: 0.01703 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 221  | total loss: [1m[32m0.01681[0m[0m | time: 0.002s
    | Adam | epoch: 221 | loss: 0.01681 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 222  | total loss: [1m[32m0.01659[0m[0m | time: 0.002s
    | Adam | epoch: 222 | loss: 0.01659 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 223  | total loss: [1m[32m0.01639[0m[0m | time: 0.002s
    | Adam | epoch: 223 | loss: 0.01639 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 224  | total loss: [1m[32m0.01618[0m[0m | time: 0.002s
    | Adam | epoch: 224 | loss: 0.01618 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 225  | total loss: [1m[32m0.01598[0m[0m | time: 0.002s
    | Adam | epoch: 225 | loss: 0.01598 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 226  | total loss: [1m[32m0.01579[0m[0m | time: 0.002s
    | Adam | epoch: 226 | loss: 0.01579 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 227  | total loss: [1m[32m0.01560[0m[0m | time: 0.002s
    | Adam | epoch: 227 | loss: 0.01560 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 228  | total loss: [1m[32m0.01541[0m[0m | time: 0.002s
    | Adam | epoch: 228 | loss: 0.01541 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 229  | total loss: [1m[32m0.01523[0m[0m | time: 0.002s
    | Adam | epoch: 229 | loss: 0.01523 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 230  | total loss: [1m[32m0.01505[0m[0m | time: 0.002s
    | Adam | epoch: 230 | loss: 0.01505 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 231  | total loss: [1m[32m0.01488[0m[0m | time: 0.002s
    | Adam | epoch: 231 | loss: 0.01488 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 232  | total loss: [1m[32m0.01471[0m[0m | time: 0.002s
    | Adam | epoch: 232 | loss: 0.01471 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 233  | total loss: [1m[32m0.01454[0m[0m | time: 0.002s
    | Adam | epoch: 233 | loss: 0.01454 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 234  | total loss: [1m[32m0.01437[0m[0m | time: 0.002s
    | Adam | epoch: 234 | loss: 0.01437 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 235  | total loss: [1m[32m0.01421[0m[0m | time: 0.002s
    | Adam | epoch: 235 | loss: 0.01421 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 236  | total loss: [1m[32m0.01405[0m[0m | time: 0.002s
    | Adam | epoch: 236 | loss: 0.01405 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 237  | total loss: [1m[32m0.01390[0m[0m | time: 0.002s
    | Adam | epoch: 237 | loss: 0.01390 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 238  | total loss: [1m[32m0.01375[0m[0m | time: 0.002s
    | Adam | epoch: 238 | loss: 0.01375 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 239  | total loss: [1m[32m0.01360[0m[0m | time: 0.002s
    | Adam | epoch: 239 | loss: 0.01360 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 240  | total loss: [1m[32m0.01345[0m[0m | time: 0.002s
    | Adam | epoch: 240 | loss: 0.01345 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 241  | total loss: [1m[32m0.01331[0m[0m | time: 0.002s
    | Adam | epoch: 241 | loss: 0.01331 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 242  | total loss: [1m[32m0.01317[0m[0m | time: 0.002s
    | Adam | epoch: 242 | loss: 0.01317 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 243  | total loss: [1m[32m0.01303[0m[0m | time: 0.002s
    | Adam | epoch: 243 | loss: 0.01303 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 244  | total loss: [1m[32m0.01289[0m[0m | time: 0.002s
    | Adam | epoch: 244 | loss: 0.01289 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 245  | total loss: [1m[32m0.01276[0m[0m | time: 0.002s
    | Adam | epoch: 245 | loss: 0.01276 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 246  | total loss: [1m[32m0.01263[0m[0m | time: 0.002s
    | Adam | epoch: 246 | loss: 0.01263 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 247  | total loss: [1m[32m0.01250[0m[0m | time: 0.002s
    | Adam | epoch: 247 | loss: 0.01250 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 248  | total loss: [1m[32m0.01237[0m[0m | time: 0.002s
    | Adam | epoch: 248 | loss: 0.01237 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 249  | total loss: [1m[32m0.01224[0m[0m | time: 0.002s
    | Adam | epoch: 249 | loss: 0.01224 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 250  | total loss: [1m[32m0.01212[0m[0m | time: 0.002s
    | Adam | epoch: 250 | loss: 0.01212 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 251  | total loss: [1m[32m0.01200[0m[0m | time: 0.002s
    | Adam | epoch: 251 | loss: 0.01200 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 252  | total loss: [1m[32m0.01188[0m[0m | time: 0.002s
    | Adam | epoch: 252 | loss: 0.01188 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 253  | total loss: [1m[32m0.01177[0m[0m | time: 0.002s
    | Adam | epoch: 253 | loss: 0.01177 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 254  | total loss: [1m[32m0.01165[0m[0m | time: 0.002s
    | Adam | epoch: 254 | loss: 0.01165 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 255  | total loss: [1m[32m0.01154[0m[0m | time: 0.002s
    | Adam | epoch: 255 | loss: 0.01154 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 256  | total loss: [1m[32m0.01143[0m[0m | time: 0.002s
    | Adam | epoch: 256 | loss: 0.01143 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 257  | total loss: [1m[32m0.01132[0m[0m | time: 0.002s
    | Adam | epoch: 257 | loss: 0.01132 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 258  | total loss: [1m[32m0.01121[0m[0m | time: 0.002s
    | Adam | epoch: 258 | loss: 0.01121 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 259  | total loss: [1m[32m0.01110[0m[0m | time: 0.002s
    | Adam | epoch: 259 | loss: 0.01110 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 260  | total loss: [1m[32m0.01100[0m[0m | time: 0.002s
    | Adam | epoch: 260 | loss: 0.01100 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 261  | total loss: [1m[32m0.01090[0m[0m | time: 0.002s
    | Adam | epoch: 261 | loss: 0.01090 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 262  | total loss: [1m[32m0.01080[0m[0m | time: 0.002s
    | Adam | epoch: 262 | loss: 0.01080 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 263  | total loss: [1m[32m0.01070[0m[0m | time: 0.002s
    | Adam | epoch: 263 | loss: 0.01070 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 264  | total loss: [1m[32m0.01060[0m[0m | time: 0.002s
    | Adam | epoch: 264 | loss: 0.01060 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 265  | total loss: [1m[32m0.01050[0m[0m | time: 0.002s
    | Adam | epoch: 265 | loss: 0.01050 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 266  | total loss: [1m[32m0.01041[0m[0m | time: 0.002s
    | Adam | epoch: 266 | loss: 0.01041 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 267  | total loss: [1m[32m0.01031[0m[0m | time: 0.002s
    | Adam | epoch: 267 | loss: 0.01031 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 268  | total loss: [1m[32m0.01022[0m[0m | time: 0.002s
    | Adam | epoch: 268 | loss: 0.01022 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 269  | total loss: [1m[32m0.01013[0m[0m | time: 0.002s
    | Adam | epoch: 269 | loss: 0.01013 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 270  | total loss: [1m[32m0.01004[0m[0m | time: 0.002s
    | Adam | epoch: 270 | loss: 0.01004 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 271  | total loss: [1m[32m0.00995[0m[0m | time: 0.002s
    | Adam | epoch: 271 | loss: 0.00995 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 272  | total loss: [1m[32m0.00986[0m[0m | time: 0.002s
    | Adam | epoch: 272 | loss: 0.00986 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 273  | total loss: [1m[32m0.00978[0m[0m | time: 0.002s
    | Adam | epoch: 273 | loss: 0.00978 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 274  | total loss: [1m[32m0.00969[0m[0m | time: 0.002s
    | Adam | epoch: 274 | loss: 0.00969 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 275  | total loss: [1m[32m0.00961[0m[0m | time: 0.002s
    | Adam | epoch: 275 | loss: 0.00961 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 276  | total loss: [1m[32m0.00953[0m[0m | time: 0.002s
    | Adam | epoch: 276 | loss: 0.00953 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 277  | total loss: [1m[32m0.00945[0m[0m | time: 0.002s
    | Adam | epoch: 277 | loss: 0.00945 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 278  | total loss: [1m[32m0.00937[0m[0m | time: 0.002s
    | Adam | epoch: 278 | loss: 0.00937 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 279  | total loss: [1m[32m0.00929[0m[0m | time: 0.002s
    | Adam | epoch: 279 | loss: 0.00929 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 280  | total loss: [1m[32m0.00921[0m[0m | time: 0.002s
    | Adam | epoch: 280 | loss: 0.00921 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 281  | total loss: [1m[32m0.00913[0m[0m | time: 0.002s
    | Adam | epoch: 281 | loss: 0.00913 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 282  | total loss: [1m[32m0.00906[0m[0m | time: 0.002s
    | Adam | epoch: 282 | loss: 0.00906 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 283  | total loss: [1m[32m0.00898[0m[0m | time: 0.002s
    | Adam | epoch: 283 | loss: 0.00898 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 284  | total loss: [1m[32m0.00891[0m[0m | time: 0.002s
    | Adam | epoch: 284 | loss: 0.00891 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 285  | total loss: [1m[32m0.00884[0m[0m | time: 0.002s
    | Adam | epoch: 285 | loss: 0.00884 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 286  | total loss: [1m[32m0.00876[0m[0m | time: 0.002s
    | Adam | epoch: 286 | loss: 0.00876 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 287  | total loss: [1m[32m0.00869[0m[0m | time: 0.002s
    | Adam | epoch: 287 | loss: 0.00869 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 288  | total loss: [1m[32m0.00862[0m[0m | time: 0.002s
    | Adam | epoch: 288 | loss: 0.00862 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 289  | total loss: [1m[32m0.00855[0m[0m | time: 0.002s
    | Adam | epoch: 289 | loss: 0.00855 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 290  | total loss: [1m[32m0.00849[0m[0m | time: 0.002s
    | Adam | epoch: 290 | loss: 0.00849 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 291  | total loss: [1m[32m0.00842[0m[0m | time: 0.002s
    | Adam | epoch: 291 | loss: 0.00842 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 292  | total loss: [1m[32m0.00835[0m[0m | time: 0.002s
    | Adam | epoch: 292 | loss: 0.00835 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 293  | total loss: [1m[32m0.00829[0m[0m | time: 0.002s
    | Adam | epoch: 293 | loss: 0.00829 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 294  | total loss: [1m[32m0.00822[0m[0m | time: 0.002s
    | Adam | epoch: 294 | loss: 0.00822 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 295  | total loss: [1m[32m0.00816[0m[0m | time: 0.002s
    | Adam | epoch: 295 | loss: 0.00816 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 296  | total loss: [1m[32m0.00810[0m[0m | time: 0.002s
    | Adam | epoch: 296 | loss: 0.00810 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 297  | total loss: [1m[32m0.00803[0m[0m | time: 0.002s
    | Adam | epoch: 297 | loss: 0.00803 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 298  | total loss: [1m[32m0.00797[0m[0m | time: 0.002s
    | Adam | epoch: 298 | loss: 0.00797 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 299  | total loss: [1m[32m0.00791[0m[0m | time: 0.002s
    | Adam | epoch: 299 | loss: 0.00791 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 300  | total loss: [1m[32m0.00785[0m[0m | time: 0.002s
    | Adam | epoch: 300 | loss: 0.00785 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 301  | total loss: [1m[32m0.00779[0m[0m | time: 0.002s
    | Adam | epoch: 301 | loss: 0.00779 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 302  | total loss: [1m[32m0.00773[0m[0m | time: 0.002s
    | Adam | epoch: 302 | loss: 0.00773 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 303  | total loss: [1m[32m0.00768[0m[0m | time: 0.002s
    | Adam | epoch: 303 | loss: 0.00768 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 304  | total loss: [1m[32m0.00762[0m[0m | time: 0.002s
    | Adam | epoch: 304 | loss: 0.00762 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 305  | total loss: [1m[32m0.00756[0m[0m | time: 0.002s
    | Adam | epoch: 305 | loss: 0.00756 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 306  | total loss: [1m[32m0.00751[0m[0m | time: 0.002s
    | Adam | epoch: 306 | loss: 0.00751 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 307  | total loss: [1m[32m0.00745[0m[0m | time: 0.001s
    | Adam | epoch: 307 | loss: 0.00745 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 308  | total loss: [1m[32m0.00740[0m[0m | time: 0.002s
    | Adam | epoch: 308 | loss: 0.00740 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 309  | total loss: [1m[32m0.00734[0m[0m | time: 0.002s
    | Adam | epoch: 309 | loss: 0.00734 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 310  | total loss: [1m[32m0.00729[0m[0m | time: 0.002s
    | Adam | epoch: 310 | loss: 0.00729 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 311  | total loss: [1m[32m0.00724[0m[0m | time: 0.002s
    | Adam | epoch: 311 | loss: 0.00724 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 312  | total loss: [1m[32m0.00718[0m[0m | time: 0.003s
    | Adam | epoch: 312 | loss: 0.00718 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 313  | total loss: [1m[32m0.00713[0m[0m | time: 0.002s
    | Adam | epoch: 313 | loss: 0.00713 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 314  | total loss: [1m[32m0.00708[0m[0m | time: 0.002s
    | Adam | epoch: 314 | loss: 0.00708 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 315  | total loss: [1m[32m0.00703[0m[0m | time: 0.002s
    | Adam | epoch: 315 | loss: 0.00703 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 316  | total loss: [1m[32m0.00698[0m[0m | time: 0.002s
    | Adam | epoch: 316 | loss: 0.00698 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 317  | total loss: [1m[32m0.00693[0m[0m | time: 0.003s
    | Adam | epoch: 317 | loss: 0.00693 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 318  | total loss: [1m[32m0.00688[0m[0m | time: 0.002s
    | Adam | epoch: 318 | loss: 0.00688 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 319  | total loss: [1m[32m0.00684[0m[0m | time: 0.002s
    | Adam | epoch: 319 | loss: 0.00684 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 320  | total loss: [1m[32m0.00679[0m[0m | time: 0.002s
    | Adam | epoch: 320 | loss: 0.00679 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 321  | total loss: [1m[32m0.00674[0m[0m | time: 0.002s
    | Adam | epoch: 321 | loss: 0.00674 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 322  | total loss: [1m[32m0.00670[0m[0m | time: 0.002s
    | Adam | epoch: 322 | loss: 0.00670 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 323  | total loss: [1m[32m0.00665[0m[0m | time: 0.002s
    | Adam | epoch: 323 | loss: 0.00665 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 324  | total loss: [1m[32m0.00660[0m[0m | time: 0.002s
    | Adam | epoch: 324 | loss: 0.00660 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 325  | total loss: [1m[32m0.00656[0m[0m | time: 0.002s
    | Adam | epoch: 325 | loss: 0.00656 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 326  | total loss: [1m[32m0.00652[0m[0m | time: 0.002s
    | Adam | epoch: 326 | loss: 0.00652 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 327  | total loss: [1m[32m0.00647[0m[0m | time: 0.002s
    | Adam | epoch: 327 | loss: 0.00647 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 328  | total loss: [1m[32m0.00643[0m[0m | time: 0.002s
    | Adam | epoch: 328 | loss: 0.00643 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 329  | total loss: [1m[32m0.00638[0m[0m | time: 0.002s
    | Adam | epoch: 329 | loss: 0.00638 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 330  | total loss: [1m[32m0.00634[0m[0m | time: 0.002s
    | Adam | epoch: 330 | loss: 0.00634 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 331  | total loss: [1m[32m0.00630[0m[0m | time: 0.002s
    | Adam | epoch: 331 | loss: 0.00630 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 332  | total loss: [1m[32m0.00626[0m[0m | time: 0.003s
    | Adam | epoch: 332 | loss: 0.00626 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 333  | total loss: [1m[32m0.00622[0m[0m | time: 0.002s
    | Adam | epoch: 333 | loss: 0.00622 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 334  | total loss: [1m[32m0.00618[0m[0m | time: 0.002s
    | Adam | epoch: 334 | loss: 0.00618 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 335  | total loss: [1m[32m0.00614[0m[0m | time: 0.002s
    | Adam | epoch: 335 | loss: 0.00614 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 336  | total loss: [1m[32m0.00610[0m[0m | time: 0.002s
    | Adam | epoch: 336 | loss: 0.00610 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 337  | total loss: [1m[32m0.00606[0m[0m | time: 0.002s
    | Adam | epoch: 337 | loss: 0.00606 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 338  | total loss: [1m[32m0.00602[0m[0m | time: 0.003s
    | Adam | epoch: 338 | loss: 0.00602 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 339  | total loss: [1m[32m0.00598[0m[0m | time: 0.002s
    | Adam | epoch: 339 | loss: 0.00598 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 340  | total loss: [1m[32m0.00594[0m[0m | time: 0.002s
    | Adam | epoch: 340 | loss: 0.00594 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 341  | total loss: [1m[32m0.00590[0m[0m | time: 0.002s
    | Adam | epoch: 341 | loss: 0.00590 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 342  | total loss: [1m[32m0.00587[0m[0m | time: 0.002s
    | Adam | epoch: 342 | loss: 0.00587 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 343  | total loss: [1m[32m0.00583[0m[0m | time: 0.002s
    | Adam | epoch: 343 | loss: 0.00583 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 344  | total loss: [1m[32m0.00579[0m[0m | time: 0.002s
    | Adam | epoch: 344 | loss: 0.00579 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 345  | total loss: [1m[32m0.00575[0m[0m | time: 0.002s
    | Adam | epoch: 345 | loss: 0.00575 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 346  | total loss: [1m[32m0.00572[0m[0m | time: 0.002s
    | Adam | epoch: 346 | loss: 0.00572 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 347  | total loss: [1m[32m0.00568[0m[0m | time: 0.002s
    | Adam | epoch: 347 | loss: 0.00568 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 348  | total loss: [1m[32m0.00565[0m[0m | time: 0.002s
    | Adam | epoch: 348 | loss: 0.00565 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 349  | total loss: [1m[32m0.00561[0m[0m | time: 0.002s
    | Adam | epoch: 349 | loss: 0.00561 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 350  | total loss: [1m[32m0.00558[0m[0m | time: 0.003s
    | Adam | epoch: 350 | loss: 0.00558 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 351  | total loss: [1m[32m0.00554[0m[0m | time: 0.002s
    | Adam | epoch: 351 | loss: 0.00554 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 352  | total loss: [1m[32m0.00551[0m[0m | time: 0.002s
    | Adam | epoch: 352 | loss: 0.00551 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 353  | total loss: [1m[32m0.00548[0m[0m | time: 0.002s
    | Adam | epoch: 353 | loss: 0.00548 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 354  | total loss: [1m[32m0.00544[0m[0m | time: 0.002s
    | Adam | epoch: 354 | loss: 0.00544 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 355  | total loss: [1m[32m0.00541[0m[0m | time: 0.002s
    | Adam | epoch: 355 | loss: 0.00541 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 356  | total loss: [1m[32m0.00538[0m[0m | time: 0.003s
    | Adam | epoch: 356 | loss: 0.00538 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 357  | total loss: [1m[32m0.00534[0m[0m | time: 0.002s
    | Adam | epoch: 357 | loss: 0.00534 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 358  | total loss: [1m[32m0.00531[0m[0m | time: 0.002s
    | Adam | epoch: 358 | loss: 0.00531 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 359  | total loss: [1m[32m0.00528[0m[0m | time: 0.002s
    | Adam | epoch: 359 | loss: 0.00528 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 360  | total loss: [1m[32m0.00525[0m[0m | time: 0.002s
    | Adam | epoch: 360 | loss: 0.00525 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 361  | total loss: [1m[32m0.00522[0m[0m | time: 0.002s
    | Adam | epoch: 361 | loss: 0.00522 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 362  | total loss: [1m[32m0.00519[0m[0m | time: 0.002s
    | Adam | epoch: 362 | loss: 0.00519 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 363  | total loss: [1m[32m0.00516[0m[0m | time: 0.002s
    | Adam | epoch: 363 | loss: 0.00516 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 364  | total loss: [1m[32m0.00513[0m[0m | time: 0.002s
    | Adam | epoch: 364 | loss: 0.00513 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 365  | total loss: [1m[32m0.00510[0m[0m | time: 0.002s
    | Adam | epoch: 365 | loss: 0.00510 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 366  | total loss: [1m[32m0.00507[0m[0m | time: 0.002s
    | Adam | epoch: 366 | loss: 0.00507 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 367  | total loss: [1m[32m0.00504[0m[0m | time: 0.002s
    | Adam | epoch: 367 | loss: 0.00504 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 368  | total loss: [1m[32m0.00501[0m[0m | time: 0.002s
    | Adam | epoch: 368 | loss: 0.00501 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 369  | total loss: [1m[32m0.00498[0m[0m | time: 0.002s
    | Adam | epoch: 369 | loss: 0.00498 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 370  | total loss: [1m[32m0.00495[0m[0m | time: 0.002s
    | Adam | epoch: 370 | loss: 0.00495 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 371  | total loss: [1m[32m0.00492[0m[0m | time: 0.002s
    | Adam | epoch: 371 | loss: 0.00492 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 372  | total loss: [1m[32m0.00489[0m[0m | time: 0.002s
    | Adam | epoch: 372 | loss: 0.00489 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 373  | total loss: [1m[32m0.00487[0m[0m | time: 0.002s
    | Adam | epoch: 373 | loss: 0.00487 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 374  | total loss: [1m[32m0.00484[0m[0m | time: 0.002s
    | Adam | epoch: 374 | loss: 0.00484 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 375  | total loss: [1m[32m0.00481[0m[0m | time: 0.002s
    | Adam | epoch: 375 | loss: 0.00481 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 376  | total loss: [1m[32m0.00478[0m[0m | time: 0.002s
    | Adam | epoch: 376 | loss: 0.00478 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 377  | total loss: [1m[32m0.00476[0m[0m | time: 0.002s
    | Adam | epoch: 377 | loss: 0.00476 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 378  | total loss: [1m[32m0.00473[0m[0m | time: 0.002s
    | Adam | epoch: 378 | loss: 0.00473 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 379  | total loss: [1m[32m0.00470[0m[0m | time: 0.001s
    | Adam | epoch: 379 | loss: 0.00470 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 380  | total loss: [1m[32m0.00468[0m[0m | time: 0.002s
    | Adam | epoch: 380 | loss: 0.00468 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 381  | total loss: [1m[32m0.00465[0m[0m | time: 0.002s
    | Adam | epoch: 381 | loss: 0.00465 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 382  | total loss: [1m[32m0.00463[0m[0m | time: 0.002s
    | Adam | epoch: 382 | loss: 0.00463 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 383  | total loss: [1m[32m0.00460[0m[0m | time: 0.001s
    | Adam | epoch: 383 | loss: 0.00460 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 384  | total loss: [1m[32m0.00457[0m[0m | time: 0.002s
    | Adam | epoch: 384 | loss: 0.00457 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 385  | total loss: [1m[32m0.00455[0m[0m | time: 0.002s
    | Adam | epoch: 385 | loss: 0.00455 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 386  | total loss: [1m[32m0.00453[0m[0m | time: 0.002s
    | Adam | epoch: 386 | loss: 0.00453 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 387  | total loss: [1m[32m0.00450[0m[0m | time: 0.002s
    | Adam | epoch: 387 | loss: 0.00450 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 388  | total loss: [1m[32m0.00448[0m[0m | time: 0.002s
    | Adam | epoch: 388 | loss: 0.00448 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 389  | total loss: [1m[32m0.00445[0m[0m | time: 0.002s
    | Adam | epoch: 389 | loss: 0.00445 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 390  | total loss: [1m[32m0.00443[0m[0m | time: 0.002s
    | Adam | epoch: 390 | loss: 0.00443 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 391  | total loss: [1m[32m0.00440[0m[0m | time: 0.002s
    | Adam | epoch: 391 | loss: 0.00440 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 392  | total loss: [1m[32m0.00438[0m[0m | time: 0.002s
    | Adam | epoch: 392 | loss: 0.00438 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 393  | total loss: [1m[32m0.00436[0m[0m | time: 0.002s
    | Adam | epoch: 393 | loss: 0.00436 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 394  | total loss: [1m[32m0.00433[0m[0m | time: 0.002s
    | Adam | epoch: 394 | loss: 0.00433 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 395  | total loss: [1m[32m0.00431[0m[0m | time: 0.002s
    | Adam | epoch: 395 | loss: 0.00431 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 396  | total loss: [1m[32m0.00429[0m[0m | time: 0.002s
    | Adam | epoch: 396 | loss: 0.00429 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 397  | total loss: [1m[32m0.00427[0m[0m | time: 0.002s
    | Adam | epoch: 397 | loss: 0.00427 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 398  | total loss: [1m[32m0.00424[0m[0m | time: 0.002s
    | Adam | epoch: 398 | loss: 0.00424 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 399  | total loss: [1m[32m0.00422[0m[0m | time: 0.002s
    | Adam | epoch: 399 | loss: 0.00422 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 400  | total loss: [1m[32m0.00420[0m[0m | time: 0.002s
    | Adam | epoch: 400 | loss: 0.00420 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 401  | total loss: [1m[32m0.00418[0m[0m | time: 0.002s
    | Adam | epoch: 401 | loss: 0.00418 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 402  | total loss: [1m[32m0.00415[0m[0m | time: 0.003s
    | Adam | epoch: 402 | loss: 0.00415 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 403  | total loss: [1m[32m0.00413[0m[0m | time: 0.003s
    | Adam | epoch: 403 | loss: 0.00413 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 404  | total loss: [1m[32m0.00411[0m[0m | time: 0.002s
    | Adam | epoch: 404 | loss: 0.00411 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 405  | total loss: [1m[32m0.00409[0m[0m | time: 0.002s
    | Adam | epoch: 405 | loss: 0.00409 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 406  | total loss: [1m[32m0.00407[0m[0m | time: 0.002s
    | Adam | epoch: 406 | loss: 0.00407 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 407  | total loss: [1m[32m0.00405[0m[0m | time: 0.002s
    | Adam | epoch: 407 | loss: 0.00405 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 408  | total loss: [1m[32m0.00403[0m[0m | time: 0.002s
    | Adam | epoch: 408 | loss: 0.00403 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 409  | total loss: [1m[32m0.00401[0m[0m | time: 0.002s
    | Adam | epoch: 409 | loss: 0.00401 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 410  | total loss: [1m[32m0.00399[0m[0m | time: 0.002s
    | Adam | epoch: 410 | loss: 0.00399 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 411  | total loss: [1m[32m0.00397[0m[0m | time: 0.002s
    | Adam | epoch: 411 | loss: 0.00397 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 412  | total loss: [1m[32m0.00395[0m[0m | time: 0.002s
    | Adam | epoch: 412 | loss: 0.00395 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 413  | total loss: [1m[32m0.00393[0m[0m | time: 0.002s
    | Adam | epoch: 413 | loss: 0.00393 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 414  | total loss: [1m[32m0.00391[0m[0m | time: 0.002s
    | Adam | epoch: 414 | loss: 0.00391 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 415  | total loss: [1m[32m0.00389[0m[0m | time: 0.002s
    | Adam | epoch: 415 | loss: 0.00389 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 416  | total loss: [1m[32m0.00387[0m[0m | time: 0.002s
    | Adam | epoch: 416 | loss: 0.00387 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 417  | total loss: [1m[32m0.00385[0m[0m | time: 0.002s
    | Adam | epoch: 417 | loss: 0.00385 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 418  | total loss: [1m[32m0.00383[0m[0m | time: 0.002s
    | Adam | epoch: 418 | loss: 0.00383 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 419  | total loss: [1m[32m0.00381[0m[0m | time: 0.002s
    | Adam | epoch: 419 | loss: 0.00381 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 420  | total loss: [1m[32m0.00379[0m[0m | time: 0.002s
    | Adam | epoch: 420 | loss: 0.00379 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 421  | total loss: [1m[32m0.00377[0m[0m | time: 0.002s
    | Adam | epoch: 421 | loss: 0.00377 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 422  | total loss: [1m[32m0.00375[0m[0m | time: 0.002s
    | Adam | epoch: 422 | loss: 0.00375 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 423  | total loss: [1m[32m0.00374[0m[0m | time: 0.002s
    | Adam | epoch: 423 | loss: 0.00374 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 424  | total loss: [1m[32m0.00372[0m[0m | time: 0.002s
    | Adam | epoch: 424 | loss: 0.00372 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 425  | total loss: [1m[32m0.00370[0m[0m | time: 0.002s
    | Adam | epoch: 425 | loss: 0.00370 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 426  | total loss: [1m[32m0.00368[0m[0m | time: 0.002s
    | Adam | epoch: 426 | loss: 0.00368 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 427  | total loss: [1m[32m0.00366[0m[0m | time: 0.002s
    | Adam | epoch: 427 | loss: 0.00366 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 428  | total loss: [1m[32m0.00365[0m[0m | time: 0.002s
    | Adam | epoch: 428 | loss: 0.00365 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 429  | total loss: [1m[32m0.00363[0m[0m | time: 0.002s
    | Adam | epoch: 429 | loss: 0.00363 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 430  | total loss: [1m[32m0.00361[0m[0m | time: 0.002s
    | Adam | epoch: 430 | loss: 0.00361 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 431  | total loss: [1m[32m0.00359[0m[0m | time: 0.002s
    | Adam | epoch: 431 | loss: 0.00359 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 432  | total loss: [1m[32m0.00358[0m[0m | time: 0.002s
    | Adam | epoch: 432 | loss: 0.00358 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 433  | total loss: [1m[32m0.00356[0m[0m | time: 0.002s
    | Adam | epoch: 433 | loss: 0.00356 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 434  | total loss: [1m[32m0.00354[0m[0m | time: 0.002s
    | Adam | epoch: 434 | loss: 0.00354 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 435  | total loss: [1m[32m0.00353[0m[0m | time: 0.002s
    | Adam | epoch: 435 | loss: 0.00353 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 436  | total loss: [1m[32m0.00351[0m[0m | time: 0.002s
    | Adam | epoch: 436 | loss: 0.00351 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 437  | total loss: [1m[32m0.00349[0m[0m | time: 0.002s
    | Adam | epoch: 437 | loss: 0.00349 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 438  | total loss: [1m[32m0.00348[0m[0m | time: 0.002s
    | Adam | epoch: 438 | loss: 0.00348 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 439  | total loss: [1m[32m0.00346[0m[0m | time: 0.002s
    | Adam | epoch: 439 | loss: 0.00346 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 440  | total loss: [1m[32m0.00344[0m[0m | time: 0.002s
    | Adam | epoch: 440 | loss: 0.00344 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 441  | total loss: [1m[32m0.00343[0m[0m | time: 0.002s
    | Adam | epoch: 441 | loss: 0.00343 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 442  | total loss: [1m[32m0.00341[0m[0m | time: 0.002s
    | Adam | epoch: 442 | loss: 0.00341 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 443  | total loss: [1m[32m0.00340[0m[0m | time: 0.002s
    | Adam | epoch: 443 | loss: 0.00340 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 444  | total loss: [1m[32m0.00338[0m[0m | time: 0.002s
    | Adam | epoch: 444 | loss: 0.00338 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 445  | total loss: [1m[32m0.00337[0m[0m | time: 0.001s
    | Adam | epoch: 445 | loss: 0.00337 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 446  | total loss: [1m[32m0.00335[0m[0m | time: 0.002s
    | Adam | epoch: 446 | loss: 0.00335 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 447  | total loss: [1m[32m0.00333[0m[0m | time: 0.002s
    | Adam | epoch: 447 | loss: 0.00333 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 448  | total loss: [1m[32m0.00332[0m[0m | time: 0.002s
    | Adam | epoch: 448 | loss: 0.00332 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 449  | total loss: [1m[32m0.00330[0m[0m | time: 0.002s
    | Adam | epoch: 449 | loss: 0.00330 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 450  | total loss: [1m[32m0.00329[0m[0m | time: 0.002s
    | Adam | epoch: 450 | loss: 0.00329 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 451  | total loss: [1m[32m0.00327[0m[0m | time: 0.002s
    | Adam | epoch: 451 | loss: 0.00327 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 452  | total loss: [1m[32m0.00326[0m[0m | time: 0.002s
    | Adam | epoch: 452 | loss: 0.00326 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 453  | total loss: [1m[32m0.00324[0m[0m | time: 0.002s
    | Adam | epoch: 453 | loss: 0.00324 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 454  | total loss: [1m[32m0.00323[0m[0m | time: 0.002s
    | Adam | epoch: 454 | loss: 0.00323 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 455  | total loss: [1m[32m0.00322[0m[0m | time: 0.002s
    | Adam | epoch: 455 | loss: 0.00322 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 456  | total loss: [1m[32m0.00320[0m[0m | time: 0.002s
    | Adam | epoch: 456 | loss: 0.00320 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 457  | total loss: [1m[32m0.00319[0m[0m | time: 0.002s
    | Adam | epoch: 457 | loss: 0.00319 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 458  | total loss: [1m[32m0.00317[0m[0m | time: 0.002s
    | Adam | epoch: 458 | loss: 0.00317 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 459  | total loss: [1m[32m0.00316[0m[0m | time: 0.002s
    | Adam | epoch: 459 | loss: 0.00316 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 460  | total loss: [1m[32m0.00314[0m[0m | time: 0.002s
    | Adam | epoch: 460 | loss: 0.00314 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 461  | total loss: [1m[32m0.00313[0m[0m | time: 0.002s
    | Adam | epoch: 461 | loss: 0.00313 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 462  | total loss: [1m[32m0.00312[0m[0m | time: 0.002s
    | Adam | epoch: 462 | loss: 0.00312 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 463  | total loss: [1m[32m0.00310[0m[0m | time: 0.002s
    | Adam | epoch: 463 | loss: 0.00310 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 464  | total loss: [1m[32m0.00309[0m[0m | time: 0.002s
    | Adam | epoch: 464 | loss: 0.00309 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 465  | total loss: [1m[32m0.00308[0m[0m | time: 0.002s
    | Adam | epoch: 465 | loss: 0.00308 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 466  | total loss: [1m[32m0.00306[0m[0m | time: 0.002s
    | Adam | epoch: 466 | loss: 0.00306 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 467  | total loss: [1m[32m0.00305[0m[0m | time: 0.001s
    | Adam | epoch: 467 | loss: 0.00305 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 468  | total loss: [1m[32m0.00304[0m[0m | time: 0.002s
    | Adam | epoch: 468 | loss: 0.00304 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 469  | total loss: [1m[32m0.00302[0m[0m | time: 0.002s
    | Adam | epoch: 469 | loss: 0.00302 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 470  | total loss: [1m[32m0.00301[0m[0m | time: 0.002s
    | Adam | epoch: 470 | loss: 0.00301 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 471  | total loss: [1m[32m0.00300[0m[0m | time: 0.002s
    | Adam | epoch: 471 | loss: 0.00300 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 472  | total loss: [1m[32m0.00298[0m[0m | time: 0.002s
    | Adam | epoch: 472 | loss: 0.00298 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 473  | total loss: [1m[32m0.00297[0m[0m | time: 0.002s
    | Adam | epoch: 473 | loss: 0.00297 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 474  | total loss: [1m[32m0.00296[0m[0m | time: 0.002s
    | Adam | epoch: 474 | loss: 0.00296 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 475  | total loss: [1m[32m0.00294[0m[0m | time: 0.002s
    | Adam | epoch: 475 | loss: 0.00294 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 476  | total loss: [1m[32m0.00293[0m[0m | time: 0.002s
    | Adam | epoch: 476 | loss: 0.00293 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 477  | total loss: [1m[32m0.00292[0m[0m | time: 0.002s
    | Adam | epoch: 477 | loss: 0.00292 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 478  | total loss: [1m[32m0.00291[0m[0m | time: 0.002s
    | Adam | epoch: 478 | loss: 0.00291 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 479  | total loss: [1m[32m0.00289[0m[0m | time: 0.002s
    | Adam | epoch: 479 | loss: 0.00289 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 480  | total loss: [1m[32m0.00288[0m[0m | time: 0.002s
    | Adam | epoch: 480 | loss: 0.00288 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 481  | total loss: [1m[32m0.00287[0m[0m | time: 0.002s
    | Adam | epoch: 481 | loss: 0.00287 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 482  | total loss: [1m[32m0.00286[0m[0m | time: 0.002s
    | Adam | epoch: 482 | loss: 0.00286 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 483  | total loss: [1m[32m0.00285[0m[0m | time: 0.002s
    | Adam | epoch: 483 | loss: 0.00285 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 484  | total loss: [1m[32m0.00283[0m[0m | time: 0.002s
    | Adam | epoch: 484 | loss: 0.00283 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 485  | total loss: [1m[32m0.00282[0m[0m | time: 0.002s
    | Adam | epoch: 485 | loss: 0.00282 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 486  | total loss: [1m[32m0.00281[0m[0m | time: 0.002s
    | Adam | epoch: 486 | loss: 0.00281 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 487  | total loss: [1m[32m0.00280[0m[0m | time: 0.002s
    | Adam | epoch: 487 | loss: 0.00280 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 488  | total loss: [1m[32m0.00279[0m[0m | time: 0.002s
    | Adam | epoch: 488 | loss: 0.00279 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 489  | total loss: [1m[32m0.00278[0m[0m | time: 0.002s
    | Adam | epoch: 489 | loss: 0.00278 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 490  | total loss: [1m[32m0.00276[0m[0m | time: 0.002s
    | Adam | epoch: 490 | loss: 0.00276 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 491  | total loss: [1m[32m0.00275[0m[0m | time: 0.002s
    | Adam | epoch: 491 | loss: 0.00275 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 492  | total loss: [1m[32m0.00274[0m[0m | time: 0.002s
    | Adam | epoch: 492 | loss: 0.00274 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 493  | total loss: [1m[32m0.00273[0m[0m | time: 0.002s
    | Adam | epoch: 493 | loss: 0.00273 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 494  | total loss: [1m[32m0.00272[0m[0m | time: 0.002s
    | Adam | epoch: 494 | loss: 0.00272 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 495  | total loss: [1m[32m0.00271[0m[0m | time: 0.002s
    | Adam | epoch: 495 | loss: 0.00271 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 496  | total loss: [1m[32m0.00270[0m[0m | time: 0.002s
    | Adam | epoch: 496 | loss: 0.00270 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 497  | total loss: [1m[32m0.00269[0m[0m | time: 0.002s
    | Adam | epoch: 497 | loss: 0.00269 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 498  | total loss: [1m[32m0.00268[0m[0m | time: 0.002s
    | Adam | epoch: 498 | loss: 0.00268 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 499  | total loss: [1m[32m0.00266[0m[0m | time: 0.002s
    | Adam | epoch: 499 | loss: 0.00266 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 500  | total loss: [1m[32m0.00265[0m[0m | time: 0.002s
    | Adam | epoch: 500 | loss: 0.00265 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 501  | total loss: [1m[32m0.00264[0m[0m | time: 0.002s
    | Adam | epoch: 501 | loss: 0.00264 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 502  | total loss: [1m[32m0.00263[0m[0m | time: 0.002s
    | Adam | epoch: 502 | loss: 0.00263 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 503  | total loss: [1m[32m0.00262[0m[0m | time: 0.002s
    | Adam | epoch: 503 | loss: 0.00262 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 504  | total loss: [1m[32m0.00261[0m[0m | time: 0.002s
    | Adam | epoch: 504 | loss: 0.00261 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 505  | total loss: [1m[32m0.00260[0m[0m | time: 0.002s
    | Adam | epoch: 505 | loss: 0.00260 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 506  | total loss: [1m[32m0.00259[0m[0m | time: 0.002s
    | Adam | epoch: 506 | loss: 0.00259 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 507  | total loss: [1m[32m0.00258[0m[0m | time: 0.002s
    | Adam | epoch: 507 | loss: 0.00258 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 508  | total loss: [1m[32m0.00257[0m[0m | time: 0.002s
    | Adam | epoch: 508 | loss: 0.00257 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 509  | total loss: [1m[32m0.00256[0m[0m | time: 0.002s
    | Adam | epoch: 509 | loss: 0.00256 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 510  | total loss: [1m[32m0.00255[0m[0m | time: 0.002s
    | Adam | epoch: 510 | loss: 0.00255 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 511  | total loss: [1m[32m0.00254[0m[0m | time: 0.002s
    | Adam | epoch: 511 | loss: 0.00254 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 512  | total loss: [1m[32m0.00253[0m[0m | time: 0.002s
    | Adam | epoch: 512 | loss: 0.00253 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 513  | total loss: [1m[32m0.00252[0m[0m | time: 0.001s
    | Adam | epoch: 513 | loss: 0.00252 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 514  | total loss: [1m[32m0.00251[0m[0m | time: 0.002s
    | Adam | epoch: 514 | loss: 0.00251 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 515  | total loss: [1m[32m0.00250[0m[0m | time: 0.002s
    | Adam | epoch: 515 | loss: 0.00250 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 516  | total loss: [1m[32m0.00249[0m[0m | time: 0.002s
    | Adam | epoch: 516 | loss: 0.00249 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 517  | total loss: [1m[32m0.00248[0m[0m | time: 0.002s
    | Adam | epoch: 517 | loss: 0.00248 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 518  | total loss: [1m[32m0.00247[0m[0m | time: 0.002s
    | Adam | epoch: 518 | loss: 0.00247 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 519  | total loss: [1m[32m0.00246[0m[0m | time: 0.002s
    | Adam | epoch: 519 | loss: 0.00246 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 520  | total loss: [1m[32m0.00245[0m[0m | time: 0.002s
    | Adam | epoch: 520 | loss: 0.00245 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 521  | total loss: [1m[32m0.00244[0m[0m | time: 0.002s
    | Adam | epoch: 521 | loss: 0.00244 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 522  | total loss: [1m[32m0.00243[0m[0m | time: 0.002s
    | Adam | epoch: 522 | loss: 0.00243 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 523  | total loss: [1m[32m0.00242[0m[0m | time: 0.002s
    | Adam | epoch: 523 | loss: 0.00242 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 524  | total loss: [1m[32m0.00240[0m[0m | time: 0.038s
    | Adam | epoch: 524 | loss: 0.00240 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 525  | total loss: [1m[32m0.00240[0m[0m | time: 0.002s
    | Adam | epoch: 525 | loss: 0.00240 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 526  | total loss: [1m[32m0.00240[0m[0m | time: 0.002s
    | Adam | epoch: 526 | loss: 0.00240 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 527  | total loss: [1m[32m0.00239[0m[0m | time: 0.002s
    | Adam | epoch: 527 | loss: 0.00239 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 528  | total loss: [1m[32m0.00238[0m[0m | time: 0.002s
    | Adam | epoch: 528 | loss: 0.00238 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 529  | total loss: [1m[32m0.00237[0m[0m | time: 0.002s
    | Adam | epoch: 529 | loss: 0.00237 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 530  | total loss: [1m[32m0.00236[0m[0m | time: 0.002s
    | Adam | epoch: 530 | loss: 0.00236 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 531  | total loss: [1m[32m0.00235[0m[0m | time: 0.002s
    | Adam | epoch: 531 | loss: 0.00235 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 532  | total loss: [1m[32m0.00234[0m[0m | time: 0.002s
    | Adam | epoch: 532 | loss: 0.00234 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 533  | total loss: [1m[32m0.00233[0m[0m | time: 0.005s
    | Adam | epoch: 533 | loss: 0.00233 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 534  | total loss: [1m[32m0.00232[0m[0m | time: 0.008s
    | Adam | epoch: 534 | loss: 0.00232 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 535  | total loss: [1m[32m0.00232[0m[0m | time: 0.002s
    | Adam | epoch: 535 | loss: 0.00232 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 536  | total loss: [1m[32m0.00231[0m[0m | time: 0.002s
    | Adam | epoch: 536 | loss: 0.00231 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 537  | total loss: [1m[32m0.00230[0m[0m | time: 0.002s
    | Adam | epoch: 537 | loss: 0.00230 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 538  | total loss: [1m[32m0.00229[0m[0m | time: 0.002s
    | Adam | epoch: 538 | loss: 0.00229 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 539  | total loss: [1m[32m0.00228[0m[0m | time: 0.002s
    | Adam | epoch: 539 | loss: 0.00228 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 540  | total loss: [1m[32m0.00227[0m[0m | time: 0.002s
    | Adam | epoch: 540 | loss: 0.00227 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 541  | total loss: [1m[32m0.00226[0m[0m | time: 0.002s
    | Adam | epoch: 541 | loss: 0.00226 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 542  | total loss: [1m[32m0.00226[0m[0m | time: 0.002s
    | Adam | epoch: 542 | loss: 0.00226 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 543  | total loss: [1m[32m0.00225[0m[0m | time: 0.002s
    | Adam | epoch: 543 | loss: 0.00225 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 544  | total loss: [1m[32m0.00224[0m[0m | time: 0.002s
    | Adam | epoch: 544 | loss: 0.00224 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 545  | total loss: [1m[32m0.00223[0m[0m | time: 0.003s
    | Adam | epoch: 545 | loss: 0.00223 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 546  | total loss: [1m[32m0.00222[0m[0m | time: 0.002s
    | Adam | epoch: 546 | loss: 0.00222 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 547  | total loss: [1m[32m0.00221[0m[0m | time: 0.002s
    | Adam | epoch: 547 | loss: 0.00221 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 548  | total loss: [1m[32m0.00221[0m[0m | time: 0.002s
    | Adam | epoch: 548 | loss: 0.00221 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 549  | total loss: [1m[32m0.00220[0m[0m | time: 0.002s
    | Adam | epoch: 549 | loss: 0.00220 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 550  | total loss: [1m[32m0.00219[0m[0m | time: 0.002s
    | Adam | epoch: 550 | loss: 0.00219 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 551  | total loss: [1m[32m0.00218[0m[0m | time: 0.002s
    | Adam | epoch: 551 | loss: 0.00218 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 552  | total loss: [1m[32m0.00217[0m[0m | time: 0.003s
    | Adam | epoch: 552 | loss: 0.00217 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 553  | total loss: [1m[32m0.00217[0m[0m | time: 0.003s
    | Adam | epoch: 553 | loss: 0.00217 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 554  | total loss: [1m[32m0.00216[0m[0m | time: 0.002s
    | Adam | epoch: 554 | loss: 0.00216 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 555  | total loss: [1m[32m0.00215[0m[0m | time: 0.002s
    | Adam | epoch: 555 | loss: 0.00215 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 556  | total loss: [1m[32m0.00214[0m[0m | time: 0.002s
    | Adam | epoch: 556 | loss: 0.00214 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 557  | total loss: [1m[32m0.00214[0m[0m | time: 0.002s
    | Adam | epoch: 557 | loss: 0.00214 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 558  | total loss: [1m[32m0.00213[0m[0m | time: 0.003s
    | Adam | epoch: 558 | loss: 0.00213 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 559  | total loss: [1m[32m0.00212[0m[0m | time: 0.003s
    | Adam | epoch: 559 | loss: 0.00212 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 560  | total loss: [1m[32m0.00211[0m[0m | time: 0.002s
    | Adam | epoch: 560 | loss: 0.00211 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 561  | total loss: [1m[32m0.00211[0m[0m | time: 0.002s
    | Adam | epoch: 561 | loss: 0.00211 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 562  | total loss: [1m[32m0.00210[0m[0m | time: 0.002s
    | Adam | epoch: 562 | loss: 0.00210 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 563  | total loss: [1m[32m0.00209[0m[0m | time: 0.002s
    | Adam | epoch: 563 | loss: 0.00209 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 564  | total loss: [1m[32m0.00208[0m[0m | time: 0.002s
    | Adam | epoch: 564 | loss: 0.00208 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 565  | total loss: [1m[32m0.00208[0m[0m | time: 0.002s
    | Adam | epoch: 565 | loss: 0.00208 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 566  | total loss: [1m[32m0.00207[0m[0m | time: 0.002s
    | Adam | epoch: 566 | loss: 0.00207 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 567  | total loss: [1m[32m0.00206[0m[0m | time: 0.002s
    | Adam | epoch: 567 | loss: 0.00206 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 568  | total loss: [1m[32m0.00205[0m[0m | time: 0.002s
    | Adam | epoch: 568 | loss: 0.00205 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 569  | total loss: [1m[32m0.00205[0m[0m | time: 0.002s
    | Adam | epoch: 569 | loss: 0.00205 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 570  | total loss: [1m[32m0.00204[0m[0m | time: 0.002s
    | Adam | epoch: 570 | loss: 0.00204 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 571  | total loss: [1m[32m0.00203[0m[0m | time: 0.002s
    | Adam | epoch: 571 | loss: 0.00203 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 572  | total loss: [1m[32m0.00203[0m[0m | time: 0.002s
    | Adam | epoch: 572 | loss: 0.00203 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 573  | total loss: [1m[32m0.00202[0m[0m | time: 0.002s
    | Adam | epoch: 573 | loss: 0.00202 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 574  | total loss: [1m[32m0.00201[0m[0m | time: 0.002s
    | Adam | epoch: 574 | loss: 0.00201 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 575  | total loss: [1m[32m0.00200[0m[0m | time: 0.002s
    | Adam | epoch: 575 | loss: 0.00200 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 576  | total loss: [1m[32m0.00200[0m[0m | time: 0.002s
    | Adam | epoch: 576 | loss: 0.00200 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 577  | total loss: [1m[32m0.00199[0m[0m | time: 0.002s
    | Adam | epoch: 577 | loss: 0.00199 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 578  | total loss: [1m[32m0.00198[0m[0m | time: 0.002s
    | Adam | epoch: 578 | loss: 0.00198 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 579  | total loss: [1m[32m0.00198[0m[0m | time: 0.002s
    | Adam | epoch: 579 | loss: 0.00198 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 580  | total loss: [1m[32m0.00197[0m[0m | time: 0.002s
    | Adam | epoch: 580 | loss: 0.00197 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 581  | total loss: [1m[32m0.00196[0m[0m | time: 0.002s
    | Adam | epoch: 581 | loss: 0.00196 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 582  | total loss: [1m[32m0.00196[0m[0m | time: 0.002s
    | Adam | epoch: 582 | loss: 0.00196 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 583  | total loss: [1m[32m0.00195[0m[0m | time: 0.002s
    | Adam | epoch: 583 | loss: 0.00195 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 584  | total loss: [1m[32m0.00194[0m[0m | time: 0.002s
    | Adam | epoch: 584 | loss: 0.00194 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 585  | total loss: [1m[32m0.00194[0m[0m | time: 0.002s
    | Adam | epoch: 585 | loss: 0.00194 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 586  | total loss: [1m[32m0.00193[0m[0m | time: 0.002s
    | Adam | epoch: 586 | loss: 0.00193 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 587  | total loss: [1m[32m0.00192[0m[0m | time: 0.002s
    | Adam | epoch: 587 | loss: 0.00192 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 588  | total loss: [1m[32m0.00192[0m[0m | time: 0.002s
    | Adam | epoch: 588 | loss: 0.00192 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 589  | total loss: [1m[32m0.00191[0m[0m | time: 0.002s
    | Adam | epoch: 589 | loss: 0.00191 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 590  | total loss: [1m[32m0.00190[0m[0m | time: 0.002s
    | Adam | epoch: 590 | loss: 0.00190 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 591  | total loss: [1m[32m0.00190[0m[0m | time: 0.002s
    | Adam | epoch: 591 | loss: 0.00190 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 592  | total loss: [1m[32m0.00189[0m[0m | time: 0.002s
    | Adam | epoch: 592 | loss: 0.00189 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 593  | total loss: [1m[32m0.00188[0m[0m | time: 0.002s
    | Adam | epoch: 593 | loss: 0.00188 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 594  | total loss: [1m[32m0.00188[0m[0m | time: 0.002s
    | Adam | epoch: 594 | loss: 0.00188 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 595  | total loss: [1m[32m0.00187[0m[0m | time: 0.002s
    | Adam | epoch: 595 | loss: 0.00187 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 596  | total loss: [1m[32m0.00187[0m[0m | time: 0.002s
    | Adam | epoch: 596 | loss: 0.00187 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 597  | total loss: [1m[32m0.00186[0m[0m | time: 0.002s
    | Adam | epoch: 597 | loss: 0.00186 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 598  | total loss: [1m[32m0.00185[0m[0m | time: 0.003s
    | Adam | epoch: 598 | loss: 0.00185 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 599  | total loss: [1m[32m0.00185[0m[0m | time: 0.002s
    | Adam | epoch: 599 | loss: 0.00185 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 600  | total loss: [1m[32m0.00184[0m[0m | time: 0.002s
    | Adam | epoch: 600 | loss: 0.00184 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 601  | total loss: [1m[32m0.00184[0m[0m | time: 0.002s
    | Adam | epoch: 601 | loss: 0.00184 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 602  | total loss: [1m[32m0.00183[0m[0m | time: 0.003s
    | Adam | epoch: 602 | loss: 0.00183 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 603  | total loss: [1m[32m0.00182[0m[0m | time: 0.003s
    | Adam | epoch: 603 | loss: 0.00182 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 604  | total loss: [1m[32m0.00182[0m[0m | time: 0.002s
    | Adam | epoch: 604 | loss: 0.00182 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 605  | total loss: [1m[32m0.00181[0m[0m | time: 0.002s
    | Adam | epoch: 605 | loss: 0.00181 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 606  | total loss: [1m[32m0.00181[0m[0m | time: 0.003s
    | Adam | epoch: 606 | loss: 0.00181 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 607  | total loss: [1m[32m0.00180[0m[0m | time: 0.003s
    | Adam | epoch: 607 | loss: 0.00180 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 608  | total loss: [1m[32m0.00179[0m[0m | time: 0.002s
    | Adam | epoch: 608 | loss: 0.00179 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 609  | total loss: [1m[32m0.00179[0m[0m | time: 0.002s
    | Adam | epoch: 609 | loss: 0.00179 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 610  | total loss: [1m[32m0.00178[0m[0m | time: 0.002s
    | Adam | epoch: 610 | loss: 0.00178 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 611  | total loss: [1m[32m0.00178[0m[0m | time: 0.002s
    | Adam | epoch: 611 | loss: 0.00178 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 612  | total loss: [1m[32m0.00177[0m[0m | time: 0.002s
    | Adam | epoch: 612 | loss: 0.00177 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 613  | total loss: [1m[32m0.00176[0m[0m | time: 0.002s
    | Adam | epoch: 613 | loss: 0.00176 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 614  | total loss: [1m[32m0.00176[0m[0m | time: 0.002s
    | Adam | epoch: 614 | loss: 0.00176 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 615  | total loss: [1m[32m0.00175[0m[0m | time: 0.002s
    | Adam | epoch: 615 | loss: 0.00175 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 616  | total loss: [1m[32m0.00175[0m[0m | time: 0.002s
    | Adam | epoch: 616 | loss: 0.00175 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 617  | total loss: [1m[32m0.00174[0m[0m | time: 0.002s
    | Adam | epoch: 617 | loss: 0.00174 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 618  | total loss: [1m[32m0.00174[0m[0m | time: 0.002s
    | Adam | epoch: 618 | loss: 0.00174 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 619  | total loss: [1m[32m0.00173[0m[0m | time: 0.002s
    | Adam | epoch: 619 | loss: 0.00173 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 620  | total loss: [1m[32m0.00172[0m[0m | time: 0.002s
    | Adam | epoch: 620 | loss: 0.00172 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 621  | total loss: [1m[32m0.00172[0m[0m | time: 0.002s
    | Adam | epoch: 621 | loss: 0.00172 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 622  | total loss: [1m[32m0.00171[0m[0m | time: 0.002s
    | Adam | epoch: 622 | loss: 0.00171 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 623  | total loss: [1m[32m0.00171[0m[0m | time: 0.002s
    | Adam | epoch: 623 | loss: 0.00171 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 624  | total loss: [1m[32m0.00170[0m[0m | time: 0.002s
    | Adam | epoch: 624 | loss: 0.00170 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 625  | total loss: [1m[32m0.00170[0m[0m | time: 0.002s
    | Adam | epoch: 625 | loss: 0.00170 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 626  | total loss: [1m[32m0.00169[0m[0m | time: 0.002s
    | Adam | epoch: 626 | loss: 0.00169 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 627  | total loss: [1m[32m0.00169[0m[0m | time: 0.002s
    | Adam | epoch: 627 | loss: 0.00169 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 628  | total loss: [1m[32m0.00168[0m[0m | time: 0.002s
    | Adam | epoch: 628 | loss: 0.00168 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 629  | total loss: [1m[32m0.00168[0m[0m | time: 0.002s
    | Adam | epoch: 629 | loss: 0.00168 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 630  | total loss: [1m[32m0.00167[0m[0m | time: 0.002s
    | Adam | epoch: 630 | loss: 0.00167 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 631  | total loss: [1m[32m0.00167[0m[0m | time: 0.002s
    | Adam | epoch: 631 | loss: 0.00167 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 632  | total loss: [1m[32m0.00166[0m[0m | time: 0.003s
    | Adam | epoch: 632 | loss: 0.00166 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 633  | total loss: [1m[32m0.00166[0m[0m | time: 0.003s
    | Adam | epoch: 633 | loss: 0.00166 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 634  | total loss: [1m[32m0.00165[0m[0m | time: 0.002s
    | Adam | epoch: 634 | loss: 0.00165 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 635  | total loss: [1m[32m0.00165[0m[0m | time: 0.002s
    | Adam | epoch: 635 | loss: 0.00165 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 636  | total loss: [1m[32m0.00164[0m[0m | time: 0.002s
    | Adam | epoch: 636 | loss: 0.00164 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 637  | total loss: [1m[32m0.00163[0m[0m | time: 0.002s
    | Adam | epoch: 637 | loss: 0.00163 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 638  | total loss: [1m[32m0.00163[0m[0m | time: 0.002s
    | Adam | epoch: 638 | loss: 0.00163 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 639  | total loss: [1m[32m0.00162[0m[0m | time: 0.002s
    | Adam | epoch: 639 | loss: 0.00162 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 640  | total loss: [1m[32m0.00162[0m[0m | time: 0.002s
    | Adam | epoch: 640 | loss: 0.00162 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 641  | total loss: [1m[32m0.00161[0m[0m | time: 0.002s
    | Adam | epoch: 641 | loss: 0.00161 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 642  | total loss: [1m[32m0.00161[0m[0m | time: 0.002s
    | Adam | epoch: 642 | loss: 0.00161 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 643  | total loss: [1m[32m0.00160[0m[0m | time: 0.002s
    | Adam | epoch: 643 | loss: 0.00160 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 644  | total loss: [1m[32m0.00160[0m[0m | time: 0.002s
    | Adam | epoch: 644 | loss: 0.00160 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 645  | total loss: [1m[32m0.00159[0m[0m | time: 0.002s
    | Adam | epoch: 645 | loss: 0.00159 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 646  | total loss: [1m[32m0.00159[0m[0m | time: 0.003s
    | Adam | epoch: 646 | loss: 0.00159 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 647  | total loss: [1m[32m0.00159[0m[0m | time: 0.002s
    | Adam | epoch: 647 | loss: 0.00159 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 648  | total loss: [1m[32m0.00158[0m[0m | time: 0.002s
    | Adam | epoch: 648 | loss: 0.00158 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 649  | total loss: [1m[32m0.00158[0m[0m | time: 0.002s
    | Adam | epoch: 649 | loss: 0.00158 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 650  | total loss: [1m[32m0.00157[0m[0m | time: 0.002s
    | Adam | epoch: 650 | loss: 0.00157 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 651  | total loss: [1m[32m0.00157[0m[0m | time: 0.002s
    | Adam | epoch: 651 | loss: 0.00157 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 652  | total loss: [1m[32m0.00156[0m[0m | time: 0.002s
    | Adam | epoch: 652 | loss: 0.00156 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 653  | total loss: [1m[32m0.00156[0m[0m | time: 0.001s
    | Adam | epoch: 653 | loss: 0.00156 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 654  | total loss: [1m[32m0.00155[0m[0m | time: 0.002s
    | Adam | epoch: 654 | loss: 0.00155 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 655  | total loss: [1m[32m0.00155[0m[0m | time: 0.002s
    | Adam | epoch: 655 | loss: 0.00155 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 656  | total loss: [1m[32m0.00154[0m[0m | time: 0.002s
    | Adam | epoch: 656 | loss: 0.00154 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 657  | total loss: [1m[32m0.00154[0m[0m | time: 0.002s
    | Adam | epoch: 657 | loss: 0.00154 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 658  | total loss: [1m[32m0.00153[0m[0m | time: 0.002s
    | Adam | epoch: 658 | loss: 0.00153 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 659  | total loss: [1m[32m0.00153[0m[0m | time: 0.002s
    | Adam | epoch: 659 | loss: 0.00153 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 660  | total loss: [1m[32m0.00152[0m[0m | time: 0.002s
    | Adam | epoch: 660 | loss: 0.00152 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 661  | total loss: [1m[32m0.00152[0m[0m | time: 0.002s
    | Adam | epoch: 661 | loss: 0.00152 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 662  | total loss: [1m[32m0.00151[0m[0m | time: 0.002s
    | Adam | epoch: 662 | loss: 0.00151 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 663  | total loss: [1m[32m0.00151[0m[0m | time: 0.002s
    | Adam | epoch: 663 | loss: 0.00151 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 664  | total loss: [1m[32m0.00151[0m[0m | time: 0.002s
    | Adam | epoch: 664 | loss: 0.00151 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 665  | total loss: [1m[32m0.00150[0m[0m | time: 0.002s
    | Adam | epoch: 665 | loss: 0.00150 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 666  | total loss: [1m[32m0.00150[0m[0m | time: 0.002s
    | Adam | epoch: 666 | loss: 0.00150 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 667  | total loss: [1m[32m0.00149[0m[0m | time: 0.002s
    | Adam | epoch: 667 | loss: 0.00149 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 668  | total loss: [1m[32m0.00149[0m[0m | time: 0.002s
    | Adam | epoch: 668 | loss: 0.00149 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 669  | total loss: [1m[32m0.00148[0m[0m | time: 0.002s
    | Adam | epoch: 669 | loss: 0.00148 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 670  | total loss: [1m[32m0.00148[0m[0m | time: 0.002s
    | Adam | epoch: 670 | loss: 0.00148 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 671  | total loss: [1m[32m0.00147[0m[0m | time: 0.002s
    | Adam | epoch: 671 | loss: 0.00147 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 672  | total loss: [1m[32m0.00147[0m[0m | time: 0.002s
    | Adam | epoch: 672 | loss: 0.00147 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 673  | total loss: [1m[32m0.00147[0m[0m | time: 0.002s
    | Adam | epoch: 673 | loss: 0.00147 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 674  | total loss: [1m[32m0.00146[0m[0m | time: 0.002s
    | Adam | epoch: 674 | loss: 0.00146 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 675  | total loss: [1m[32m0.00146[0m[0m | time: 0.002s
    | Adam | epoch: 675 | loss: 0.00146 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 676  | total loss: [1m[32m0.00145[0m[0m | time: 0.002s
    | Adam | epoch: 676 | loss: 0.00145 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 677  | total loss: [1m[32m0.00145[0m[0m | time: 0.002s
    | Adam | epoch: 677 | loss: 0.00145 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 678  | total loss: [1m[32m0.00144[0m[0m | time: 0.002s
    | Adam | epoch: 678 | loss: 0.00144 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 679  | total loss: [1m[32m0.00144[0m[0m | time: 0.002s
    | Adam | epoch: 679 | loss: 0.00144 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 680  | total loss: [1m[32m0.00144[0m[0m | time: 0.002s
    | Adam | epoch: 680 | loss: 0.00144 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 681  | total loss: [1m[32m0.00143[0m[0m | time: 0.002s
    | Adam | epoch: 681 | loss: 0.00143 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 682  | total loss: [1m[32m0.00143[0m[0m | time: 0.002s
    | Adam | epoch: 682 | loss: 0.00143 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 683  | total loss: [1m[32m0.00142[0m[0m | time: 0.002s
    | Adam | epoch: 683 | loss: 0.00142 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 684  | total loss: [1m[32m0.00142[0m[0m | time: 0.002s
    | Adam | epoch: 684 | loss: 0.00142 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 685  | total loss: [1m[32m0.00142[0m[0m | time: 0.002s
    | Adam | epoch: 685 | loss: 0.00142 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 686  | total loss: [1m[32m0.00141[0m[0m | time: 0.002s
    | Adam | epoch: 686 | loss: 0.00141 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 687  | total loss: [1m[32m0.00141[0m[0m | time: 0.002s
    | Adam | epoch: 687 | loss: 0.00141 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 688  | total loss: [1m[32m0.00140[0m[0m | time: 0.002s
    | Adam | epoch: 688 | loss: 0.00140 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 689  | total loss: [1m[32m0.00140[0m[0m | time: 0.002s
    | Adam | epoch: 689 | loss: 0.00140 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 690  | total loss: [1m[32m0.00140[0m[0m | time: 0.002s
    | Adam | epoch: 690 | loss: 0.00140 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 691  | total loss: [1m[32m0.00139[0m[0m | time: 0.003s
    | Adam | epoch: 691 | loss: 0.00139 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 692  | total loss: [1m[32m0.00139[0m[0m | time: 0.002s
    | Adam | epoch: 692 | loss: 0.00139 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 693  | total loss: [1m[32m0.00138[0m[0m | time: 0.002s
    | Adam | epoch: 693 | loss: 0.00138 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 694  | total loss: [1m[32m0.00138[0m[0m | time: 0.002s
    | Adam | epoch: 694 | loss: 0.00138 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 695  | total loss: [1m[32m0.00138[0m[0m | time: 0.003s
    | Adam | epoch: 695 | loss: 0.00138 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 696  | total loss: [1m[32m0.00137[0m[0m | time: 0.002s
    | Adam | epoch: 696 | loss: 0.00137 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 697  | total loss: [1m[32m0.00137[0m[0m | time: 0.003s
    | Adam | epoch: 697 | loss: 0.00137 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 698  | total loss: [1m[32m0.00136[0m[0m | time: 0.002s
    | Adam | epoch: 698 | loss: 0.00136 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 699  | total loss: [1m[32m0.00136[0m[0m | time: 0.002s
    | Adam | epoch: 699 | loss: 0.00136 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 700  | total loss: [1m[32m0.00136[0m[0m | time: 0.002s
    | Adam | epoch: 700 | loss: 0.00136 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 701  | total loss: [1m[32m0.00135[0m[0m | time: 0.003s
    | Adam | epoch: 701 | loss: 0.00135 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 702  | total loss: [1m[32m0.00135[0m[0m | time: 0.002s
    | Adam | epoch: 702 | loss: 0.00135 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 703  | total loss: [1m[32m0.00134[0m[0m | time: 0.002s
    | Adam | epoch: 703 | loss: 0.00134 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 704  | total loss: [1m[32m0.00134[0m[0m | time: 0.002s
    | Adam | epoch: 704 | loss: 0.00134 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 705  | total loss: [1m[32m0.00134[0m[0m | time: 0.002s
    | Adam | epoch: 705 | loss: 0.00134 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 706  | total loss: [1m[32m0.00133[0m[0m | time: 0.002s
    | Adam | epoch: 706 | loss: 0.00133 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 707  | total loss: [1m[32m0.00133[0m[0m | time: 0.002s
    | Adam | epoch: 707 | loss: 0.00133 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 708  | total loss: [1m[32m0.00133[0m[0m | time: 0.003s
    | Adam | epoch: 708 | loss: 0.00133 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 709  | total loss: [1m[32m0.00132[0m[0m | time: 0.002s
    | Adam | epoch: 709 | loss: 0.00132 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 710  | total loss: [1m[32m0.00132[0m[0m | time: 0.002s
    | Adam | epoch: 710 | loss: 0.00132 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 711  | total loss: [1m[32m0.00132[0m[0m | time: 0.002s
    | Adam | epoch: 711 | loss: 0.00132 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 712  | total loss: [1m[32m0.00131[0m[0m | time: 0.002s
    | Adam | epoch: 712 | loss: 0.00131 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 713  | total loss: [1m[32m0.00131[0m[0m | time: 0.002s
    | Adam | epoch: 713 | loss: 0.00131 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 714  | total loss: [1m[32m0.00130[0m[0m | time: 0.003s
    | Adam | epoch: 714 | loss: 0.00130 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 715  | total loss: [1m[32m0.00130[0m[0m | time: 0.002s
    | Adam | epoch: 715 | loss: 0.00130 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 716  | total loss: [1m[32m0.00130[0m[0m | time: 0.002s
    | Adam | epoch: 716 | loss: 0.00130 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 717  | total loss: [1m[32m0.00129[0m[0m | time: 0.002s
    | Adam | epoch: 717 | loss: 0.00129 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 718  | total loss: [1m[32m0.00129[0m[0m | time: 0.002s
    | Adam | epoch: 718 | loss: 0.00129 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 719  | total loss: [1m[32m0.00129[0m[0m | time: 0.002s
    | Adam | epoch: 719 | loss: 0.00129 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 720  | total loss: [1m[32m0.00128[0m[0m | time: 0.002s
    | Adam | epoch: 720 | loss: 0.00128 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 721  | total loss: [1m[32m0.00128[0m[0m | time: 0.002s
    | Adam | epoch: 721 | loss: 0.00128 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 722  | total loss: [1m[32m0.00128[0m[0m | time: 0.002s
    | Adam | epoch: 722 | loss: 0.00128 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 723  | total loss: [1m[32m0.00127[0m[0m | time: 0.002s
    | Adam | epoch: 723 | loss: 0.00127 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 724  | total loss: [1m[32m0.00127[0m[0m | time: 0.003s
    | Adam | epoch: 724 | loss: 0.00127 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 725  | total loss: [1m[32m0.00127[0m[0m | time: 0.003s
    | Adam | epoch: 725 | loss: 0.00127 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 726  | total loss: [1m[32m0.00126[0m[0m | time: 0.002s
    | Adam | epoch: 726 | loss: 0.00126 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 727  | total loss: [1m[32m0.00126[0m[0m | time: 0.002s
    | Adam | epoch: 727 | loss: 0.00126 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 728  | total loss: [1m[32m0.00126[0m[0m | time: 0.002s
    | Adam | epoch: 728 | loss: 0.00126 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 729  | total loss: [1m[32m0.00125[0m[0m | time: 0.002s
    | Adam | epoch: 729 | loss: 0.00125 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 730  | total loss: [1m[32m0.00125[0m[0m | time: 0.002s
    | Adam | epoch: 730 | loss: 0.00125 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 731  | total loss: [1m[32m0.00125[0m[0m | time: 0.002s
    | Adam | epoch: 731 | loss: 0.00125 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 732  | total loss: [1m[32m0.00124[0m[0m | time: 0.002s
    | Adam | epoch: 732 | loss: 0.00124 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 733  | total loss: [1m[32m0.00124[0m[0m | time: 0.001s
    | Adam | epoch: 733 | loss: 0.00124 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 734  | total loss: [1m[32m0.00124[0m[0m | time: 0.002s
    | Adam | epoch: 734 | loss: 0.00124 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 735  | total loss: [1m[32m0.00123[0m[0m | time: 0.002s
    | Adam | epoch: 735 | loss: 0.00123 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 736  | total loss: [1m[32m0.00123[0m[0m | time: 0.002s
    | Adam | epoch: 736 | loss: 0.00123 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 737  | total loss: [1m[32m0.00123[0m[0m | time: 0.002s
    | Adam | epoch: 737 | loss: 0.00123 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 738  | total loss: [1m[32m0.00122[0m[0m | time: 0.002s
    | Adam | epoch: 738 | loss: 0.00122 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 739  | total loss: [1m[32m0.00122[0m[0m | time: 0.002s
    | Adam | epoch: 739 | loss: 0.00122 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 740  | total loss: [1m[32m0.00122[0m[0m | time: 0.002s
    | Adam | epoch: 740 | loss: 0.00122 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 741  | total loss: [1m[32m0.00121[0m[0m | time: 0.002s
    | Adam | epoch: 741 | loss: 0.00121 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 742  | total loss: [1m[32m0.00121[0m[0m | time: 0.002s
    | Adam | epoch: 742 | loss: 0.00121 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 743  | total loss: [1m[32m0.00121[0m[0m | time: 0.002s
    | Adam | epoch: 743 | loss: 0.00121 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 744  | total loss: [1m[32m0.00120[0m[0m | time: 0.003s
    | Adam | epoch: 744 | loss: 0.00120 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 745  | total loss: [1m[32m0.00120[0m[0m | time: 0.002s
    | Adam | epoch: 745 | loss: 0.00120 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 746  | total loss: [1m[32m0.00120[0m[0m | time: 0.002s
    | Adam | epoch: 746 | loss: 0.00120 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 747  | total loss: [1m[32m0.00119[0m[0m | time: 0.002s
    | Adam | epoch: 747 | loss: 0.00119 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 748  | total loss: [1m[32m0.00119[0m[0m | time: 0.002s
    | Adam | epoch: 748 | loss: 0.00119 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 749  | total loss: [1m[32m0.00119[0m[0m | time: 0.002s
    | Adam | epoch: 749 | loss: 0.00119 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 750  | total loss: [1m[32m0.00118[0m[0m | time: 0.002s
    | Adam | epoch: 750 | loss: 0.00118 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 751  | total loss: [1m[32m0.00118[0m[0m | time: 0.002s
    | Adam | epoch: 751 | loss: 0.00118 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 752  | total loss: [1m[32m0.00118[0m[0m | time: 0.002s
    | Adam | epoch: 752 | loss: 0.00118 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 753  | total loss: [1m[32m0.00117[0m[0m | time: 0.001s
    | Adam | epoch: 753 | loss: 0.00117 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 754  | total loss: [1m[32m0.00117[0m[0m | time: 0.002s
    | Adam | epoch: 754 | loss: 0.00117 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 755  | total loss: [1m[32m0.00117[0m[0m | time: 0.002s
    | Adam | epoch: 755 | loss: 0.00117 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 756  | total loss: [1m[32m0.00117[0m[0m | time: 0.002s
    | Adam | epoch: 756 | loss: 0.00117 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 757  | total loss: [1m[32m0.00116[0m[0m | time: 0.002s
    | Adam | epoch: 757 | loss: 0.00116 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 758  | total loss: [1m[32m0.00116[0m[0m | time: 0.002s
    | Adam | epoch: 758 | loss: 0.00116 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 759  | total loss: [1m[32m0.00116[0m[0m | time: 0.002s
    | Adam | epoch: 759 | loss: 0.00116 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 760  | total loss: [1m[32m0.00115[0m[0m | time: 0.002s
    | Adam | epoch: 760 | loss: 0.00115 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 761  | total loss: [1m[32m0.00115[0m[0m | time: 0.002s
    | Adam | epoch: 761 | loss: 0.00115 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 762  | total loss: [1m[32m0.00115[0m[0m | time: 0.002s
    | Adam | epoch: 762 | loss: 0.00115 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 763  | total loss: [1m[32m0.00114[0m[0m | time: 0.002s
    | Adam | epoch: 763 | loss: 0.00114 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 764  | total loss: [1m[32m0.00114[0m[0m | time: 0.003s
    | Adam | epoch: 764 | loss: 0.00114 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 765  | total loss: [1m[32m0.00114[0m[0m | time: 0.002s
    | Adam | epoch: 765 | loss: 0.00114 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 766  | total loss: [1m[32m0.00114[0m[0m | time: 0.002s
    | Adam | epoch: 766 | loss: 0.00114 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 767  | total loss: [1m[32m0.00113[0m[0m | time: 0.002s
    | Adam | epoch: 767 | loss: 0.00113 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 768  | total loss: [1m[32m0.00113[0m[0m | time: 0.002s
    | Adam | epoch: 768 | loss: 0.00113 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 769  | total loss: [1m[32m0.00113[0m[0m | time: 0.002s
    | Adam | epoch: 769 | loss: 0.00113 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 770  | total loss: [1m[32m0.00112[0m[0m | time: 0.002s
    | Adam | epoch: 770 | loss: 0.00112 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 771  | total loss: [1m[32m0.00112[0m[0m | time: 0.002s
    | Adam | epoch: 771 | loss: 0.00112 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 772  | total loss: [1m[32m0.00112[0m[0m | time: 0.002s
    | Adam | epoch: 772 | loss: 0.00112 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 773  | total loss: [1m[32m0.00112[0m[0m | time: 0.002s
    | Adam | epoch: 773 | loss: 0.00112 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 774  | total loss: [1m[32m0.00111[0m[0m | time: 0.003s
    | Adam | epoch: 774 | loss: 0.00111 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 775  | total loss: [1m[32m0.00111[0m[0m | time: 0.002s
    | Adam | epoch: 775 | loss: 0.00111 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 776  | total loss: [1m[32m0.00111[0m[0m | time: 0.002s
    | Adam | epoch: 776 | loss: 0.00111 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 777  | total loss: [1m[32m0.00110[0m[0m | time: 0.003s
    | Adam | epoch: 777 | loss: 0.00110 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 778  | total loss: [1m[32m0.00110[0m[0m | time: 0.003s
    | Adam | epoch: 778 | loss: 0.00110 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 779  | total loss: [1m[32m0.00110[0m[0m | time: 0.002s
    | Adam | epoch: 779 | loss: 0.00110 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 780  | total loss: [1m[32m0.00110[0m[0m | time: 0.002s
    | Adam | epoch: 780 | loss: 0.00110 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 781  | total loss: [1m[32m0.00109[0m[0m | time: 0.002s
    | Adam | epoch: 781 | loss: 0.00109 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 782  | total loss: [1m[32m0.00109[0m[0m | time: 0.002s
    | Adam | epoch: 782 | loss: 0.00109 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 783  | total loss: [1m[32m0.00109[0m[0m | time: 0.002s
    | Adam | epoch: 783 | loss: 0.00109 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 784  | total loss: [1m[32m0.00108[0m[0m | time: 0.002s
    | Adam | epoch: 784 | loss: 0.00108 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 785  | total loss: [1m[32m0.00108[0m[0m | time: 0.002s
    | Adam | epoch: 785 | loss: 0.00108 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 786  | total loss: [1m[32m0.00108[0m[0m | time: 0.002s
    | Adam | epoch: 786 | loss: 0.00108 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 787  | total loss: [1m[32m0.00108[0m[0m | time: 0.002s
    | Adam | epoch: 787 | loss: 0.00108 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 788  | total loss: [1m[32m0.00107[0m[0m | time: 0.002s
    | Adam | epoch: 788 | loss: 0.00107 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 789  | total loss: [1m[32m0.00107[0m[0m | time: 0.003s
    | Adam | epoch: 789 | loss: 0.00107 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 790  | total loss: [1m[32m0.00107[0m[0m | time: 0.002s
    | Adam | epoch: 790 | loss: 0.00107 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 791  | total loss: [1m[32m0.00107[0m[0m | time: 0.002s
    | Adam | epoch: 791 | loss: 0.00107 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 792  | total loss: [1m[32m0.00106[0m[0m | time: 0.002s
    | Adam | epoch: 792 | loss: 0.00106 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 793  | total loss: [1m[32m0.00106[0m[0m | time: 0.002s
    | Adam | epoch: 793 | loss: 0.00106 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 794  | total loss: [1m[32m0.00106[0m[0m | time: 0.002s
    | Adam | epoch: 794 | loss: 0.00106 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 795  | total loss: [1m[32m0.00105[0m[0m | time: 0.002s
    | Adam | epoch: 795 | loss: 0.00105 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 796  | total loss: [1m[32m0.00105[0m[0m | time: 0.003s
    | Adam | epoch: 796 | loss: 0.00105 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 797  | total loss: [1m[32m0.00105[0m[0m | time: 0.003s
    | Adam | epoch: 797 | loss: 0.00105 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 798  | total loss: [1m[32m0.00105[0m[0m | time: 0.002s
    | Adam | epoch: 798 | loss: 0.00105 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 799  | total loss: [1m[32m0.00104[0m[0m | time: 0.002s
    | Adam | epoch: 799 | loss: 0.00104 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 800  | total loss: [1m[32m0.00104[0m[0m | time: 0.003s
    | Adam | epoch: 800 | loss: 0.00104 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 801  | total loss: [1m[32m0.00104[0m[0m | time: 0.002s
    | Adam | epoch: 801 | loss: 0.00104 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 802  | total loss: [1m[32m0.00104[0m[0m | time: 0.003s
    | Adam | epoch: 802 | loss: 0.00104 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 803  | total loss: [1m[32m0.00103[0m[0m | time: 0.002s
    | Adam | epoch: 803 | loss: 0.00103 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 804  | total loss: [1m[32m0.00103[0m[0m | time: 0.002s
    | Adam | epoch: 804 | loss: 0.00103 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 805  | total loss: [1m[32m0.00103[0m[0m | time: 0.002s
    | Adam | epoch: 805 | loss: 0.00103 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 806  | total loss: [1m[32m0.00103[0m[0m | time: 0.003s
    | Adam | epoch: 806 | loss: 0.00103 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 807  | total loss: [1m[32m0.00102[0m[0m | time: 0.002s
    | Adam | epoch: 807 | loss: 0.00102 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 808  | total loss: [1m[32m0.00102[0m[0m | time: 0.002s
    | Adam | epoch: 808 | loss: 0.00102 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 809  | total loss: [1m[32m0.00102[0m[0m | time: 0.002s
    | Adam | epoch: 809 | loss: 0.00102 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 810  | total loss: [1m[32m0.00102[0m[0m | time: 0.002s
    | Adam | epoch: 810 | loss: 0.00102 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 811  | total loss: [1m[32m0.00101[0m[0m | time: 0.002s
    | Adam | epoch: 811 | loss: 0.00101 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 812  | total loss: [1m[32m0.00101[0m[0m | time: 0.003s
    | Adam | epoch: 812 | loss: 0.00101 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 813  | total loss: [1m[32m0.00101[0m[0m | time: 0.002s
    | Adam | epoch: 813 | loss: 0.00101 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 814  | total loss: [1m[32m0.00101[0m[0m | time: 0.002s
    | Adam | epoch: 814 | loss: 0.00101 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 815  | total loss: [1m[32m0.00100[0m[0m | time: 0.002s
    | Adam | epoch: 815 | loss: 0.00100 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 816  | total loss: [1m[32m0.00100[0m[0m | time: 0.002s
    | Adam | epoch: 816 | loss: 0.00100 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 817  | total loss: [1m[32m0.00100[0m[0m | time: 0.002s
    | Adam | epoch: 817 | loss: 0.00100 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 818  | total loss: [1m[32m0.00100[0m[0m | time: 0.002s
    | Adam | epoch: 818 | loss: 0.00100 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 819  | total loss: [1m[32m0.00099[0m[0m | time: 0.002s
    | Adam | epoch: 819 | loss: 0.00099 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 820  | total loss: [1m[32m0.00099[0m[0m | time: 0.002s
    | Adam | epoch: 820 | loss: 0.00099 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 821  | total loss: [1m[32m0.00099[0m[0m | time: 0.002s
    | Adam | epoch: 821 | loss: 0.00099 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 822  | total loss: [1m[32m0.00099[0m[0m | time: 0.002s
    | Adam | epoch: 822 | loss: 0.00099 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 823  | total loss: [1m[32m0.00099[0m[0m | time: 0.002s
    | Adam | epoch: 823 | loss: 0.00099 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 824  | total loss: [1m[32m0.00098[0m[0m | time: 0.002s
    | Adam | epoch: 824 | loss: 0.00098 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 825  | total loss: [1m[32m0.00098[0m[0m | time: 0.002s
    | Adam | epoch: 825 | loss: 0.00098 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 826  | total loss: [1m[32m0.00098[0m[0m | time: 0.002s
    | Adam | epoch: 826 | loss: 0.00098 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 827  | total loss: [1m[32m0.00098[0m[0m | time: 0.002s
    | Adam | epoch: 827 | loss: 0.00098 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 828  | total loss: [1m[32m0.00097[0m[0m | time: 0.002s
    | Adam | epoch: 828 | loss: 0.00097 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 829  | total loss: [1m[32m0.00097[0m[0m | time: 0.002s
    | Adam | epoch: 829 | loss: 0.00097 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 830  | total loss: [1m[32m0.00097[0m[0m | time: 0.003s
    | Adam | epoch: 830 | loss: 0.00097 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 831  | total loss: [1m[32m0.00097[0m[0m | time: 0.002s
    | Adam | epoch: 831 | loss: 0.00097 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 832  | total loss: [1m[32m0.00096[0m[0m | time: 0.003s
    | Adam | epoch: 832 | loss: 0.00096 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 833  | total loss: [1m[32m0.00096[0m[0m | time: 0.002s
    | Adam | epoch: 833 | loss: 0.00096 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 834  | total loss: [1m[32m0.00096[0m[0m | time: 0.002s
    | Adam | epoch: 834 | loss: 0.00096 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 835  | total loss: [1m[32m0.00096[0m[0m | time: 0.002s
    | Adam | epoch: 835 | loss: 0.00096 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 836  | total loss: [1m[32m0.00096[0m[0m | time: 0.003s
    | Adam | epoch: 836 | loss: 0.00096 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 837  | total loss: [1m[32m0.00095[0m[0m | time: 0.002s
    | Adam | epoch: 837 | loss: 0.00095 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 838  | total loss: [1m[32m0.00095[0m[0m | time: 0.002s
    | Adam | epoch: 838 | loss: 0.00095 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 839  | total loss: [1m[32m0.00095[0m[0m | time: 0.002s
    | Adam | epoch: 839 | loss: 0.00095 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 840  | total loss: [1m[32m0.00095[0m[0m | time: 0.002s
    | Adam | epoch: 840 | loss: 0.00095 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 841  | total loss: [1m[32m0.00094[0m[0m | time: 0.002s
    | Adam | epoch: 841 | loss: 0.00094 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 842  | total loss: [1m[32m0.00094[0m[0m | time: 0.003s
    | Adam | epoch: 842 | loss: 0.00094 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 843  | total loss: [1m[32m0.00094[0m[0m | time: 0.002s
    | Adam | epoch: 843 | loss: 0.00094 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 844  | total loss: [1m[32m0.00094[0m[0m | time: 0.002s
    | Adam | epoch: 844 | loss: 0.00094 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 845  | total loss: [1m[32m0.00094[0m[0m | time: 0.002s
    | Adam | epoch: 845 | loss: 0.00094 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 846  | total loss: [1m[32m0.00093[0m[0m | time: 0.002s
    | Adam | epoch: 846 | loss: 0.00093 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 847  | total loss: [1m[32m0.00093[0m[0m | time: 0.002s
    | Adam | epoch: 847 | loss: 0.00093 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 848  | total loss: [1m[32m0.00093[0m[0m | time: 0.002s
    | Adam | epoch: 848 | loss: 0.00093 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 849  | total loss: [1m[32m0.00093[0m[0m | time: 0.002s
    | Adam | epoch: 849 | loss: 0.00093 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 850  | total loss: [1m[32m0.00092[0m[0m | time: 0.002s
    | Adam | epoch: 850 | loss: 0.00092 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 851  | total loss: [1m[32m0.00092[0m[0m | time: 0.002s
    | Adam | epoch: 851 | loss: 0.00092 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 852  | total loss: [1m[32m0.00092[0m[0m | time: 0.002s
    | Adam | epoch: 852 | loss: 0.00092 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 853  | total loss: [1m[32m0.00092[0m[0m | time: 0.002s
    | Adam | epoch: 853 | loss: 0.00092 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 854  | total loss: [1m[32m0.00092[0m[0m | time: 0.002s
    | Adam | epoch: 854 | loss: 0.00092 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 855  | total loss: [1m[32m0.00091[0m[0m | time: 0.002s
    | Adam | epoch: 855 | loss: 0.00091 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 856  | total loss: [1m[32m0.00091[0m[0m | time: 0.002s
    | Adam | epoch: 856 | loss: 0.00091 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 857  | total loss: [1m[32m0.00091[0m[0m | time: 0.002s
    | Adam | epoch: 857 | loss: 0.00091 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 858  | total loss: [1m[32m0.00091[0m[0m | time: 0.002s
    | Adam | epoch: 858 | loss: 0.00091 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 859  | total loss: [1m[32m0.00091[0m[0m | time: 0.002s
    | Adam | epoch: 859 | loss: 0.00091 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 860  | total loss: [1m[32m0.00090[0m[0m | time: 0.002s
    | Adam | epoch: 860 | loss: 0.00090 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 861  | total loss: [1m[32m0.00090[0m[0m | time: 0.002s
    | Adam | epoch: 861 | loss: 0.00090 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 862  | total loss: [1m[32m0.00090[0m[0m | time: 0.002s
    | Adam | epoch: 862 | loss: 0.00090 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 863  | total loss: [1m[32m0.00090[0m[0m | time: 0.002s
    | Adam | epoch: 863 | loss: 0.00090 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 864  | total loss: [1m[32m0.00090[0m[0m | time: 0.002s
    | Adam | epoch: 864 | loss: 0.00090 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 865  | total loss: [1m[32m0.00089[0m[0m | time: 0.002s
    | Adam | epoch: 865 | loss: 0.00089 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 866  | total loss: [1m[32m0.00089[0m[0m | time: 0.002s
    | Adam | epoch: 866 | loss: 0.00089 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 867  | total loss: [1m[32m0.00089[0m[0m | time: 0.002s
    | Adam | epoch: 867 | loss: 0.00089 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 868  | total loss: [1m[32m0.00089[0m[0m | time: 0.003s
    | Adam | epoch: 868 | loss: 0.00089 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 869  | total loss: [1m[32m0.00088[0m[0m | time: 0.003s
    | Adam | epoch: 869 | loss: 0.00088 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 870  | total loss: [1m[32m0.00088[0m[0m | time: 0.002s
    | Adam | epoch: 870 | loss: 0.00088 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 871  | total loss: [1m[32m0.00088[0m[0m | time: 0.002s
    | Adam | epoch: 871 | loss: 0.00088 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 872  | total loss: [1m[32m0.00088[0m[0m | time: 0.002s
    | Adam | epoch: 872 | loss: 0.00088 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 873  | total loss: [1m[32m0.00088[0m[0m | time: 0.002s
    | Adam | epoch: 873 | loss: 0.00088 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 874  | total loss: [1m[32m0.00088[0m[0m | time: 0.002s
    | Adam | epoch: 874 | loss: 0.00088 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 875  | total loss: [1m[32m0.00087[0m[0m | time: 0.002s
    | Adam | epoch: 875 | loss: 0.00087 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 876  | total loss: [1m[32m0.00087[0m[0m | time: 0.002s
    | Adam | epoch: 876 | loss: 0.00087 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 877  | total loss: [1m[32m0.00087[0m[0m | time: 0.002s
    | Adam | epoch: 877 | loss: 0.00087 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 878  | total loss: [1m[32m0.00087[0m[0m | time: 0.002s
    | Adam | epoch: 878 | loss: 0.00087 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 879  | total loss: [1m[32m0.00087[0m[0m | time: 0.012s
    | Adam | epoch: 879 | loss: 0.00087 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 880  | total loss: [1m[32m0.00086[0m[0m | time: 0.003s
    | Adam | epoch: 880 | loss: 0.00086 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 881  | total loss: [1m[32m0.00086[0m[0m | time: 0.002s
    | Adam | epoch: 881 | loss: 0.00086 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 882  | total loss: [1m[32m0.00086[0m[0m | time: 0.007s
    | Adam | epoch: 882 | loss: 0.00086 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 883  | total loss: [1m[32m0.00086[0m[0m | time: 0.009s
    | Adam | epoch: 883 | loss: 0.00086 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 884  | total loss: [1m[32m0.00086[0m[0m | time: 0.003s
    | Adam | epoch: 884 | loss: 0.00086 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 885  | total loss: [1m[32m0.00085[0m[0m | time: 0.002s
    | Adam | epoch: 885 | loss: 0.00085 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 886  | total loss: [1m[32m0.00085[0m[0m | time: 0.003s
    | Adam | epoch: 886 | loss: 0.00085 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 887  | total loss: [1m[32m0.00085[0m[0m | time: 0.002s
    | Adam | epoch: 887 | loss: 0.00085 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 888  | total loss: [1m[32m0.00085[0m[0m | time: 0.004s
    | Adam | epoch: 888 | loss: 0.00085 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 889  | total loss: [1m[32m0.00085[0m[0m | time: 0.003s
    | Adam | epoch: 889 | loss: 0.00085 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 890  | total loss: [1m[32m0.00084[0m[0m | time: 0.002s
    | Adam | epoch: 890 | loss: 0.00084 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 891  | total loss: [1m[32m0.00084[0m[0m | time: 0.002s
    | Adam | epoch: 891 | loss: 0.00084 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 892  | total loss: [1m[32m0.00084[0m[0m | time: 0.002s
    | Adam | epoch: 892 | loss: 0.00084 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 893  | total loss: [1m[32m0.00084[0m[0m | time: 0.002s
    | Adam | epoch: 893 | loss: 0.00084 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 894  | total loss: [1m[32m0.00084[0m[0m | time: 0.002s
    | Adam | epoch: 894 | loss: 0.00084 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 895  | total loss: [1m[32m0.00083[0m[0m | time: 0.002s
    | Adam | epoch: 895 | loss: 0.00083 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 896  | total loss: [1m[32m0.00083[0m[0m | time: 0.002s
    | Adam | epoch: 896 | loss: 0.00083 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 897  | total loss: [1m[32m0.00083[0m[0m | time: 0.001s
    | Adam | epoch: 897 | loss: 0.00083 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 898  | total loss: [1m[32m0.00083[0m[0m | time: 0.002s
    | Adam | epoch: 898 | loss: 0.00083 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 899  | total loss: [1m[32m0.00083[0m[0m | time: 0.002s
    | Adam | epoch: 899 | loss: 0.00083 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 900  | total loss: [1m[32m0.00083[0m[0m | time: 0.002s
    | Adam | epoch: 900 | loss: 0.00083 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 901  | total loss: [1m[32m0.00082[0m[0m | time: 0.002s
    | Adam | epoch: 901 | loss: 0.00082 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 902  | total loss: [1m[32m0.00082[0m[0m | time: 0.002s
    | Adam | epoch: 902 | loss: 0.00082 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 903  | total loss: [1m[32m0.00082[0m[0m | time: 0.002s
    | Adam | epoch: 903 | loss: 0.00082 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 904  | total loss: [1m[32m0.00082[0m[0m | time: 0.002s
    | Adam | epoch: 904 | loss: 0.00082 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 905  | total loss: [1m[32m0.00082[0m[0m | time: 0.002s
    | Adam | epoch: 905 | loss: 0.00082 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 906  | total loss: [1m[32m0.00082[0m[0m | time: 0.002s
    | Adam | epoch: 906 | loss: 0.00082 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 907  | total loss: [1m[32m0.00081[0m[0m | time: 0.002s
    | Adam | epoch: 907 | loss: 0.00081 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 908  | total loss: [1m[32m0.00081[0m[0m | time: 0.002s
    | Adam | epoch: 908 | loss: 0.00081 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 909  | total loss: [1m[32m0.00081[0m[0m | time: 0.002s
    | Adam | epoch: 909 | loss: 0.00081 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 910  | total loss: [1m[32m0.00081[0m[0m | time: 0.002s
    | Adam | epoch: 910 | loss: 0.00081 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 911  | total loss: [1m[32m0.00081[0m[0m | time: 0.002s
    | Adam | epoch: 911 | loss: 0.00081 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 912  | total loss: [1m[32m0.00080[0m[0m | time: 0.002s
    | Adam | epoch: 912 | loss: 0.00080 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 913  | total loss: [1m[32m0.00080[0m[0m | time: 0.001s
    | Adam | epoch: 913 | loss: 0.00080 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 914  | total loss: [1m[32m0.00080[0m[0m | time: 0.002s
    | Adam | epoch: 914 | loss: 0.00080 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 915  | total loss: [1m[32m0.00080[0m[0m | time: 0.002s
    | Adam | epoch: 915 | loss: 0.00080 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 916  | total loss: [1m[32m0.00080[0m[0m | time: 0.002s
    | Adam | epoch: 916 | loss: 0.00080 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 917  | total loss: [1m[32m0.00080[0m[0m | time: 0.002s
    | Adam | epoch: 917 | loss: 0.00080 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 918  | total loss: [1m[32m0.00079[0m[0m | time: 0.003s
    | Adam | epoch: 918 | loss: 0.00079 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 919  | total loss: [1m[32m0.00079[0m[0m | time: 0.002s
    | Adam | epoch: 919 | loss: 0.00079 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 920  | total loss: [1m[32m0.00079[0m[0m | time: 0.002s
    | Adam | epoch: 920 | loss: 0.00079 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 921  | total loss: [1m[32m0.00079[0m[0m | time: 0.002s
    | Adam | epoch: 921 | loss: 0.00079 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 922  | total loss: [1m[32m0.00079[0m[0m | time: 0.002s
    | Adam | epoch: 922 | loss: 0.00079 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 923  | total loss: [1m[32m0.00079[0m[0m | time: 0.002s
    | Adam | epoch: 923 | loss: 0.00079 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 924  | total loss: [1m[32m0.00078[0m[0m | time: 0.003s
    | Adam | epoch: 924 | loss: 0.00078 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 925  | total loss: [1m[32m0.00078[0m[0m | time: 0.002s
    | Adam | epoch: 925 | loss: 0.00078 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 926  | total loss: [1m[32m0.00078[0m[0m | time: 0.002s
    | Adam | epoch: 926 | loss: 0.00078 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 927  | total loss: [1m[32m0.00078[0m[0m | time: 0.002s
    | Adam | epoch: 927 | loss: 0.00078 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 928  | total loss: [1m[32m0.00078[0m[0m | time: 0.002s
    | Adam | epoch: 928 | loss: 0.00078 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 929  | total loss: [1m[32m0.00078[0m[0m | time: 0.002s
    | Adam | epoch: 929 | loss: 0.00078 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 930  | total loss: [1m[32m0.00077[0m[0m | time: 0.003s
    | Adam | epoch: 930 | loss: 0.00077 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 931  | total loss: [1m[32m0.00077[0m[0m | time: 0.002s
    | Adam | epoch: 931 | loss: 0.00077 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 932  | total loss: [1m[32m0.00077[0m[0m | time: 0.002s
    | Adam | epoch: 932 | loss: 0.00077 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 933  | total loss: [1m[32m0.00077[0m[0m | time: 0.003s
    | Adam | epoch: 933 | loss: 0.00077 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 934  | total loss: [1m[32m0.00077[0m[0m | time: 0.002s
    | Adam | epoch: 934 | loss: 0.00077 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 935  | total loss: [1m[32m0.00077[0m[0m | time: 0.002s
    | Adam | epoch: 935 | loss: 0.00077 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 936  | total loss: [1m[32m0.00076[0m[0m | time: 0.002s
    | Adam | epoch: 936 | loss: 0.00076 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 937  | total loss: [1m[32m0.00076[0m[0m | time: 0.002s
    | Adam | epoch: 937 | loss: 0.00076 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 938  | total loss: [1m[32m0.00076[0m[0m | time: 0.003s
    | Adam | epoch: 938 | loss: 0.00076 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 939  | total loss: [1m[32m0.00076[0m[0m | time: 0.003s
    | Adam | epoch: 939 | loss: 0.00076 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 940  | total loss: [1m[32m0.00076[0m[0m | time: 0.002s
    | Adam | epoch: 940 | loss: 0.00076 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 941  | total loss: [1m[32m0.00076[0m[0m | time: 0.003s
    | Adam | epoch: 941 | loss: 0.00076 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 942  | total loss: [1m[32m0.00075[0m[0m | time: 0.002s
    | Adam | epoch: 942 | loss: 0.00075 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 943  | total loss: [1m[32m0.00075[0m[0m | time: 0.002s
    | Adam | epoch: 943 | loss: 0.00075 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 944  | total loss: [1m[32m0.00075[0m[0m | time: 0.002s
    | Adam | epoch: 944 | loss: 0.00075 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 945  | total loss: [1m[32m0.00075[0m[0m | time: 0.002s
    | Adam | epoch: 945 | loss: 0.00075 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 946  | total loss: [1m[32m0.00075[0m[0m | time: 0.002s
    | Adam | epoch: 946 | loss: 0.00075 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 947  | total loss: [1m[32m0.00075[0m[0m | time: 0.002s
    | Adam | epoch: 947 | loss: 0.00075 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 948  | total loss: [1m[32m0.00075[0m[0m | time: 0.002s
    | Adam | epoch: 948 | loss: 0.00075 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 949  | total loss: [1m[32m0.00074[0m[0m | time: 0.002s
    | Adam | epoch: 949 | loss: 0.00074 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 950  | total loss: [1m[32m0.00074[0m[0m | time: 0.002s
    | Adam | epoch: 950 | loss: 0.00074 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 951  | total loss: [1m[32m0.00074[0m[0m | time: 0.002s
    | Adam | epoch: 951 | loss: 0.00074 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 952  | total loss: [1m[32m0.00074[0m[0m | time: 0.002s
    | Adam | epoch: 952 | loss: 0.00074 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 953  | total loss: [1m[32m0.00074[0m[0m | time: 0.002s
    | Adam | epoch: 953 | loss: 0.00074 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 954  | total loss: [1m[32m0.00074[0m[0m | time: 0.002s
    | Adam | epoch: 954 | loss: 0.00074 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 955  | total loss: [1m[32m0.00073[0m[0m | time: 0.003s
    | Adam | epoch: 955 | loss: 0.00073 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 956  | total loss: [1m[32m0.00073[0m[0m | time: 0.003s
    | Adam | epoch: 956 | loss: 0.00073 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 957  | total loss: [1m[32m0.00073[0m[0m | time: 0.003s
    | Adam | epoch: 957 | loss: 0.00073 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 958  | total loss: [1m[32m0.00073[0m[0m | time: 0.002s
    | Adam | epoch: 958 | loss: 0.00073 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 959  | total loss: [1m[32m0.00073[0m[0m | time: 0.002s
    | Adam | epoch: 959 | loss: 0.00073 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 960  | total loss: [1m[32m0.00073[0m[0m | time: 0.002s
    | Adam | epoch: 960 | loss: 0.00073 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 961  | total loss: [1m[32m0.00073[0m[0m | time: 0.002s
    | Adam | epoch: 961 | loss: 0.00073 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 962  | total loss: [1m[32m0.00072[0m[0m | time: 0.002s
    | Adam | epoch: 962 | loss: 0.00072 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 963  | total loss: [1m[32m0.00072[0m[0m | time: 0.002s
    | Adam | epoch: 963 | loss: 0.00072 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 964  | total loss: [1m[32m0.00072[0m[0m | time: 0.003s
    | Adam | epoch: 964 | loss: 0.00072 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 965  | total loss: [1m[32m0.00072[0m[0m | time: 0.002s
    | Adam | epoch: 965 | loss: 0.00072 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 966  | total loss: [1m[32m0.00072[0m[0m | time: 0.002s
    | Adam | epoch: 966 | loss: 0.00072 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 967  | total loss: [1m[32m0.00072[0m[0m | time: 0.002s
    | Adam | epoch: 967 | loss: 0.00072 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 968  | total loss: [1m[32m0.00071[0m[0m | time: 0.003s
    | Adam | epoch: 968 | loss: 0.00071 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 969  | total loss: [1m[32m0.00071[0m[0m | time: 0.002s
    | Adam | epoch: 969 | loss: 0.00071 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 970  | total loss: [1m[32m0.00071[0m[0m | time: 0.002s
    | Adam | epoch: 970 | loss: 0.00071 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 971  | total loss: [1m[32m0.00071[0m[0m | time: 0.002s
    | Adam | epoch: 971 | loss: 0.00071 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 972  | total loss: [1m[32m0.00071[0m[0m | time: 0.002s
    | Adam | epoch: 972 | loss: 0.00071 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 973  | total loss: [1m[32m0.00071[0m[0m | time: 0.002s
    | Adam | epoch: 973 | loss: 0.00071 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 974  | total loss: [1m[32m0.00071[0m[0m | time: 0.002s
    | Adam | epoch: 974 | loss: 0.00071 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 975  | total loss: [1m[32m0.00070[0m[0m | time: 0.002s
    | Adam | epoch: 975 | loss: 0.00070 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 976  | total loss: [1m[32m0.00070[0m[0m | time: 0.002s
    | Adam | epoch: 976 | loss: 0.00070 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 977  | total loss: [1m[32m0.00070[0m[0m | time: 0.002s
    | Adam | epoch: 977 | loss: 0.00070 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 978  | total loss: [1m[32m0.00070[0m[0m | time: 0.002s
    | Adam | epoch: 978 | loss: 0.00070 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 979  | total loss: [1m[32m0.00070[0m[0m | time: 0.002s
    | Adam | epoch: 979 | loss: 0.00070 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 980  | total loss: [1m[32m0.00070[0m[0m | time: 0.002s
    | Adam | epoch: 980 | loss: 0.00070 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 981  | total loss: [1m[32m0.00070[0m[0m | time: 0.002s
    | Adam | epoch: 981 | loss: 0.00070 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 982  | total loss: [1m[32m0.00069[0m[0m | time: 0.003s
    | Adam | epoch: 982 | loss: 0.00069 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 983  | total loss: [1m[32m0.00069[0m[0m | time: 0.002s
    | Adam | epoch: 983 | loss: 0.00069 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 984  | total loss: [1m[32m0.00069[0m[0m | time: 0.002s
    | Adam | epoch: 984 | loss: 0.00069 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 985  | total loss: [1m[32m0.00069[0m[0m | time: 0.002s
    | Adam | epoch: 985 | loss: 0.00069 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 986  | total loss: [1m[32m0.00069[0m[0m | time: 0.003s
    | Adam | epoch: 986 | loss: 0.00069 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 987  | total loss: [1m[32m0.00069[0m[0m | time: 0.002s
    | Adam | epoch: 987 | loss: 0.00069 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 988  | total loss: [1m[32m0.00069[0m[0m | time: 0.003s
    | Adam | epoch: 988 | loss: 0.00069 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 989  | total loss: [1m[32m0.00069[0m[0m | time: 0.002s
    | Adam | epoch: 989 | loss: 0.00069 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 990  | total loss: [1m[32m0.00068[0m[0m | time: 0.002s
    | Adam | epoch: 990 | loss: 0.00068 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 991  | total loss: [1m[32m0.00068[0m[0m | time: 0.002s
    | Adam | epoch: 991 | loss: 0.00068 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 992  | total loss: [1m[32m0.00068[0m[0m | time: 0.002s
    | Adam | epoch: 992 | loss: 0.00068 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 993  | total loss: [1m[32m0.00068[0m[0m | time: 0.002s
    | Adam | epoch: 993 | loss: 0.00068 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 994  | total loss: [1m[32m0.00068[0m[0m | time: 0.002s
    | Adam | epoch: 994 | loss: 0.00068 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 995  | total loss: [1m[32m0.00068[0m[0m | time: 0.002s
    | Adam | epoch: 995 | loss: 0.00068 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 996  | total loss: [1m[32m0.00068[0m[0m | time: 0.002s
    | Adam | epoch: 996 | loss: 0.00068 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 997  | total loss: [1m[32m0.00067[0m[0m | time: 0.002s
    | Adam | epoch: 997 | loss: 0.00067 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 998  | total loss: [1m[32m0.00067[0m[0m | time: 0.003s
    | Adam | epoch: 998 | loss: 0.00067 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 999  | total loss: [1m[32m0.00067[0m[0m | time: 0.002s
    | Adam | epoch: 999 | loss: 0.00067 - acc: 1.0000 -- iter: 1/1
    --
    Training Step: 1000  | total loss: [1m[32m0.00067[0m[0m | time: 0.002s
    | Adam | epoch: 1000 | loss: 0.00067 - acc: 1.0000 -- iter: 1/1
    --
    INFO:tensorflow:/Users/stephaniefissel/model.tflearn is not in all_model_checkpoint_paths. Manually adding it.
    INFO:tensorflow:Restoring parameters from /Users/stephaniefissel/model.tflearn
    Start talking with the bot! (type quit to stop)



```python

```
