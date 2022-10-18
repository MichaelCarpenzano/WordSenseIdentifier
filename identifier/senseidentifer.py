#!/usr/bin/python3
# Michael Carpenzano
# CSE354, Spring 2021
##########################################################

import sys
import gzip
#import json #for reading json encoded files into opjects
import re #regular expressions
import numpy as np
import torch
import torch.nn as nn  #pytorch
import csv
from collections import Counter
import itertools

# Comment this line out if you wish to see results on the console
#sys.stdout = open('senseidentifier_OUTPUT.txt', 'w', encoding = 'utf8') # EDIT THIS

#########################################################
## Part 1.1 Read the data (15 pts).
def load_data(filename:str, REMOVE_LEMMAPOS = True): # return data:dict and wordCounts:dict
    #split each line
    tsv_file = open(filename, 'r', encoding='utf8')
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    # load records into dictionaries, create a dictionary where each class holds a list of dicts
    records = {
        "process": [],
        "machine": [],
        "language": []
    }
    processRE = re.compile(r"process")
    machineRE = re.compile(r"machine")
    languageRE = re.compile(r"language")
    for record in list(read_tsv):
        if not processRE.match(record[0]) is None:
            # create dictionary
            entry = {}
            # split line on tabs, add element 0 to entry as lemma.POS.id, 1 as sense, 2 as context
            elements = record
            entry["lemmaPosID"] = elements[0]
            entry["sense"] = elements[1]
            entry["context"] = elements[2]
            # append entry to the list
            records.setdefault("process", []).append(entry)
        elif not machineRE.match(record[0]) is None:
            # create dictionary
            entry = {}
            # split line on tabs, add element 0 to entry as lemma.POS.id, 1 as sense, 2 as context
            elements = record
            entry["lemmaPosID"] = elements[0]
            entry["sense"] = elements[1]
            entry["context"] = elements[2]
            # append entry to the list
            records.setdefault("machine", []).append(entry)
        elif not languageRE.match(record[0]) is None:
            # create dictionary
            entry = {}
            # split line on tabs, add element 0 to entry as lemma.POS.id, 1 as sense, 2 as context
            elements = record
            entry["lemmaPosID"] = elements[0]
            entry["sense"] = elements[1]
            entry["context"] = elements[2]
            # append entry to the list
            records.setdefault("language", []).append(entry)
        else:
            continue
    # convert sense into integers
    for entry_list in records.values():
##        sense_dict = {}
##        sense_int = 0
        for entry in entry_list:
##            if entry.get("sense") in sense_dict:
##                entry["senseID"] = sense_dict.get(entry.get("sense"))
##            else:
##                sense_dict[entry["sense"]] = sense_int
##                entry["senseID"] = sense_dict.get(entry["sense"])
##                sense_int += 1
            if entry["sense"] == "process%1:09:00::":
                entry["senseID"] = 0
            elif entry["sense"] == "process%1:04:00::":
                entry["senseID"] = 1
            elif entry["sense"] == "process%1:03:00::":
                entry["senseID"] = 2
            elif entry["sense"] == "process%1:10:00::":
                entry["senseID"] = 3
            elif entry["sense"] == "process%1:08:00::":
                entry["senseID"] = 4
            elif entry["sense"] == "process%1:09:01::":
                entry["senseID"] = 5
            elif entry["sense"] == "machine%1:06:02::":
                entry["senseID"] = 0
            elif entry["sense"] == "machine%1:06:00::":
                entry["senseID"] = 1
            elif entry["sense"] == "machine%1:18:00::":
                entry["senseID"] = 2
            elif entry["sense"] == "machine%1:14:01::":
                entry["senseID"] = 3
            elif entry["sense"] == "machine%1:06:01::":
                entry["senseID"] = 4
            elif entry["sense"] == "machine%1:14:00::":
                entry["senseID"] = 5
            elif entry["sense"] == "language%1:10:03::":
                entry["senseID"] = 0
            elif entry["sense"] == "language%1:10:01::":
                entry["senseID"] = 1
            elif entry["sense"] == "language%1:10:02::":
                entry["senseID"] = 2
            elif entry["sense"] == "language%1:09:00::":
                entry["senseID"] = 3
            elif entry["sense"] == "language%1:10:00::":
                entry["senseID"] = 4
            elif entry["sense"] == "language%1:09:01::":
                entry["senseID"] = 5
            
    # remove head, save index
    for entry_list in records.values():
        for entry in entry_list:
            headMatch=re.compile(r'<head>([^<]+)</head>') #matches contents of head     
            tokens = entry["context"].split() #get the tokens
            entry["headIndex"] = -1 #will be set to the index of the target word
            for i in range(len(tokens)):
                m = headMatch.match(tokens[i])
                if m: #a match: we are at the target token
                    tokens[i] = m.groups()[0]
                    entry["headIndex"] = i
                    entry["context"] = ' '.join(tokens) #turn context back into string (optional)
    # count words
    tokenRE = re.compile("(?:(?!/).)*")
    wordCounts = {}
    for entry_list in records.values():
        for entry in entry_list:
            contexts = entry["context"].split()
            for i in range(len(contexts)):
                token = tokenRE.search(contexts[i]).group(0)
                token = token.lower()
                if token in wordCounts:
                    wordCounts[token] += 1
                else:
                    wordCounts[token] = 1
                if REMOVE_LEMMAPOS:
                    contexts[i] = token
                else:
                    contexts[i] = re.sub(tokenRE, token, contexts[i], count=1)
            entry["context"] = ' '.join(contexts)

    return (records,wordCounts)

# takes a data entry and wordToIndex (works like a map), returns OneHot encoding as a list
def extractOneHot(entry, wordToIndex:dict):
    # set up lists
    previous_OneHot = []
    next_OneHot = []

    for i in range(len(wordToIndex)):
        previous_OneHot.append(0)
        next_OneHot.append(0)

    # get context and pointer
    context = entry.get("context")
    index = entry["headIndex"]
    context = context.split()
    
    # find previous and next words of head
    if not index <= 0:
        prev_word = context[index - 1]
    else:
        prev_word = None
    
    if not index >= len(context)-1:
        next_word = context[index + 1]
    else:
        next_word = None
    
    # get index in map for prev and next, and flip bit in list
    if not prev_word == None:
        prev_index = wordToIndex.get(prev_word)
        # if the word is not found, it is out of the vocab (not in top 2000), set index 2000 to 1
        if prev_index == None:
            previous_OneHot[len(wordToIndex)-1] = 1
        else:
            previous_OneHot[prev_index] = 1
    
    if not next_word == None:
        next_index = wordToIndex.get(next_word)
        # if the word is not found, it is out of the vocab (not in top 2000), set index 2000 to 1
        if next_index == None:
            next_OneHot[len(wordToIndex)-1] = 1
        else:
            next_OneHot[next_index] = 1

    # concantenate the onehots and return that list
    OneHot = previous_OneHot + next_OneHot
    return OneHot

# takes in loadData dataset and a classifer key, returns np.array of onehots
def get_np_onehot(data, classifer):
    onehots = []
    entries = data[0].get(classifer)
    for entry in entries:
        onehots.append(entry.get("onehot"))
    return np.array(onehots)

# takes in loadData dataset and a classifer key, returns np.array of senses
def get_np_senses(data, classifer):
    senses = []
    entries = data[0].get(classifer)
    for entry in entries:
            senses.append(entry.get("senseID"))
    return np.array(senses)

# takes in loadData dataset and returns the 2000 most common words (+ index for OOV) in dict format where key = word, value = # of occurences
def get_mostCommonWords(data):
    wordCounts = data[1]
    count = Counter(wordCounts)
    mostFreqWords = count.most_common(2000)
    mostFreqWords = dict(mostFreqWords)
    mostFreqWords["*OUT_OF_VOCAB*"] = 0
    return mostFreqWords

# takes in mostCommonWords (output of get_most_common_words) and returns wordToIndex dict
def get_wordToIndex(mostCommonWords):
    wordToIndex = mostCommonWords
    index_value = 0
    for k, v in wordToIndex.items():
        wordToIndex[k] = index_value
        index_value += 1
    return wordToIndex

# Logistic Regression class
class LogReg(nn.Module):
    def __init__(self, num_feats, learn_rate = 0.01, device = torch.device("cpu") ):
        super(LogReg, self).__init__()
        self.linear = nn.Linear(num_feats+1, 6) #add 1 to features for intercept #features are inputs, arg1 is output classes

    def forward(self, X):
        #This is where the model itself is defined.
        #For logistic regression the model takes in X and returns
        #a probability (a value between 0 and 1)

        newX = torch.cat((X, torch.ones(X.shape[0], 1)), 1) #add intercept
        #return 1/(1 + torch.exp(-self.linear(newX))) #logistic function on the linear output
        return self.linear(newX) # only use linear if using cross-entropy loss

# uses the training loop and the model setup
def train(model, Xtrain, ytrain, learning_rate = .01, epochs = 2500):
    # model setup
    print("\nTraining Logistic Regression...")
    sgd = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()

    #training loop:
    for i in range(epochs):
        model.train()
        sgd.zero_grad()
        #forward pass:
        ypred = model(Xtrain)
        loss = loss_func(ypred, ytrain) # use nn.CrossEntropyLoss() in HW2
        #backward:
        loss.backward()
        sgd.step()

        if i % 100 == 0:
            print("  epoch: %d, loss: %.5f" %(i, loss.item()))
    
    return model

def test_accuracy(model, Xtest, ytest, test_data, classifer):
    # set model to evaluation mode
    model.eval()
    # calculate accuracy on test set: # torch.sum(ytestpred_class == ytest)
    with torch.no_grad():
        ytestpred_predict = model(Xtest)
        ytestpred_class = ytestpred_predict.argmax(dim=1)
##        print(ytestpred_predict)
##        print(ytestpred_class)
        correct = torch.sum(ytestpred_class == ytest).item()
        print(classifer + ":")
        if(classifer == "process"):
            print("    predictions for process.NOUN.000018: " + str(ytestpred_predict[3].tolist()) + "\n" + "    predictions for process.NOUN.000024: " + str(ytestpred_predict[4].tolist()))
        elif(classifer == "machine"):
            print("    predictions for machine.NOUN.000004: " + str(ytestpred_predict[0].tolist()) + "\n" + "    predictions for machine.NOUN.000008: " + str(ytestpred_predict[1].tolist()))
        elif(classifer == "language"):
            print("    predictions for language.NOUN.000008: " + str(ytestpred_predict[1].tolist()) + "\n" + "    predictions for language.NOUN.000014: " + str(ytestpred_predict[2].tolist()))
        print("    correct: " + str(correct) + " out of " + str(len(ytest)))
    return model

# takes in contexts and wordToIndex, returns 2d np.array cooccurence matrix
def getCoOccMatrix(contexts, wordToIndex):
    # 2001x2001 array of 0s
    coOccMatrix = np.zeros((len(wordToIndex),len(wordToIndex)))
    #print(wordToIndex)

    # loop through contexts, compare words in context with others and count them in the matrix
    for sentence in contexts:
        words = sentence.split()
        for i in range(len(words)-1):
            for j in range(i+1, len(words)):
                # get indexes of both words and add 1 to each tuple-based coordinate in the coOccMatrix
                #print(i)
                #print(j)
                i_index = wordToIndex[words[i] if words[i] in wordToIndex else "*OUT_OF_VOCAB*"]
                j_index = wordToIndex[words[j] if words[j] in wordToIndex else "*OUT_OF_VOCAB*"]
                coOccMatrix[i_index,j_index] += 1
                coOccMatrix[j_index,i_index] += 1
    
    return coOccMatrix

# vocab is wordToIndex, returns dictionary of np.arrays
def getPCAEmb(coOcc, vocab):
    wordEmbeddings = {}
    # standardize the coOcc matrix
    coOcc = (coOcc - coOcc.mean()) / coOcc.std()
    # use torch.svd to extract static, 50 dimensional embeddings
    u, s, v = torch.svd(torch.from_numpy(coOcc))
    u = u[:, :50]
    #print(u) # debug

    # store embeddings in wordEmbeddings (dict)
    u = u.numpy()

    for word, index in vocab.items():
        wordEmbeddings[word] = u[index]

    #print(wordEmbeddings) # debug
    
    return wordEmbeddings

# each embedding a vector of length 200, 1x200
# takes in load_data dataset, pca_embeddings, returns embedded features as a np.array (807x202 for train, 202x200)
def getEmbFeats(data, classifer, pca_emb:dict):
    embFeats = []
    # get words from each classifer's context, the two before the head and the two after
    # use zeros in place of the embedding when word is at the beginning or end of the context, a zero vector of length 1x50
    #print(type(data[0]))
    entries = data[0].get(classifer)
    OOV = "*OUT_OF_VOCAB*"
    for entry in entries:
        context = entry["context"].split()
        headIndex = entry["headIndex"]
        zeros = np.zeros((1,50))
        # get embedding from pca_emb for each word from context that is in range of the head (range = 2)
        two_words_before = None
        one_word_before = None
        one_word_after = None
        two_words_after = None

        if not headIndex <= 1 and not headIndex >= len(context)-2:
            two_words_before = pca_emb[context[headIndex-2] if context[headIndex-2] in pca_emb else OOV]
            one_word_before = pca_emb[context[headIndex-1] if context[headIndex-1] in pca_emb else OOV]
            one_word_after = pca_emb[context[headIndex+1] if context[headIndex+1] in pca_emb else OOV]
            two_words_after = pca_emb[context[headIndex+2] if context[headIndex+2] in pca_emb else OOV]
        elif headIndex <= 1:
            if headIndex == 0:
                two_words_before = zeros
                one_word_before = zeros
                one_word_after = pca_emb[context[headIndex+1] if context[headIndex+1] in pca_emb else OOV]
                two_words_after = pca_emb[context[headIndex+2] if context[headIndex+2] in pca_emb else OOV]
            else: # if headIndex == 1
                two_words_before = zeros
                one_word_before = pca_emb[context[headIndex-1] if context[headIndex-1] in pca_emb else OOV]
                one_word_after = pca_emb[context[headIndex+1] if context[headIndex+1] in pca_emb else OOV]
                two_words_after = pca_emb[context[headIndex+2] if context[headIndex+2] in pca_emb else OOV]
        elif headIndex >= len(context)-2:
            if headIndex == len(context)-1:
                two_words_before = pca_emb[context[headIndex-2] if context[headIndex-2] in pca_emb else OOV]
                one_word_before = pca_emb[context[headIndex-1] if context[headIndex-1] in pca_emb else OOV]
                one_word_after = zeros
                two_words_after = zeros
            else:
                two_words_before = pca_emb[context[headIndex-2] if context[headIndex-2] in pca_emb else OOV]
                one_word_before = pca_emb[context[headIndex-1] if context[headIndex-1] in pca_emb else OOV]
                one_word_after = pca_emb[context[headIndex+1] if context[headIndex+1] in pca_emb else OOV]
                two_words_after = zeros
        prev_vec = np.concatenate((two_words_before, one_word_before), axis = None)
        next_vec = np.concatenate((one_word_after, two_words_after), axis = None)
        embFeats.append(np.concatenate((prev_vec,next_vec), axis = None))

    return np.array(embFeats)

#########################################################
# Main

# Part 1.1
# ####### load the data ######
train_data = load_data("onesec_train.tsv")
test_data = load_data("onesec_test.tsv")
# find the top 2000 words
train_mostCommonWords = get_mostCommonWords(train_data)
test_mostCommonWords = get_mostCommonWords(test_data)

# Part 1.2
# ####### one-hot feature encoding ########
train_wordToIndex = get_wordToIndex(train_mostCommonWords)
test_wordToIndex = get_wordToIndex(test_mostCommonWords)

# loop through all entries in the dictionary, add the onehot as a new key in each entry dict
# print(extractOneHot(data[0].get("process")[0], wordToIndex))
for classifer in train_data[0].values():
    for entry in classifer:
        # get onehot
        onehot = extractOneHot(entry, train_wordToIndex)
        #print(onehot) # debug
        # add onehot to entry dictionary
        entry["onehot"] = onehot

for classifer in test_data[0].values():
    for entry in classifer:
        onehot = extractOneHot(entry, test_wordToIndex)
        entry["onehot"] = onehot

#print(train_data[0].get("process")[0].get("onehot")) #debug

# Part 1.3
# ####### train logistic regress classifers #######
# print(train_data[0].get("process"))

# get onehot length
processOneHot = train_data[0].get("process")[0].get("onehot")
machineOneHot = train_data[0].get("machine")[0].get("onehot")
languageOneHot = train_data[0].get("language")[0].get("onehot")

##print(len(processOneHot))
##print(len(machineOneHot))
##print(len(languageOneHot))

# set up features, including for 1.4
# create training and test for each classifier
# Xtrain
Xtrain_process = torch.from_numpy(get_np_onehot(train_data, "process").astype(np.float32))
Xtrain_machine = torch.from_numpy(get_np_onehot(train_data, "machine").astype(np.float32))
Xtrain_language = torch.from_numpy(get_np_onehot(train_data, "language").astype(np.float32))

# Xtest
Xtest_process = torch.from_numpy(get_np_onehot(test_data, "process").astype(np.float32))
Xtest_machine = torch.from_numpy(get_np_onehot(test_data, "machine").astype(np.float32))
Xtest_language = torch.from_numpy(get_np_onehot(test_data, "language").astype(np.float32))

# ytrain
ytrain_process = torch.from_numpy(get_np_senses(train_data, "process").astype(np.int64))
ytrain_machine = torch.from_numpy(get_np_senses(train_data, "machine").astype(np.int64))
ytrain_language = torch.from_numpy(get_np_senses(train_data, "language").astype(np.int64))

# ytest
ytest_process = torch.from_numpy(get_np_senses(test_data, "process").astype(np.int64))
ytest_machine = torch.from_numpy(get_np_senses(test_data, "machine").astype(np.int64))
ytest_language = torch.from_numpy(get_np_senses(test_data, "language").astype(np.int64))

# train models 
process_model = LogReg(len(processOneHot))
print("\nTraining process model:")
process_model = train(process_model, Xtrain_process, ytrain_process)

machine_model = LogReg(len(machineOneHot))
print("\nTraining machine model:")
machine_model = train(machine_model, Xtrain_machine, ytrain_machine)

language_model = LogReg(len(languageOneHot))
print("\nTraining language model:")
language_model = train(language_model, Xtrain_language, ytrain_language)

# Part 1.4
# ###### test each classifer on the test set ######
print("\n[TESTING UNIGRAM WSD MODELS]")
process_model = test_accuracy(process_model, Xtest_process, ytest_process, test_data, "process")
machine_model = test_accuracy(machine_model, Xtest_machine, ytest_machine, test_data, "machine")
language_model = test_accuracy(language_model, Xtest_language, ytest_language, test_data, "language")

# Part 2.1
#### Convert corpus into co-occurrence matrix ####
contexts = []
for entries in train_data[0].values():
    for entry in entries:
        contexts.append(entry["context"])
context_COMatrix = getCoOccMatrix(contexts, train_wordToIndex)
#print(context_COMatrix)

# Part 2.2
#### Run PCA and extract static, 50 dimensional embeddings ####
pca_emb = getPCAEmb(context_COMatrix, train_wordToIndex)

# Part 2.3
#### Find the euclidean distance between the vectors for the following pairs of words
print("\nDistances between select words:")

#print(pca_emb["language"])

# distances
lang_process = np.linalg.norm(pca_emb["language"] - pca_emb["process"])
mach_process = np.linalg.norm(pca_emb["machine"] - pca_emb["process"])
lang_speak = np.linalg.norm(pca_emb["language"] - pca_emb["speak"])
word_words = np.linalg.norm(pca_emb["word"] - pca_emb["words"])
word_the = np.linalg.norm(pca_emb["word"] - pca_emb["the"])

print("('language', 'process') : " + str(lang_process))
print("('machine', 'process') : " + str(mach_process))
print("('language', 'speak') : " + str(lang_speak))
print("('word', 'words') : " + str(word_words))
print("('word', 'the') : " + str(word_the))

# Part 3.1
#### Extract embedding features ####
# Xtrain
Xtrain_process = torch.from_numpy(getEmbFeats(train_data, "process", pca_emb).astype(np.float32))
Xtrain_machine = torch.from_numpy(getEmbFeats(train_data, "machine", pca_emb).astype(np.float32))
Xtrain_language = torch.from_numpy(getEmbFeats(train_data, "language", pca_emb).astype(np.float32))

# Xtest
Xtest_process = torch.from_numpy(getEmbFeats(test_data, "process", pca_emb).astype(np.float32))
Xtest_machine = torch.from_numpy(getEmbFeats(test_data, "machine", pca_emb).astype(np.float32))
Xtest_language = torch.from_numpy(getEmbFeats(test_data, "language", pca_emb).astype(np.float32))

# Part 3.2
#### Rerun logistic regression training using your word embeddings ####
# train models 
process_model = LogReg(200)
print("\nTraining process model:")
process_model = train(process_model, Xtrain_process, ytrain_process, learning_rate = 1.0, epochs = 500)

machine_model = LogReg(200)
print("\nTraining machine model:")
machine_model = train(machine_model, Xtrain_machine, ytrain_machine, learning_rate = 1.0, epochs = 500)

language_model = LogReg(200)
print("\nTraining language model:")
language_model = train(language_model, Xtrain_language, ytrain_language, learning_rate = 1.0 , epochs = 500)
print("\n")

# Part 3.3
#### Test the new logistic regression classifier ####
process_model = test_accuracy(process_model, Xtest_process, ytest_process, test_data, "process")
machine_model = test_accuracy(machine_model, Xtest_machine, ytest_machine, test_data, "machine")
language_model = test_accuracy(language_model, Xtest_language, ytest_language, test_data, "language")


