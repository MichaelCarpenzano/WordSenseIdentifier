#!/usr/bin/python3
# Michael Carpenzano
# CSE354, Spring 2021
##########################################################

import sys
#import json #for reading json encoded files into opjects
import re #regular expressions
import numpy as np
import torch
import torch.nn as nn  #pytorch
import csv
from collections import Counter
import itertools

# Comment this line out if you wish to see results on the console
#sys.stdout = open('languagegenerator_OUTPUT.txt', 'w', encoding = 'utf8') # EDIT THIS

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
    wordCounts = {"<s>": 0,
                  "</s>": 0}
    for entry_list in records.values():
        for entry in entry_list:
            contexts = entry["context"].split()
            wordCounts["<s>"] += 1
            wordCounts["</s>"] += 1
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

    # add beginning and end tags
    for entry_list in records.values():
        for entry in entry_list:
            contexts = entry["context"].split()
            contexts = ["<s>"] + contexts
            contexts[len(contexts):] = ["</s>"]
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

# takes in loadData dataset and returns the 5000 most common words (+ index for OOV) in dict format where key = word, value = # of occurences
def get_mostCommonWords(data):
    wordCounts = data[1]
    del wordCounts[""]
    count = Counter(wordCounts)
    mostFreqWords = count.most_common(5000)
    mostFreqWords = dict(mostFreqWords)
    mostFreqWords["<OOV>"] = 0
    return mostFreqWords

# takes in mostCommonWords (output of get_most_common_words) and returns wordToIndex dict
def get_wordToIndex(mostCommonWords):
    wordToIndex = mostCommonWords
    index_value = 0
    for k, v in wordToIndex.items():
        wordToIndex[k] = index_value
        index_value += 1
    return wordToIndex

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
                i_index = wordToIndex[words[i] if words[i] in wordToIndex else "<OOV>"]
                j_index = wordToIndex[words[j] if words[j] in wordToIndex else "<OOV>"]
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
    OOV = "<OOV>"
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

def extract_unigram(contexts, mostCommonWords):
    unigramCounts = {}

    for context in contexts:
        for word in context:
            if word in mostCommonWords:
                if word in unigramCounts:
                    unigramCounts[word] = unigramCounts[word] + 1
                else:
                    unigramCounts[word] = 1
            else:
                if "<OOV>" in unigramCounts:
                    unigramCounts["<OOV>"] += 1
                else:
                    unigramCounts["<OOV>"] = 1
    
    return unigramCounts

def extract_bigram(contexts, mostCommonWords):
    bigramCounts = {"<OOV>": {"<OOV>": 0}}
    OOV = "<OOV>"

    for context in contexts:
        sequences = [context[i:i + 2] for i in range(len(context) - 2 + 1)]
        
        for bigram in sequences:
                try:
                    bigramCounts[bigram[0] if bigram[0] in mostCommonWords else OOV][bigram[1] if bigram[1] in mostCommonWords else OOV] += 1
                except KeyError:
                    try:
                        bigramCounts[bigram[0] if bigram[0] in mostCommonWords else OOV][bigram[1] if bigram[1] in mostCommonWords else OOV] = 1
                    except KeyError:
                        bigramCounts[bigram[0] if bigram[0] in mostCommonWords else OOV] = {bigram[1] if bigram[1] in mostCommonWords else OOV: 1}

    return bigramCounts

def extract_trigram(contexts, mostCommonWords):
    trigramCounts = {("<OOV>","<OOV>"): {"<OOV>": 0}}
    OOV = "<OOV>"

    for context in contexts:
        sequences = [context[i:i + 3] for i in range(len(context) - 3 + 1)]

        for trigram in sequences:
            try:
                trigramCounts[(trigram[0] if trigram[0] in mostCommonWords else OOV, trigram[1] if trigram[1] in mostCommonWords else OOV)][trigram[2] if trigram[2] in mostCommonWords else OOV] += 1
            except KeyError:
                try:
                    trigramCounts[(trigram[0] if trigram[0] in mostCommonWords else OOV, trigram[1] if trigram[1] in mostCommonWords else OOV)][trigram[2] if trigram[2] in mostCommonWords else OOV] = 1
                except KeyError:
                    trigramCounts[(trigram[0] if trigram[0] in mostCommonWords else OOV, trigram[1] if trigram[1] in mostCommonWords else OOV)] = {trigram[2] if trigram[2] in mostCommonWords else OOV: 1}

    return trigramCounts

def find_possible_words(bigramCounts, word):
    possible_words = []

    for k in bigramCounts[word]:
        possible_words = [k] + possible_words

    possible_words = list(dict.fromkeys(possible_words))

    return possible_words

def compute_probs(unigramCounts, bigramCounts, trigramCounts, wordMinus1, wordMinus2 = None, vocab_size = 5000):
    probs = {}
    bigramProbs = {}
    trigramProbs = {}

    # make sure OOV is valid in all counts
##    OOV = "<OOV>"
##    if OOV not in unigramCounts:
##        unigramCounts[OOV] = -1
##    if OOV not in bigramCounts[wordMinus1]:
##        bigramCounts[wordMinus1][OOV] = -1
##    if wordMinus2:
##        try:
##            if OOV not in trigramCounts[(wordMinus2, wordMinus1)]:
##                trigramCounts[(wordMinus2, wordMinus1)][OOV] = -1
##        except KeyError:
##            try:
##                trigramCounts[(wordMinus2, wordMinus1)] = {OOV: -1}
##            except:
##                print("oh no")

    # get possible words
    possible = find_possible_words(bigramCounts, wordMinus1)
    # make sure OOV is in the list
    if "<OOV>" not in possible:
        possible = ["<OOV>"] + possible

    # calculate probabilities
    # bigram probabilities
    # get count of wordMinus1
    wordMin1Count = unigramCounts[wordMinus1]
    for word in possible:
        ## get bigram prob
        try:
            bigramProbs[word] = (bigramCounts[wordMinus1][word] + 1) / (wordMin1Count + vocab_size)
        except:
            bigramProbs[word] = 0
    
    if not wordMinus2 is None:
        # if there are two prior words
        # calculate trigram probs
        for word in possible:
            try:
                trigramProbs[word] = (trigramCounts[(wordMinus2, wordMinus1)][word] + 1) / (bigramCounts[wordMinus2][wordMinus1] + vocab_size)
            except KeyError:
                trigramProbs[word] = "Not valid Wi"
        # interpolate
        for word in possible:
            if trigramProbs[word] == "Not valid Wi":
                try:
                    probs[word] = 1 / (bigramCounts[wordMinus2][wordMinus1] + vocab_size)
                    continue
                except:
                    probs[word] = 1 / vocab_size
            else:
                probs[word] = (bigramProbs[word] + trigramProbs[word]) / 2
    else:
        probs = bigramProbs
        
    return probs

def generate_language(unigramCounts, bigramCounts, trigramCounts, words, max_length = 32):
    full_sentence = words

    while len(full_sentence) < max_length and not full_sentence[-1] == "</s>":
        if len(full_sentence) < 2:
            prob_dict = compute_probs(unigramCounts, bigramCounts, trigramCounts, full_sentence[-1])
            choices = list(prob_dict.keys())
            probs = list(prob_dict.values())
            
            ## normalize probabilities
            if not sum(probs) == 1:
                original_sum = sum(probs)
                for i, element in enumerate(probs):
                    probs[i] = probs[i] / original_sum
            
            full_sentence.append(np.random.choice(np.asarray(choices), p = np.asarray(probs)))
        else:
            prob_dict = compute_probs(unigramCounts, bigramCounts, trigramCounts, full_sentence[-1], full_sentence[-2])
            choices = list(prob_dict.keys())
            probs = list(prob_dict.values())
            
            ## normalize probabilities
            if not sum(probs) == 1:
                original_sum = sum(probs)
                for i, element in enumerate(probs):
                    probs[i] = probs[i] / original_sum

            full_sentence.append(np.random.choice(np.asarray(choices), p = np.asarray(probs)))
            
    return full_sentence

#########################################################
# Main

# Part 1.1
# ####### load the data ######
train_data = load_data("onesec_train.tsv")
# find the top 5000 words
mostCommonWords = get_mostCommonWords(train_data)

#### get corpus ####
# tokenize sentences and add <s> and </s> tags
contexts = []
for entries in train_data[0].values():
    for entry in entries:
        contexts.append(entry["context"].split())
##print(*contexts, sep="\n")

##for i, context in enumerate(contexts):
##    contexts[i] = ["<s>"] + context
##    contexts[i].append("</s>")

#### part 2.2 ####
#### extract unigram, bigram, and trigram counts ####
unigramCounts = extract_unigram(contexts, mostCommonWords)
bigramCounts = extract_bigram(contexts, mostCommonWords)
trigramCounts = extract_trigram(contexts, mostCommonWords)

print("CHECKPOINT 2.2 - counts \n  1-grams:")
print("\t('language',) " + str(unigramCounts["language"]))
print("\t('the',) " + str(unigramCounts["the"]))
print("\t('formal',) " + str(unigramCounts["format"]))

print("  2-grams:")
print("\t('the', 'language') " + str(bigramCounts["the"]["language"]))
print("\t('<OOV>', 'language') " + str(bigramCounts["<OOV>"]["language"]))
print("\t('to', 'process') " + str(bigramCounts["to"]["process"]))

to_process_OOV = 0
try:
    to_process_OOV = trigramCounts[("to", "process")]["<OOV>"]
except:
    to_process_OOV = 0

specific_formal_event = 0
try:
    specific_formal_event = trigramCounts[("specific", "formal")]["event"]
except:
    specific_formal_event = 0

print("  3-grams:")
print("\t('specific', 'formal', 'languages') " + str(trigramCounts[("specific","formal")]["languages"]))
print("\t('to', 'process', '<OOV>') " + str(to_process_OOV))
print("\t('specific', 'formal', 'event') " + str(specific_formal_event))
print()

#### part 2.3 ####
# compute probablities of all possible words
##wi_possible = find_possible_words(bigramCounts)
##print(wi_possible)

#compute_probs(unigramCounts, bigramCounts, trigramCounts, wordMinus1, wordMinus2 = None, vocab_size = 5000)
the_probs = compute_probs(unigramCounts, bigramCounts, trigramCounts, "the")
OOV_probs = compute_probs(unigramCounts, bigramCounts, trigramCounts, "<OOV>")
to_probs = compute_probs(unigramCounts, bigramCounts, trigramCounts, "to")

specific_formal_probs = compute_probs(unigramCounts, bigramCounts, trigramCounts, "formal", "specific")
to_process_probs = compute_probs(unigramCounts, bigramCounts, trigramCounts, "process", "to")

print("CHECKPOINT 2.3 - Probs with add-one \n  2-grams:")
print("\t('the', 'language') " + str(the_probs["language"]))
print("\t('<OOV>', 'language') " + str(OOV_probs["language"]))
print("\t('to', 'process') " + str(to_probs["process"]))
print("  3-grams:")
print("\t('specific', 'formal', 'languages') " + str(specific_formal_probs["languages"]))
print("\t('to', 'process', '<OOV>') " + str(to_process_probs["<OOV>"]))

try:
    print("\t('specific', 'formal', 'event') " + str(specific_formal_probs["event"]))
except KeyError:
    print("\t('specific', 'formal', 'event') " + "INVALID W_i")

#### part 2.4 ####

print("\n\nFINAL CHECKPOINT - Generated Language")
print()

print(" PROMPT: <s>")
for i in range(3):
    s_sentence = ' '.join(generate_language(unigramCounts, bigramCounts, trigramCounts, ["<s>"]))
    print("\t" + str(s_sentence))

print(" PROMPT: <s> language is")
for i in range(3):
    s_language_is_sentence = ' '.join(generate_language(unigramCounts, bigramCounts, trigramCounts, ["<s>", "language", "is"]))
    print("\t" + str(s_language_is_sentence))

print(" PROMPT: <s> machines")
for i in range(3):
    s_machines_sentence = ' '.join(generate_language(unigramCounts, bigramCounts, trigramCounts, ["<s>", "machines"]))
    print("\t" + str(s_machines_sentence))

print(" PROMPT: <s> they want to process")
for i in range(3):
    s_they_want_to_process_sentence = ' '.join(generate_language(unigramCounts, bigramCounts, trigramCounts, ["<s>", "they", "want", "to", "process"]))
    print("\t" + str(s_they_want_to_process_sentence))












