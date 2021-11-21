# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 21:39:34 2021

@author: lukev
"""
import pandas as pd
import numpy as np
import re
import spacy
from tqdm import tqdm
import random
import json
from spacy.tokens import DocBin
from sklearn.model_selection import train_test_split


def load_data():
    texts_train = np.load('data/full_texts.npy', allow_pickle=True)
    data_train = np.load('data/data_standard.npy', allow_pickle=True)
    indices_train = np.load('data/indices.npy', allow_pickle=True)
    texts_test = np.load('data/full_texts_test.npy', allow_pickle=True)
    data_test = np.load('data/data_standard_test.npy', allow_pickle=True)
    indices_test = np.load('data/indices_test.npy', allow_pickle=True)
    return data_train, texts_train, indices_train, data_test, texts_test, indices_test

def make_spacy_format(annotations, text, indices):
    labels = [[x[0] for x in X] for X in annotations]
    tokens = [[x[1] for x in X] for X in annotations]
    #print(tokens[0])
    list_entities = list()
    for j in range(len(tokens)):
        index_count = 0 #Keep track of the index of indices
        entities = list()
        for i in range(len(tokens[j])):
            label = labels[j][i]
           # print(word,label)
            if(label != "None"): #Check if there is a label
                entities.append((int(indices[j][index_count][0]), int(indices[j][index_count][1]), label))
                index_count += 1
        list_entities.append(entities)
    spacy_form = list()
    for i in range(len(list_entities)):
        dict_entities = dict()
        dict_entities['entities'] = list_entities[i]
        spacy_form.append([text[i][1], dict_entities]) #Spacy formats
    return spacy_form

def save_spacy_format(spacy_form, is_train, is_dev, is_hier):
    nlp = spacy.blank("en") # load a new spacy model
    db = DocBin() # create a DocBin object
    for text, annot in tqdm(spacy_form):
        #print(annot)# data in previous format
        doc = nlp.make_doc(str(text)) # create doc object from text
        ents = []
        for start, end, label in annot["entities"]: # add character indexes
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                pass
            else:
                ents.append(span)
        doc.ents = ents # label the text with the ents
        db.add(doc)
    if(is_hier):
        if(is_train):
            db.to_disk("./spacy_eff/trainhier.spacy")
            db.to_disk("./spacy_acc/trainhier.spacy")
        elif(is_dev):
            db.to_disk("./spacy_eff/devhier.spacy")
            db.to_disk("./spacy_acc/devhier.spacy")
        else:
            db.to_disk("./spacy_eff/testhier.spacy")
            db.to_disk("./spacy_acc/testhier.spacy")
    else:
         if(is_train):
            db.to_disk("./spacy_eff/train.spacy")
            db.to_disk("./spacy_acc/train.spacy")
         elif(is_dev):
            db.to_disk("./spacy_eff/dev.spacy")
            db.to_disk("./spacy_acc/dev.spacy")
         else:
            db.to_disk("./spacy_eff/test.spacy")
            db.to_disk("./spacy_acc/test.spacy")
 
def shuffle_three_arrays(a, b, c):
    perm = np.random.permutation(len(a))
    return a[perm], b[perm], c[perm]
    
def main():    
    data_train, texts_train, indices_train, data_test, texts_test, indices_test  = load_data()
    #data_train, data_dev, texts_train, texts_dev, indices_train, indices_dev = train_test_split(data_train, texts_train, indices_train, test_size=0.10, random_state=10) 
  #  data_train, texts_train, indices_train = shuffle_three_arrays(data_train, texts_train, indices_train)
    dev_indx = int(0.10*len(data_train))
    data_train, data_dev, texts_train, texts_dev, indices_train, indices_dev = data_train[dev_indx:], data_train[:dev_indx], texts_train[dev_indx:], texts_train[:dev_indx], indices_train[dev_indx:], indices_train[:dev_indx]
    file_names = [i[0] for i in texts_dev]
    with open('data/file_names_dev.npy', 'wb') as file: #For evaluation
        np.save(file, file_names)
    spacy_format_train = make_spacy_format(data_train, texts_train, indices_train)
    #spacy_format_train = make_spacy_format(data_train, texts_train, indices_train)
    spacy_format_test = make_spacy_format(data_test, texts_test, indices_test)
    spacy_format_dev = make_spacy_format(data_dev, texts_dev, indices_dev)
    with open('data/spacy_format_test.npy', 'wb') as file: #For evaluation
        np.save(file, spacy_format_dev)
 #   with open('data/spacy_form_dev.npy', 'wb') as file: #For evaluation
  #      np.save(file, spacy_format_train[:dev_indx])
    is_hier = True
    save_spacy_format(spacy_format_train, True, False, is_hier)
    save_spacy_format(spacy_format_dev, False, True, is_hier)
    save_spacy_format(spacy_format_test, False, False, is_hier)

if __name__ == '__main__':
    main()