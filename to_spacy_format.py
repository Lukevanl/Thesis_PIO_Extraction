# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 21:39:34 2021

@author: lukev
#===============================#
# Transform into spacy 3 format #
#===============================#
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
#Transforms data into spacy format

def load_data():
    #Load all transformed data
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
            if(label != "None"): #Check if there is a label
                #Append new entity format
                entities.append((int(indices[j][index_count][0]), int(indices[j][index_count][1]), label))
                index_count += 1
        list_entities.append(entities)
    spacy_form = list() #Holds spacy 2.0 format
    with open('data/stand_off_ann.npy', 'wb') as file: #For evaluation
        np.save(file, list_entities)
    for i in range(len(list_entities)):
        dict_entities = dict()
        dict_entities['entities'] = list_entities[i]
        spacy_form.append([text[i][1], dict_entities]) #Spacy formats
    return spacy_form

def save_spacy_format(spacy_form, is_train, is_dev, is_hier):
    nlp = spacy.blank("en")
    db = DocBin()
    print("Converting to spaCy 3 format...")
    for text, annot in tqdm(spacy_form):
        doc = nlp.make_doc(str(text)) # Make a Doc object out of the text
        ents = []
        for start, end, label in annot["entities"]:
            #Generate the entities of the doc object
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                pass
            else:
                ents.append(span)
        doc.ents = ents # assign entities
        db.add(doc)
    if(is_hier): #Save hierarchical PICO format to disk
        if(is_train): #Is training set
            db.to_disk("./spacy_acc/trainhier.spacy")
        elif(is_dev): #Is development set
            db.to_disk("./spacy_acc/devhier.spacy")
        else: #Is test set
            db.to_disk("./spacy_acc/testhier.spacy")
    else: #Save regular PIO format to disk
         if(is_train): #Is training set
            db.to_disk("./spacy_acc/train.spacy")
         elif(is_dev): #Is development set
            db.to_disk("./spacy_acc/dev.spacy")
         else: #Is test set
            db.to_disk("./spacy_acc/test.spacy")
 
    
def main():    
    data_train, texts_train, indices_train, data_test, texts_test, indices_test  = load_data()
    dev_indx = int(0.10*len(data_train)) 
    #Make development sets
    data_train, data_dev, texts_train, texts_dev, indices_train, indices_dev = data_train[dev_indx:], data_train[:dev_indx], texts_train[dev_indx:], texts_train[:dev_indx], indices_train[dev_indx:], indices_train[:dev_indx]
    file_names = [i[0] for i in texts_dev]
    with open('data/file_names_dev.npy', 'wb') as file: #For evaluation
        np.save(file, file_names)
    print("Making spacy 2 format...")
    spacy_format_train = make_spacy_format(data_train, texts_train, indices_train)
    spacy_format_test = make_spacy_format(data_test, texts_test, indices_test)
    spacy_format_dev = make_spacy_format(data_dev, texts_dev, indices_dev)
    print("Done...")
    with open('data/spacy_form_train.npy', 'wb') as file: #For evaluation
        np.save(file, spacy_format_train)
    with open('data/spacy_form_dev.npy', 'wb') as file: #For evaluation
        np.save(file, spacy_format_train[:dev_indx])
    is_hier = True #Set true if you want hierarchical labels and false for regular ones
    save_spacy_format(spacy_format_train, True, False, is_hier) #Generate spacy format
    save_spacy_format(spacy_format_dev, False, True, is_hier)
    save_spacy_format(spacy_format_test, False, False, is_hier)

if __name__ == '__main__':
    main()
    print("Done")