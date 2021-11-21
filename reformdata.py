# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 08:45:09 2021

@author: lukev
"""

import numpy as np
import re
from loaddata import save_files


def strip_keys():#Strips the text of the keys and only preserves the numerical ID
    merged_ann_keys_train = np.load('keys_merged_ann.npy', allow_pickle=True)
    merged_ann_keys_train = [int(re.search(r'\d+', key).group()) for key in merged_ann_keys_train]
    merged_ann_keys_test = np.load('keys_merged_ann_test.npy', allow_pickle=True)
    merged_ann_keys_test = [int(re.search(r'\d+', key).group()) for key in merged_ann_keys_test]
    merged_tok_keys = np.load('keys_tokens.npy', allow_pickle=True)
    merged_tok_keys = [int(re.search(r'\d+', key).group()) for key in merged_tok_keys]
    return merged_ann_keys_train, merged_ann_keys_test, merged_tok_keys

def load_data(): 
    annotations_train = np.load('merged.npy', allow_pickle=True)
    annotations_test = np.load('merged_test.npy', allow_pickle=True)
    tokens = np.load('tokens.npy', allow_pickle=True)
    texts = np.load('full_texts.npy', allow_pickle=True)
    return annotations_train, annotations_test, tokens, texts

def merge_with_keys(arr, keys): #Makes a tuple of the text and keys
    arr_with_keys = list()
    for i in range(len(arr)):
        arr_with_keys.append((keys[i], arr[i]))
    return arr_with_keys
    

def match_tokens_with_annotations(annotations, tokens, keys_annotations, keys_tokens):
    tupled = list() #Holds all matching documents and tokens in a tuple
    matched = list() #Final array
    for i in range(len(annotations)):
        try: 
            #Match annotation i to the corresponding tokens of i
            index = keys_tokens.index(keys_annotations[i])
            tupled.append((annotations[i], tokens[index]))
        except ValueError: 
            #No matching index
            continue
    #print(tupled[0])
    for i in range(len(tupled)):
        tuples = list() #Holds the tuples of all tokens of one document
        for j in range(len(tupled[i][0])):
            tuples.append((tupled[i][0][j], tupled[i][1][j])) 
        matched.append(tuples) # Add tuples of one document
   # print(matched)
    return matched #Return all tuples of all tokens of all documents

def match_texts_with_annotations(texts, annotations, keys_annotations):
    text_list = list()
    keys = [int(re.search(r'\d+', key[0]).group()) for key in texts]
    for i in range(len(annotations)):
        try: 
            index = keys.index(keys_annotations[i])
            text_list.append(texts[index])
        except ValueError: 
            #No matching index
            continue
    return text_list
    
def map_numerical_label_to_string(xtrain):
# =============================================================================
#     targetlabel = {0: 'None', 1: 'Population', 2: 'Population', 3: 'Population', 
#                    4: 'Population', 5: 'Intervention', 6: 'Intervention', 
#                    7: 'Intervention', 8: 'Intervention', 9: 'Intervention', 
#                    10: 'Intervention', 11: 'Intervention', 12: 'Outcome', 13: 'Outcome', 
#                    14: 'Outcome', 15: 'Outcome', 16: 'Outcome',
#                    17: 'Outcome'} #Maps labels to regular PICO elements
# =============================================================================
    targetlabel = {0: 'None', 1: 'P: Age', 2: 'P: Sex', 3: 'P: Sample size', 
                    4: 'P: Condition', 5: 'I: Surgical', 6: 'I: Physical', 
                    7: 'I: Drug', 8: 'I: Educational', 9: 'I: Psychological', 
                    10: 'I: Other', 11: 'I: Control', 12: 'O: Physical', 13: 'O: Pain', 
                    14: 'O: Mortality', 15: 'O: Adverse effects', 16: 'O: Mental',
                    17: 'O: Other'} #Maps label to hierarchical PICO elements"""

    names_list = list() #Contains the names and tokens of all documents
    for i in range(len(xtrain)):
        names = list() #Keeps track of all the names of 1 document
        for j in range(len(xtrain[i])):
            names.append((targetlabel[xtrain[i][j][0]], xtrain[i][j][1])) #Convert num to targetlabel
        names_list.append(names)
    return names_list
 
def get_indices(data, full_texts):
    tokens = [[x[1] for x in X] for X in data]
    labels = [[x[0] for x in X] for X in data]
    list_indices = list()
    for i in range(len(tokens)):
        indices = list()
        last_index = 0
        for j in range(len(tokens[i])):
            index_start = full_texts[i][1].find(tokens[i][j], last_index)
            last_index = index_start+len(tokens[i][j])
            if (labels[i][j] != "None"):
                indices.append([index_start, index_start+len(tokens[i][j])])
        list_indices.append(indices)
    return list_indices
    
    
def load_all_data():
    save_files()
    keys_ann_train, keys_ann_test, keys_tok = strip_keys() #Get keys
    ann_train, ann_test, tok, texts = load_data() #Get annotations and tokens
    texts_train = match_texts_with_annotations(texts, ann_train, keys_ann_train)
    data_standard_train = match_tokens_with_annotations(ann_train, tok, keys_ann_train, keys_tok) #Tuple annotations and tokens
    data_standard_train = map_numerical_label_to_string(data_standard_train)
    indices_train = get_indices(data_standard_train, texts_train)
    texts_test = match_texts_with_annotations(texts, ann_test, keys_ann_test)
    data_standard_test = match_tokens_with_annotations(ann_test, tok, keys_ann_test, keys_tok) #Tuple annotations and tokens
    data_standard_test = map_numerical_label_to_string(data_standard_test)
    indices_test = get_indices(data_standard_test, texts_test)
   # print(data_standard[0])
    with open('data/data_standard.npy', 'wb') as file:
        np.save(file, data_standard_train)
    with open('data/full_texts.npy', 'wb') as file:
        np.save(file, texts_train)
    with open('data/indices.npy', 'wb') as file:
        np.save(file, indices_train)
    with open('data/data_standard_test.npy', 'wb') as file:
        np.save(file, data_standard_test)
    with open('data/full_texts_test.npy', 'wb') as file:
        np.save(file, texts_test)
    with open('data/indices_test.npy', 'wb') as file:
        np.save(file, indices_test)
    return data_standard_train, texts_train, indices_train, data_standard_test, texts_test, indices_test


if __name__ == '__main__':
    data_structured_train, texts_train, indices_train, data_structured_test, texts_test, indices_test = load_all_data()
    print("Sizes of training and test data: ", len(data_structured_train),len(texts_train),len(indices_train),
          len(data_structured_test),len(texts_test),len(indices_test))
   