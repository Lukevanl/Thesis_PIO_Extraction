# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 09:44:15 2021

@author: lukev
#================================#
# Load and merge individual data #
#================================#
"""

import os
import numpy as np

DOC_DIR = './ebm_nlp_2_00/documents/'
PATH = './ebm_nlp_2_00/annotations/aggregated/hierarchical_labels'
INTV_PATH = PATH + '/interventions'
OUTC_PATH = PATH + '/outcomes'
PART_PATH = PATH + '/participants' 
filepath_intv_train = INTV_PATH + '/train/'
filepath_outc_train = OUTC_PATH + '/train/'
filepath_part_train = PART_PATH + '/train/'
filepath_intv_test = INTV_PATH + '/test/gold/'
filepath_outc_test = OUTC_PATH + '/test/gold/'
filepath_part_test = PART_PATH + '/test/gold/'

def read_documents(directory, file_format):
    docs = list()
    for file in os.listdir(directory): #Loop through all the files in directory
        filename = os.fsdecode(file)
        if(file_format in filename):
            if (file_format == '.tokens'): #.tokens file containing the tokenized text
                with open(directory + filename, encoding='utf-8') as file:  
                    values = file.readlines()
                    values = [i.strip() for i in values if i != '\n'] #Remove newlines
                    docs.append([filename,values]) #Save tokens together with filename
            elif (file_format == '.txt'):#.txt file containing full text 
                with open(directory + filename, encoding='utf-8') as file:
                    values = file.read()
                    values = values.strip()
                    docs.append([filename, values])
    return docs


def read_annotations(path):
    annotations = list()
    keys = list()
    for file in os.listdir(path): #Loop through all the files in directory
         filename = os.fsdecode(file)
         with open(path + filename) as file: #open .ann file containing annotations 
             values = file.readlines()
             values = [int(i) for i in values] #Convert string to the int value of the annotation
             annotations.append(values) #Add annotation value
             keys.append(filename) #Add filename
    return keys, annotations

def convert_annotations(intv, outc):
    #Convert values of PIO annotations to make the classes
    #exclusive so that we can combine them
    interventions = list()
    outcomes = list()
    for i in range(len(intv)):
        labels = [i if i == 0 else i+4 for i in intv[i]] #There are 4 P classes
        interventions.append(labels)
    for i in range(len(outc)):
        labels = [i if i == 0 else i+11 for i in outc[i]] # 4 + 7 Intervention classes
        outcomes.append(labels)
    return interventions, outcomes

def merge_annotations(arr1, keys1, arr2, keys2, arr3, keys3):
    merged_annotations = list()
    arr1, arr2 = convert_annotations(arr1, arr2)
    for i in range(len(arr1)):
        try:
            index = keys2.index(keys1[i])
            index2 = keys3.index(keys1[i])
            maximum = list()
            for j in range(len(arr2[index])):
                maximum.append(max(arr1[i][j], arr2[index][j], arr3[index2][j]))
            merged_annotations.append([keys1[i], maximum])
        except ValueError:
            continue
    return merged_annotations      
        
def save_files():
    #Read train and test data
    keys_intv_train, intv_train = read_annotations(filepath_intv_train)
    keys_outc_train, outc_train = read_annotations(filepath_outc_train)
    keys_part_train, part_train = read_annotations(filepath_part_train)
    keys_intv_test, intv_test = read_annotations(filepath_intv_test)
    keys_outc_test, outc_test = read_annotations(filepath_outc_test)
    keys_part_test, part_test = read_annotations(filepath_part_test)
    #Combine individual PIO annotations into single annotation file
    merged_train = merge_annotations(intv_train, keys_intv_train,
                               outc_train, keys_outc_train, part_train,
                               keys_part_train)
    merged_test = merge_annotations(intv_test, keys_intv_test,
                               outc_test, keys_outc_test, part_test,
                               keys_part_test)
    merged_ann_train = [i[1] for i in merged_train]
    with open('merged.npy', 'wb') as file:
        np.save(file, np.asarray(merged_ann_train))
    merged_ann_test = [i[1] for i in merged_test]
    with open('merged_test.npy', 'wb') as file:
        np.save(file, np.asarray(merged_ann_test))
    keys_merged_train = [i[0] for i in merged_train]
    keys_merged_test = [i[0] for i in merged_test]
    #Read tokens and full texts
    tokens = read_documents(DOC_DIR, ".tokens")
    text = read_documents(DOC_DIR, ".txt")
    with open('full_texts.npy', 'wb') as file:
        np.save(file, text) #Save full texts
    keys_tokens = [i[0] for i in tokens]
    tokens = [i[1] for i in tokens]
    #Save leftover files
    with open('keys_merged_ann.npy', 'wb') as file:
        np.save(file, keys_merged_train) #Save keys of annotations
    with open('keys_merged_ann_test.npy', 'wb') as file:
        np.save(file, keys_merged_test)   
    with open('keys_tokens.npy', 'wb') as file:
        np.save(file, keys_tokens) #Save keys of token files
    with open('tokens.npy', 'wb') as file:
        np.save(file, tokens) #Save tokens
    
if __name__ == '__main__':
    print("Loading data...")
    save_files()
    print("Done with loading data")