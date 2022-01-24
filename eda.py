# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 10:15:46 2021

@author: lukev
"""
import to_spacy_format as spacy
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def plot_count_elements(elements, hier):
    #Get the counts for each element
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    hier_pico = ['P: Combined', 'I: Combined', 'O: Combined', 'P: Age', 'P: Sex', 'P: Sample size', 
                 'P: Condition', 'I: Surgical', 'I: Physical', 
                 'I: Drug', 'I: Educational', 'I: Psychological', 
                 'I: Other', 'I: Control', 'O: Physical', 'O: Pain', 
                 'O: Mortality', 'O: Adverse effects', 'O: Mental',
                 'O: Other']
    pico = ['Population', 'Intervention', 'Outcome']
    if(hier):
        counts = np.zeros(len(hier_pico)) #Init counts to 0 of hierarchical
        for i in range(len(elements)): #For each document
            for j in range(len(elements[i])): #For each label in the document
                if(elements[i][j] != 'None'):
                    index = hier_pico.index(elements[i][j])
                    counts[index] += 1
        counts[0] = np.sum(counts[3:7])
        counts[1] = np.sum(counts[7:14])
        counts[2] = np.sum(counts[14:20])
        ax.bar(hier_pico,counts)
        counts[-3]
        plt.xticks(
            rotation=45, 
            horizontalalignment='right',
            fontweight='light',
            fontsize='x-large'  )
        print(counts)
        plt.show()
    else:
        counts = np.zeros(len(pico)) #Init counts to 0 of regular
        for i in range(len(elements)): #For each document
            for j in range(len(elements[i])): #For each label in the document
                if(elements[i][j] != 'None'):
                    index = pico.index(elements[i][j])
                    counts[index] += 1
        ax.bar(pico,counts)
        plt.xticks(
            rotation=45, 
            horizontalalignment='right',
            fontweight='light',
            fontsize='x-large'  )
        print(counts)
        plt.show()
   
def data_of_texts():
    list_of_tokens = np.load('tokens.npy', allow_pickle=True)
    lengths_of_texts = [len(i) for i in list_of_tokens]
    print(F"Average amount of words in texts = {np.mean(lengths_of_texts)}")
    print(F"Maximum amount of words in texts = {np.max(lengths_of_texts)}")
    print(F"Minimum amount of words in texts = {np.min(lengths_of_texts)}")
    print(F"Standard deviation of length of words in texts = {np.std(lengths_of_texts)}")
    
           
data, texts, indices, _, _, _ = spacy.load_data() #Load training date
#_, _, _, data, texts, indices = ner.load_data()#Load test data
pio_values = [[x[0] for x in X] for X in data]
plot_count_elements(pio_values, True)
data_of_texts()

        