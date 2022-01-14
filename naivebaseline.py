# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 17:28:03 2021

@author: lukev

#================================#
# Scores of naive baseline model #
#================================#
"""
import numpy as np

def get_scores(data): 
    #Precision, Recall and F1 when always guessing the most common class
    targetlabel = {0: 'None', 1: 'P: Age', 2: 'P: Sex', 3: 'P: Sample size', 
                    4: 'P: Condition', 5: 'I: Surgical', 6: 'I: Physical', 
                    7: 'I: Drug', 8: 'I: Educational', 9: 'I: Psychological', 
                    10: 'I: Other', 11: 'I: Control', 12: 'O: Physical', 13: 'O: Pain', 
                    14: 'O: Mortality', 15: 'O: Adverse effects', 16: 'O: Mental',
                    17: 'O: Other'}
    label_to_target = dict((v,k) for k,v in targetlabel.items())
    tokens = np.load('data/data_standard_test.npy', allow_pickle = True)
    counts_test = get_counts(tokens, label_to_target)
    tp = counts_test[label_to_target["O: Physical"]]
    fp = np.sum(counts_test) - tp
    fn = fp - counts_test[label_to_target["None"]]
    p = tp / (tp + fp)
    r = tp / (tp + (fn))
    f1 = (2*p*r) / (p+r)
    return p, r, f1
            
def get_counts(tokens, label_to_target):
    #Calculate counts of each hierarchical PIO element
    counts = [0] * len(label_to_target)
    for doc in tokens:
        for token in doc:
            entity = token[0]
            counts[label_to_target[entity]] += 1
    return counts
            

def naive_baseline():
    data_test = np.load('data/spacy_format_test.npy', allow_pickle=True)
    precision, recall, f1_score = get_scores(data_test)
    print("Precision:" + str(precision) + "\nRecall: " + str(recall) + "\nF1-score: " + str(f1_score))
            
    
naive_baseline()