# Thesis_PIO_Extraction

### Used versions:

 spaCy version 3.1 (pip install spacy==3.1) + en_core_web_lg pipeline (python -m spacy download en_core_web_lg)
 numpy version 1.18.5 (pip install numpy==1.18.5)
 pandas version 1.0.5 (pip install pandas==1.0.5)
 matplotlib version 3.2.2 (pip install matplotlib==3.2.2)
 seaborn version 0.10.1 (pip install seaborn==0.10.1)
 scikitlearn version 0.23.1 (pip install scikit-learn==0.23.1)
 

## Replication instructions: 

1. Clone the repository
2. Open the command line in the root directory and unzip the 'ebm_nlp_2_00.tar.gz' file so you get the ebm_nlp_2_00 folder in this same directory.
3. Run the prepare.sh file which prepares the data by running the following files:
  loaddata.py: 
    This file first loads all the .tokens files which hold the tokenized texts and all the .txt files which hold the full texts.
    Both of these are stored in seperate arrays together with the document ID. Now all the annotations are read into arrays. 
    The dataset provides individual annotations for every PIO element, since we want to have a dataset where all entities are combined we merge the annotations. 
    We do this by first making all the assigned labels distinct so that we can distin
    
  reformdata.py: 
  
  ner_pico: 
  
