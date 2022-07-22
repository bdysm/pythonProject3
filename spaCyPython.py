https://chrisalbon.com/code/machine_learning/preprocessing_text/bag_of_words/

Load library

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Create text
text_data = np.array(['I love Brazil. Brazil!',
                      'Sweden is best',
                      'Germany beats both'])
########################################################

import spacy

nlp = spacy.load("en_core_web_sm")


# Documents and Tokens

doc = nlp("This is a text")

# Spacy
from spacy.lang.en
import English
nlp = English()
sbd = nlp.create_pipe‐
('sentencizer')
nlp.add_pipe(sbd)
doc = nlp(paragraph)
[sent for sent in doc.sents]
nlp = English()
doc = nlp(paragraph)

