# %% [markdown]
# # 1.0 Libraries

# %%
!pip install langid
!pip install gensim
#!pip install -U spacy
!python -m spacy download it_core_news_sm
!pip install langdetect
# !wget https://github.com/explosion/sense2vec/releases/download/v1.0.0/s2v_reddit_2015_md.tar.gz
# !tar -xzvf s2v_reddit_2015_md.tar.gz
# !pip install sense2vec
#!pip install spacy_fastlang


# %% [markdown]
# #Imports

# %%
import numpy as np
import pandas as pd

# from google.colab import drive
# drive.mount('/content/gdrive')

import re
import json
import gensim.downloader as api

import langid
from gensim.models import Word2Vec

import random

import torch

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# import seaborn as sns
import matplotlib.pyplot as plt

# import nltk
# nltk.download('omw-1.4')
# nltk.download('wordnet')
# from nltk.corpus import wordnet as wn

import time
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# # 2.0 Set up

# %%
import random

# %%
# load the data full dataset
# def load_data(data_path, gold_path):
#   hypernyms_dict = {}
#   # read the data and gold files
#   with open(data_path, "r", encoding = 'utf-8') as data_file, open(gold_path, "r", encoding = 'utf-8') as gold_file:
#     for data_line, gold_line in zip(data_file, gold_file):
#       # get the term and the hypernyms
#       term_list = [term for term in data_line.split()[:-1]]
#       term = " ".join(term_list)
#       hypernyms = [hypernym.replace("\n", "") for hypernym in gold_line.split("\t")]
#       hypernyms_dict[term] = hypernyms

#   return hypernyms_dict

import random

def load_data(data_path, gold_path):
    hypernyms_dict = {}
    with open(data_path, "r", encoding='utf-8') as data_file, open(gold_path, "r", encoding='utf-8') as gold_file:
        # Read all lines from both files
        data_lines = data_file.readlines()
        gold_lines = gold_file.readlines()

        # Combine lines into pairs
        combined_lines = list(zip(data_lines, gold_lines))

        # Shuffle the combined lines
        random.shuffle(combined_lines)

        # Take a random sample of 20 pairs
        sample_pairs = random.sample(combined_lines, k=30)

        # Process sampled pairs
        for data_line, gold_line in sample_pairs:
            term_list = [term for term in data_line.split()[:-1]]
            term = " ".join(term_list)
            hypernyms = [hypernym.replace("\n", "") for hypernym in gold_line.split("\t")]
            hypernyms_dict[term] = hypernyms

    return hypernyms_dict

# %%
# path = "/content/gdrive/MyDrive/"

# training data
train_hypernyms = load_data("1B.italian.training.data.txt", "1B.italian.training.gold.txt")

# test data
test_hypernyms = load_data("1B.italian.test.data.txt", "1B.italian.test.gold.txt")

# %%
print(train_hypernyms)

# %% [markdown]
# #Generate Distractors from italian dataset
# 

# %%
# !git clone https://github.com/napolux/paroleitaliane.git
# %cd /content/Parole_Italiane

# %%
# !wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.it.300.bin.gz
# !gunzip cc.it.300.bin.gz
# !pip install fasttext

# %%
# get the italian word list from the file 280000_parole_italiane.txt
with open('280000_parole_italiane.txt', 'r') as file:
  italian_word_list=[]
  for line in file:
    italian_word_list.append(line)


# %%
import fasttext
import numpy as np
import fasttext.util

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def check_letter(word1, word2):

    common_letters = set(word1).intersection(set(word2))

    threshold = len(word1) * 0.4

    if len(common_letters) >= threshold:
        return False
    else:
        return True

# %%
def compute_embeddings(italian_word_list, model):
  word_list = italian_word_list
  embedding_words = []

  for word in word_list:
      word_vector = model.get_word_vector(word)
      embedding_words.append(word_vector)

  return embedding_words

ft = fasttext.load_model('cc.it.300.bin')
embedding_list = compute_embeddings(italian_word_list, ft)

# %%

def find_distractors(italian_word_list, embedding_list ,target_word, number, model):

  target_vector = model.get_word_vector(target_word)

  word_list = italian_word_list
  similar_words = []
  th = 0.5

  while(len(similar_words)< number):
    similar_words = []
    for i, embed in enumerate(embedding_list):
        similarity = cosine_similarity(target_vector, embed)

        if similarity > th and check_letter(target_word, word_list[i]):
            similar_words.append(word_list[i].replace('\n', ''))

    th = th - 0.025


  results = []
  while(len(results)<3):
    index = random.randint(0, len(similar_words)-1)
    if similar_words[index] not in(results):
      results.append(similar_words[index])

  return results


ft = fasttext.load_model('cc.it.300.bin')

print(find_distractors(italian_word_list, embedding_list, "uccello", 3 ,ft))


# %% [markdown]
# # 4.0 Create entries

# %%
def save_jsonl(file_path, data, model, italian_word_list, embedding_list):
  id_seq = 0
  with open(file_path, "w") as output_file:
    for term, hypernyms in data.items():
      for hypernym in hypernyms:
        distractors = find_distractors(italian_word_list,embedding_list, hypernym, 3, model)
        entries = (hypernym, *distractors)
        choices = list(entries)
        random.shuffle(choices)
        reformatted_json_data = {
              'id' : id_seq,
              'text': term,
              'choices': choices,
              'label' : choices.index(hypernym)
        }
        json.dump(reformatted_json_data, output_file)
        output_file.write("\n")
        id_seq +=1

# %%
def read_lines_jsonl(file_path, num_lines):
  with open(file_path, 'r') as f:
    json_list = list(f)
    for line in json_list[:num_lines]:
      data = json.loads(line)
      print(data)

# %%
save_jsonl("hypernym_discovery-task26-train-data.jsonl", train_hypernyms, ft, italian_word_list, embedding_list)
read_lines_jsonl("hypernym_discovery-task26-train-data.jsonl", num_lines = 10)

# %%
save_jsonl("hypernym_discovery-task26-test-data.jsonl", train_hypernyms, ft, italian_word_list, embedding_list)
read_lines_jsonl("hypernym_discovery-task26-test-data.jsonl", num_lines = 10)

# %% [markdown]
# ## compute some Metrics 

# %%
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_metrics(data_file, num_samples):
    similarity_list, distractors_list = [], []
    with open(data_file, "r") as f:
        json_data = list(f)[:num_samples]
        for line in json_data:
            pair = json.loads(line)
            text = pair['text']
            choices = pair['choices']

            input_embeddings = ft.get_word_vector(text)
            for distractor in choices:
                distractor_embeddings = ft.get_word_vector(distractor)
                # Compute semantic similarity
                similarity = cosine_similarity([input_embeddings], [distractor_embeddings])[0][0]
                similarity_list.append(similarity)

                # Distractors list to compute lexical diversity
                distractors_list.append(distractor)

    avg_similarity = np.mean(similarity_list)
    lexical_diversity = len(set(distractors_list)) / len(distractors_list)

    print("Average Semantic Similarity:", avg_similarity)
    print("Average Lexical Diversity:", lexical_diversity)

print("Metrics for training data:")
compute_metrics("hypernym_discovery-task26-train-data.jsonl", num_samples=176)
print("\nMetrics for test data:")
compute_metrics("hypernym_discovery-task26-test-data.jsonl", num_samples=176)



