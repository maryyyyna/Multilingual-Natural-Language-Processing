# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from torch import cuda
import jsonlines
import time

import random

import torch
import torch.nn as nn
from datasets import load_dataset, load_metric
from sklearn.metrics import classification_report, confusion_matrix
from transformers import pipeline
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaModel, RobertaTokenizer

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)

set_seed(42)

import argparse

import warnings
warnings.filterwarnings("ignore")

def map_labels(example):
  label_map = {'ENTAILMENT': 0, 'CONTRADICTION': 1, 'NEUTRAL': 2}
  example['label'] = label_map[example['label']]
  return example

def tokenize_function(examples):
  inputs = ["[CLS] " + p + " [SEP] " + h for p, h in zip(examples['premise'], examples['hypothesis'])]
  tokens = tokenizer(inputs,
                  padding = True,
                  truncation = True,
                  max_length = 400)
  return tokens

def compute_metrics(eval_pred):
    load_accuracy = load_metric("accuracy", trust_remote_code=True)
    load_f1 = load_metric("f1", trust_remote_code=True)
    load_precision = load_metric("precision", trust_remote_code=True)
    load_recall = load_metric("recall", trust_remote_code=True)

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Compute metrics
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels, average='weighted')["f1"]
    precision = load_precision.compute(predictions=predictions, references=labels, average='weighted')["precision"]
    recall = load_recall.compute(predictions=predictions, references=labels, average='weighted')["recall"]

    # Print classification report
    classification_rep = classification_report(labels, predictions)
    print(classification_rep)

    # Compute confusion matrix
    cm = confusion_matrix(labels, predictions)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',
                xticklabels=range(len(cm)),
                yticklabels=range(len(cm)))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}


def add_adversarial_examples(fever_data_augmented, file_name):
  ids = fever_data_augmented['train']['id']
  max_id = max(ids)
  num = 0

  with jsonlines.open(file_name) as reader:
    for dic in reader:
      new_id = int(max_id) + 1
      dic['id'] = str(new_id)
      fever_data_augmented['train'] = fever_data_augmented['train'].add_item(dic)
      num += 1

  print(f"Added {num} examples to the Train-Set \n")
  return fever_data_augmented['train']
     

def main(args):

  # Training Argurments
  batch_size = 32
  learning_rate = 1e-5
  weight_decay = 0.001
  epochs = 1
  device = "cuda"
  model_name = "microsoft/deberta-v3-small"
  set_seed(42)

  # Initialize the model
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  data_collator = DataCollatorWithPadding(tokenizer = tokenizer)


  if args.mode == 'train' and args.data == 'original':
    model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                              ignore_mismatched_sizes = True,
                                                              output_attentions = False,
                                                              output_hidden_states = False,
                                                              num_labels = 3)
    model.to(device)

    #Load dataset
    fever_data = load_dataset("tommasobonomo/sem_augmented_fever_nli")
    fever_data['train'] = fever_data['train'].map(map_labels)
    fever_data['validation'] = fever_data['validation'].map(map_labels)
    fever_data['test'] = fever_data['test'].map(map_labels)
    tokenized_fever_data = fever_data.map(tokenize_function, batched = True)

    training_args = TrainingArguments(output_dir = "training_dir",
                                      num_train_epochs = epochs,
                                      per_device_train_batch_size = batch_size,
                                      warmup_steps = 500,
                                      weight_decay = weight_decay,
                                      save_strategy = "no",
                                      save_total_limit = 5,
                                      learning_rate = learning_rate,
                                    )



    trainer = Trainer(model = model,
                      args=training_args,
                      train_dataset = tokenized_fever_data["train"],
                      eval_dataset = tokenized_fever_data["validation"],
                      tokenizer = tokenizer,
                      data_collator = data_collator,
                      compute_metrics = compute_metrics,
                      )

    training_out = trainer.train()
    evaluation_out = trainer.evaluate()

    with open(r'./training_original.txt', 'a') as f:
      f.write('\n ' + '#'*10 + ' TRAINING ' + '#'*10 )
      f.write(training_out)
      f.write('\n ' + '#'*10 + ' EVALUATION ' + '#'*10 )
      f.write(evaluation_out)

  elif args.mode == 'train' and args.data == 'adversarial':
      fever_data_augmented = load_dataset("tommasobonomo/sem_augmented_fever_nli")
      add_adversarial_examples("neutral_adversarial_examples_first_approach.jsonl", fever_data_augmented)
      add_adversarial_examples("neutral_adversarial_examples_second_approach.jsonl", fever_data_augmented)
      add_adversarial_examples("contradiction_adversarial_examples_first_approach.jsonl", fever_data_augmented)
      add_adversarial_examples("contradiction_adversarial_examples_second_approach.jsonl", fever_data_augmented)
      add_adversarial_examples("contradiction_adversarial_examples_third_approach.jsonl", fever_data_augmented)

      fever_data_augmented['train'] = fever_data_augmented['train'].map(map_labels)
      fever_data_augmented['validation'] = fever_data_augmented['validation'].map(map_labels)
      fever_data_augmented['test'] = fever_data_augmented['test'].map(map_labels)
      tokenized_fever_data_augmented = fever_data_augmented.map(tokenize_function, batched = True)

      
      trainer = Trainer(model = model,
                  args = training_args,
                  train_dataset = tokenized_fever_data_augmented["train"],  # augmented data
                  eval_dataset = tokenized_fever_data_augmented["validation"], # aumented data
                  tokenizer = tokenizer,
                  data_collator = data_collator,
                  compute_metrics = compute_metrics,
                  )
      
      training_out = trainer.train()
      evaluation_out = trainer.evaluate()
  
      with open(r'./training_augmented.txt', 'a') as f:
        f.write('\n ' + '#'*10 + ' TRAINING ' + '#'*10 )
        f.write(training_out)
        f.write('\n ' + '#'*10 + ' EVALUATION ' + '#'*10 )
        f.write(evaluation_out)
      
  elif args.mode == 'test' and args.data == 'original':
      adversarial_test_set = load_dataset("tommasobonomo/sem_augmented_fever_nli")
      adversarial_test_set['test'] = adversarial_test_set['test'].map(map_labels)
      tokenized_adversarial_test_data = adversarial_test_set.map(tokenize_function, batched=True)

      test_out = trainer.evaluate(eval_dataset = tokenized_fever_data['test'])
      test_out_adverarial = trainer.evaluate(eval_dataset = tokenized_adversarial_test_data['test'])

      with open(r'./test_original.txt', 'a') as f:
        f.write('\n ' + '#'*10 + ' TEST FEVER' + '#'*10 )
        f.write(test_out)
        f.write('\n ' + '#'*10 + ' TEST ADVERSARIAL ' + '#'*10 )
        f.write(test_out_adverarial)

  elif args.mode == 'test' and args.data == 'adversarial':
      adversarial_test_set = load_dataset("tommasobonomo/sem_augmented_fever_nli")
      adversarial_test_set['test'] = adversarial_test_set['test'].map(map_labels)
      tokenized_adversarial_test_data = adversarial_test_set.map(tokenize_function, batched=True)

      test_out = trainer.evaluate(eval_dataset = tokenized_fever_data['test'])
      test_out_adverarial = trainer.evaluate(eval_dataset = tokenized_adversarial_test_data['test'])

      with open(r'./test_augmented.txt', 'a') as f:
        f.write('\n ' + '#'*10 + ' TEST FEVER' + '#'*10 )
        f.write(test_out)
        f.write('\n ' + '#'*10 + ' TEST ADVERSARIAL ' + '#'*10 )
        f.write(test_out_adverarial)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train, test, and evaluate the model')
    parser.add_argument('mode', choices=['train', 'test'], help = "The mode to 'train' or 'test' the model")

    parser.add_argument('--data',
                        choices = ['original', 'adversarial'],
                        required = False,
                        help = "The type of test set to use: 'original'  or 'adversarial', for the original test set and the adversarial one respectly.")

    args = parser.parse_args()
    print(args)
    main(args)