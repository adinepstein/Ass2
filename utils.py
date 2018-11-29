import utils
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PAD ="__PAD__"
UNKOWN = "__UNKOWN__"

def read_data(filename):
    data =[]
    with open(filename) as dataSource:
        i=0
        for line in dataSource:
            i+=1
            if line != '\n':
                split = line.strip().split()
                word = split[0]
                label = split[1]
                data.append((word, label))
    return data

def set_data_and_indexs(filename,word_label_structures):
    word_to_index, index_to_word, label_to_index, index_to_label, word_to_label=word_label_structures
    data= utils.read_data(filename)
    for word , label in data:
        word_to_label[word]=label
        if word not in word_to_index:
            word_to_index[word]=len(word_to_index)
            index_to_word.append(word)
        if label not in label_to_index:
            label_to_index[label]= len(label_to_index)
            index_to_label.append(label)
    word_label_structures=[word_to_index,index_to_word,label_to_index,index_to_label,word_to_label]
    return word_label_structures

def word_to_index_f(word,word_to_index):
    if word in word_to_index:
        return word_to_index[word]
    else:
        return 1

def word_to_label_f(word,word_to_label,index_to_label):
    if word in word_to_label:
        return word_to_label[word]
    else:
        return index_to_label[1]

def data_to_sentences(filename):
    all_sentences=[]
    with open(filename) as dataSource:
        sentence =[PAD,PAD]
        for line in dataSource:
            if line!="\n":
                split= line.strip().split()
                sentence.append(split[0])
            else:
                sentence.append(PAD)
                sentence.append(PAD)
                all_sentences.append(sentence)
                sentence=[PAD,PAD]
    return all_sentences

def sentences_to_sequences(sentences,word_to_label,index_to_label):
    sequences= []
    for sentence in sentences:
        for i in range(2,len(sentence)-2):
            s = ([sentence[i-2],sentence[i-1],sentence[i],sentence[i+1],sentence[i+2]],word_to_label_f(sentence[i],word_to_label,index_to_label))
            sequences.append(s)
    return sequences

def add_prefix_suffix(seq,word_to_index,context_size):
    prex_suff_seq =torch.zeros(( 3,context_size)).long()
    for i,s in enumerate(seq):
         if len(s)>=3:
            prex_suff_seq[0,i]=word_to_index_f(s[:3],word_to_index)
            prex_suff_seq[2, i]=word_to_index_f(s[-3:],word_to_index)
         prex_suff_seq[1, i] = word_to_index_f(s, word_to_index)
    return prex_suff_seq




def run_single_epoch(data, model, optimizer,batch_size,context_size,word_to_index,label_to_index,index_to_label, train, pos):
    loss = 0
    match = 0
    total = 0
    for start in range(0, len(data), batch_size):
        batch = data[start:start + batch_size]
        # if start % 4000 == 0 and start > 0:
        #     print(loss, match / total)
        cur_batch_size = len(batch)
        input_array = torch.zeros((cur_batch_size, context_size)).long()
        output_array = torch.zeros((cur_batch_size, 1)).long()
        for n, (seq, label) in enumerate(batch):
            seq_index = torch.tensor([word_to_index_f(s,word_to_index) for s in seq], dtype=torch.long)
            label_index = torch.tensor(label_to_index[label], dtype=torch.long)
            input_array[n, :len(seq_index)] = seq_index
            output_array[n] = label_index
        if torch.cuda.is_available():
            input_array, output_array = input_array.to(device), output_array.to(device)
        batch_loss, output = model(input_array, output_array, cur_batch_size)
        if train:
            batch_loss.backward()
            optimizer.step()
            model.zero_grad()
        loss += batch_loss.item()
        predicted = output.cpu().data.numpy()
        if pos:
            for (_, g), a in zip(batch, predicted):
                total += 1
                g_i = label_to_index[g]
                if g_i == a:
                    match += 1
        else:
            for (_, g), a in zip(batch, predicted):
                if (index_to_label[a] == 'O' and g == 'O'):
                    pass
                else:
                    total += 1
                    if label_to_index[g] == a:
                        match += 1
    return loss / (len(data) / batch_size), match / total

def run_single_epoch_with_prex_suffix(data, model, optimizer,batch_size,context_size,word_to_index,label_to_index,index_to_label, train, pos):
    loss = 0
    match = 0
    total = 0
    for start in range(0, len(data), batch_size):
        batch = data[start:start + batch_size]
        # if start % 4000 == 0 and start > 0:
        #     print(loss, match / total)
        cur_batch_size = len(batch)
        input_array = torch.zeros((cur_batch_size, 3,context_size)).long()
        output_array = torch.zeros((cur_batch_size, 1)).long()
        for n, (seq, label) in enumerate(batch):
            seq_index = add_prefix_suffix(seq,word_to_index,context_size)
            label_index = torch.tensor(label_to_index[label], dtype=torch.long)
            input_array[n,:,:] = seq_index
            output_array[n] = label_index
        if torch.cuda.is_available():
            input_array, output_array = input_array.to(device), output_array.to(device)
        batch_loss, output = model(input_array, output_array, cur_batch_size)
        if train:
            batch_loss.backward()
            optimizer.step()
            model.zero_grad()
        loss += batch_loss.item()
        predicted = output.cpu().data.numpy()
        if pos:
            for (_, g), a in zip(batch, predicted):
                total += 1
                g_i = label_to_index[g]
                if g_i == a:
                    match += 1
        else:
            for (_, g), a in zip(batch, predicted):
                if (index_to_label[a] == 'O' and g == 'O'):
                    pass
                else:
                    total += 1
                    if label_to_index[g] == a:
                        match += 1
    return loss / (len(data) / batch_size), match / total

