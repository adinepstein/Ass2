import numpy as np
import random
import torch
import tagger1
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(0)

PAD ="__PAD__"
UNKOWN = "__UNKOWN__"
DIM_EMBEDDING= 50
CONTEXT_SIZE = 5
DIM_LAYER = 100
LR = 0.04
BATCH_SIZE= 30
EPOCHS = 100

word_to_index= {PAD:0,UNKOWN:1}
index_to_word= [PAD, UNKOWN]
label_to_index = {PAD:0}
index_to_label = [PAD]
word_to_label = {}
pretrained_emnedding_list =[]

def load_pretrained_embedding_vectors(words_file,vectors_file):
    vecs=np.loadtxt(vectors_file,dtype=float)
    vocabulary=np.loadtxt(words_file,dtype=str)
    pretrained_dic= {}
    for i in range(0,len(vocabulary)):
        pretrained_dic[vocabulary[i]] = vecs[i]
    return pretrained_dic

def set_pretrained_matrix(pretained_dic):
    scale = np.sqrt(3/DIM_EMBEDDING)
    for word in index_to_word:
        if word.lower() in pretained_dic:
            pretrained_emnedding_list.append(np.array(pretained_dic[word]))
        else:
            random_vector= np.random.uniform(-scale,scale,DIM_EMBEDDING)
            pretrained_emnedding_list.append(random_vector)


if __name__=='__main__':
        print (" ")
        # pretrained_tensor= torch.FloatTensor(pretrained_list)
        # self.embedding= nn.Embedding.from_pretrained(pretrained_tensor,freeze=False)
        # self.linear1=


