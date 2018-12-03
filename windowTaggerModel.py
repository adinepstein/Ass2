import utils
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(0)




class WindowTaggerModeler(nn.Module):
    def __init__(self,output_size, dim_embedding,context_size,dim_layer,embedder,with_prefix_suffix):
        super(WindowTaggerModeler,self).__init__()
        self.train
        self.embeddings=embedder
        self.context_size= context_size
        self.dim_embedding= dim_embedding
        self.linear1=nn.Linear(context_size* dim_embedding,dim_layer,True)
        self.linear2=nn.Linear(dim_layer,output_size,True)
        self.with_prefix_suffix=with_prefix_suffix

    def forward(self,inputs, labels,cur_batch_size):
        if self.with_prefix_suffix:
            embeds=self.set_sum_embeds(inputs)
        else:
            embeds=self.embeddings(inputs)
        embeds=embeds.view(cur_batch_size,self.context_size*self.dim_embedding)
        out=torch.tanh(self.linear1(embeds))
        out=self.linear2(out)
        probs = F.softmax(out,dim=1)
        predicted_tags= torch.argmax(probs,1)
        loss_function = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
        flat_labels=labels.view(cur_batch_size)
        loss=loss_function(probs,flat_labels)

        return loss, predicted_tags

    # gets the ebmedding vectors of the prefix, suffix and the word and sums them up
    def set_sum_embeds(self,inputs):
        prefix= inputs[:,0,:]
        sequence=inputs[:,1,:]
        suffix = inputs [:,2,:]
        embed_prex=self.embeddings(prefix)
        embed_seq= self.embeddings(sequence)
        embed_suff= self.embeddings(suffix)
        return embed_prex+embed_seq+embed_suff