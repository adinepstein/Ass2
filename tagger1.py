import utils
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

PAD ="__PAD__"
UNKOWN = "__UNKOWN__"
DIM_EMBEDDING= 50
CONTEXT_SIZE = 5
DIM_LAYER = 100
LR = 0.01
BATCH_SIZE= 100
EPOCHS = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class WindowTaggerModeler(nn.Module):
    def __init__(self,vocab_size,output_size, dim_embedding,context_size,dim_layer):
        super(WindowTaggerModeler,self).__init__()
        self.embeddings=nn.Embedding(vocab_size,dim_embedding)
        self.linear1=nn.Linear(context_size* dim_embedding,dim_layer,True)
        self.linear2=nn.Linear(dim_layer,output_size,True)

    def forward(self,inputs):
        embeds=self.embeddings(inputs).view(1,-1)
        out=torch.tanh(self.linear1(embeds))
        out=self.linear2(out)
        probs= F.softmax(out,dim=1)
        return probs



# def pars_data(filename):
#     word_to_label, word_to_index, index_to_word, label_to_index, index_to_label = utils.set_data_and_indexs(filename)
#     sentences= utils.data_to_sentences(filename)
#     sequences= utils.sentences_to_sequences(sentences,word_to_label)
#     return sequences


def main():
    word_to_label, word_to_index, index_to_word, label_to_index, index_to_label = utils.set_data_and_indexs("C:\\DeepLearning\\pos\\train.txt")
    sequences = utils.sentences_to_sequences(utils.data_to_sentences("C:\\DeepLearning\\pos\\train.txt"), word_to_label)
    loss_function=nn.CrossEntropyLoss(ignore_index=0)
    vocab_size=len(index_to_word)
    target_size =len(index_to_label)
    model= WindowTaggerModeler(vocab_size,target_size,DIM_EMBEDDING,CONTEXT_SIZE,DIM_LAYER)
    a= torch.cuda.device_count()
    print(a)
    model = nn.DataParallel(model)
    model.to(device)
    optimizer = optim.SGD(model.parameters(),lr=LR)
    train_losses=[]
    train_accuracy = []
    dev_losses = []
    dev_accuracy = []
    for epoch in range(EPOCHS):
        random.shuffle(sequences)
        total_loss=0
        for n, (seq, label) in enumerate(sequences):
            seq_index= torch.tensor([word_to_index[s] for s in seq],dtype=torch.long)
            label_index= torch.tensor([label_to_index[label]],dtype=torch.long)
            if torch.cuda.is_available():
                seq_index, label_index = seq_index.to(device), label_index.to(device)
                # input_array[n,:len(seq_index)]=seq_index
                # output_array[n]=label_index
                # input_array, output_array = input_array.to(device), output_array.to(device)

            probs = model(seq_index)
            loss = loss_function(probs,label_index)
            if n% BATCH_SIZE==0 and n>0:
                model.zero_grad()
                loss.backward()
                optimizer.step()

        train_losses.append(loss.item())
        print(loss.item())



if __name__ == '__main__':
    main()
