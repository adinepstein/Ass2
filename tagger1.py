import utils
import numpy as np
import random
import torch
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class WindowTaggerModeler(nn.Module):
    def __init__(self,vocab_size,output_size, dim_embedding,context_size,dim_layer):
        super(WindowTaggerModeler,self).__init__()
        self.train
        self.embeddings=nn.Embedding(vocab_size,dim_embedding)
        self.linear1=nn.Linear(context_size* dim_embedding,dim_layer,True)
        self.linear2=nn.Linear(dim_layer,output_size,True)

    def forward(self,inputs, labels,cur_batch_size):
        embeds=self.embeddings(inputs)
        embeds=embeds.view(cur_batch_size,CONTEXT_SIZE*DIM_EMBEDDING)
        out=torch.tanh(self.linear1(embeds))
        out=self.linear2(out)
        probs = F.softmax(out,dim=1)
        predicted_tags= torch.argmax(probs,1)
        loss_function = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
        flat_labels=labels.view(cur_batch_size)
        loss=loss_function(probs,flat_labels)

        return loss, predicted_tags


def word_to_index_f(word,word_to_index):
    if word in word_to_index:
        return word_to_index[word]
    else:
        return 1
def label_to_index_f(label,label_to_index):
    if label in label_to_index:
        return label_to_index[label]
    else:
        return 1

def do_pass(data,word_to_index,label_to_index,index_to_label,model,optimizer,train,pos):
        loss = 0
        match = 0
        total = 0

        for start in range(0, len(data), BATCH_SIZE):
            batch = data[start:start + BATCH_SIZE]
            # if start % 4000 == 0 and start > 0:
            #     print(loss, match / total)
            cur_batch_size = len(batch)
            input_array = torch.zeros((cur_batch_size, CONTEXT_SIZE)).long()
            output_array = torch.zeros((cur_batch_size, 1)).long()
            for n, (seq, label) in enumerate(batch):
                seq_index = torch.tensor([word_to_index_f(s,word_to_index) for s in seq], dtype=torch.long)
                label_index = torch.tensor(label_to_index_f(label,label_to_index), dtype=torch.long)
                input_array[n, :len(seq_index)] = seq_index
                output_array[n] = label_index
            if torch.cuda.is_available():
                input_array, output_array = input_array.to(device), output_array.to(device)
            batch_loss, output = model(input_array, output_array, cur_batch_size)
            if train:
                batch_loss.backward()
                optimizer.step()
                model.zero_grad()
            loss+=batch_loss.item()
            predicted =output.cpu().data.numpy()
            if pos:
                for (_,g), a in zip(batch,predicted):
                    total+=1
                    g_i=label_to_index_f(g,label_to_index)
                    if g_i==a:
                        match+=1
            else:
                for (_, g), a in zip(batch, predicted):
                    if (index_to_label[a]=='O' and g=='O'):
                        pass
                    else:
                        total += 1
                        if label_to_index_f(g,label_to_index) == a:
                            match += 1
        return loss/(len(data)/BATCH_SIZE), match/total


def train_and_test_model(train_path,dev_path,test_path,model_dave_path,pos):
    word_to_label, word_to_index, index_to_word, label_to_index, index_to_label = utils.set_data_and_indexs(train_path)
    train_sequences = utils.sentences_to_sequences(utils.data_to_sentences(train_path), word_to_label)
    dev_sequences = utils.sentences_to_sequences(utils.data_to_sentences(dev_path), word_to_label)
    test_sequences = utils.sentences_to_sequences(utils.data_to_sentences(test_path),word_to_label)
    vocab_size = len(index_to_word)
    target_size = len(index_to_label)
    model = WindowTaggerModeler(vocab_size, target_size, DIM_EMBEDDING, CONTEXT_SIZE, DIM_LAYER)
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        random.shuffle(train_sequences)
        model.train()
        model.zero_grad()
        train_loss, train_acc = do_pass(train_sequences, word_to_index, label_to_index,index_to_label, model, optimizer, True,pos)
        model.eval()
        dev_loss, dev_acc = do_pass(dev_sequences, word_to_index, label_to_index,index_to_label, model, optimizer, False,pos)
        print("{} - train loss {} train-accuracy {} dev loss {}  dev-accuracy {}".format(epoch, train_loss, train_acc,
                                                                                         dev_loss, dev_acc))
        torch.save(model.state_dict(), model_dave_path)
        # # Load model
        # model.load_state_dict(torch.load('tagger.pt.model'))
        #
        # # Evaluation pass.
        # _, test_acc = do_pass(test_sequences, word_to_index, label_to_index,index_to_label, model,optimizer, False,pos)
        # print("Test Accuracy: {:.3f}".format(test_acc))

def main():
    train_and_test_model("C:\\DeepLearning\\ner\\train.txt","C:\\DeepLearning\\ner\\dev.txt","C:\\DeepLearning\\ner\\test.txt","model2_ner.pt",False)





if __name__ == '__main__':
    main()
