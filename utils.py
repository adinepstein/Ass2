import utils
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import torch.optim as optim
import pylab
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

# inserts all words and indexs to the structures
def set_data_and_indexs(filename,word_label_structures):
    word_to_index, index_to_word, label_to_index, index_to_label, word_to_label=word_label_structures
    data= read_data(filename)
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
# returns the label of the given word
def word_to_label_f(word,word_to_label,index_to_label):
    if word in word_to_label:
        return word_to_label[word]
    else:
        return index_to_label[1]

# loads the data as sentences from the file
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

# creates sequances of 5 words from the given sentences
def sentences_to_sequences(sentences,word_to_label,index_to_label):
    sequences= []
    for sentence in sentences:
        for i in range(2,len(sentence)-2):
            s = ([sentence[i-2],sentence[i-1],sentence[i],sentence[i+1],sentence[i+2]],word_to_label_f(sentence[i],word_to_label,index_to_label))
            sequences.append(s)
    return sequences

# add prefix and suffix to the word and index structures
def add_prefix_suffix(seq,word_to_index,context_size):
    prex_suff_seq =torch.zeros(( 3,context_size)).long()
    for i,s in enumerate(seq):
         if len(s)>=3:
            prex_suff_seq[0,i]=word_to_index_f(s[:3],word_to_index)
            prex_suff_seq[2, i]=word_to_index_f(s[-3:],word_to_index)
         prex_suff_seq[1, i] = word_to_index_f(s, word_to_index)
    return prex_suff_seq



# run a full epoch
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
# run a full epoch with the added prefix and suffix
def run_single_epoch_with_prex_suffix(data, model, optimizer,batch_size,context_size,word_to_index,label_to_index,index_to_label, train, pos):
    loss = 0
    match = 0
    total = 0
    for start in range(0, len(data), batch_size):
        batch = data[start:start + batch_size]
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

    # trains the full model- saving each epoch in order to be able to choose the best epoch
def train_model(model,train_sequences,dev_sequences,word_to_index, label_to_index, index_to_label,epochs,lr,pos,batch_size,context_size,dim_layer,save_model_path,prefix_suffix):
    train_loss_list = []
    dev_loss_list = []
    train_acc_list = []
    dev_acc_list = []
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs):
        random.shuffle(train_sequences)
        model.train()
        model.zero_grad()
        if prefix_suffix:
            train_loss, train_acc = run_single_epoch_with_prex_suffix(train_sequences, model, optimizer, batch_size, context_size,
                                                           word_to_index, label_to_index, index_to_label, True, pos)
            model.eval()
            dev_loss, dev_acc = run_single_epoch_with_prex_suffix(dev_sequences, model, optimizer, batch_size, context_size,
                                                       word_to_index, label_to_index, index_to_label, False, pos)
        else:
            train_loss, train_acc = run_single_epoch(train_sequences, model, optimizer, batch_size, context_size,
                                                       word_to_index, label_to_index, index_to_label, True, pos)
            model.eval()
            dev_loss, dev_acc = run_single_epoch(dev_sequences, model, optimizer, batch_size, context_size,
                                                   word_to_index, label_to_index, index_to_label, False, pos)
        print("{} - train loss {} train-accuracy {} dev loss {}  dev-accuracy {}".format(epoch, train_loss, train_acc,
                                                                                         dev_loss, dev_acc))
        dev_loss_list.append(dev_loss)
        dev_acc_list.append(dev_acc)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        torch.save(model.state_dict(), save_model_path+str(epoch))
    write_loss_accuracy_to_file(train_loss_list, train_acc_list, dev_loss_list, dev_acc_list, pos, "tagger1", lr,
                                      batch_size, dim_layer, "tagger1_ner_loss_and_accuracy.txt")
# given a model and test data - prints the predictions in the given file
def predict_test(data, model, batch_size,context_size,word_to_index,index_to_label,model_saved_file, prediction_file,prefix_suffix):
    model.load_state_dict(torch.load(model_saved_file))
    model.eval()
    words=[]
    for s in data:
        words.append(s[0][2])
    for start in range(0, len(data), batch_size):
        batch = data[start:start + batch_size]
        cur_batch_size = len(batch)
        if prefix_suffix:
            input_array = torch.zeros((cur_batch_size, 3, context_size)).long()
            output_array = torch.zeros((cur_batch_size, 1)).long()
            for n, (seq, label) in enumerate(batch):
                seq_index = add_prefix_suffix(seq, word_to_index, context_size)
                input_array[n, :, :] = seq_index
        else:
            input_array = torch.zeros((cur_batch_size, context_size)).long()
            output_array = torch.zeros((cur_batch_size, 1)).long()
            for n, (seq, label) in enumerate(batch):
                seq_index = torch.tensor([word_to_index_f(s,word_to_index) for s in seq], dtype=torch.long)
                input_array[n, :len(seq_index)] = seq_index
        if torch.cuda.is_available():
            input_array, output_array = input_array.to(device), output_array.to(device)
        _, output = model(input_array, output_array, cur_batch_size)
        predicted = output.cpu().data.numpy()
        f=open(prediction_file,"w")
        for i,p in enumerate(predicted):
            f.write(words[i] + " " + index_to_label[p]+ "\n")
        f.close()


# save the loss and accuracy of train and dev
def write_loss_accuracy_to_file(train_loss,train_accuracy,dev_loss,dev_accuracy,pos,tagger,lr,batch,layer_dim,file_name):
    f=open(file_name,'a')
    f.write("pos(true)/ner(false) " + str(pos) + " | question: " + tagger + " | lr: " + str(lr)+ "  | batch size " + str(batch)+ " | dim layer "+ str(layer_dim)+ " | "+ str(datetime.datetime.now()) + "\n")
    f.write("Loss\n")
    size=len(train_loss)
    for i in range(size):
        f.write(str(i)+ " " + str(train_loss[i])+ " " + str(dev_loss[i])+ "\n")
    f.write("accuracy\n")
    for i in range(size):
        f.write(str(i) + " " + str(train_accuracy[i]) + " " + str(dev_accuracy[i]) + "\n")
    f.write("end\n\n")
    f.close()

# reads the loss and accuracy from file
def read_loss_and_accuracy(file_name):
    f=open(file_name,"r")
    train_loss=[]
    dev_loss=[]
    train_accuracy=[]
    dev_accuracy=[]
    f.readline()
    f.readline()
    line=f.readline()
    while line != "accuracy\n":
        split= line.strip().split(" ")
        print(split[0])
        train_loss.append(float(split[1]))
        dev_loss.append(float(split[2]))
        line = f.readline()
    line=f.readline()
    while line != "end\n":
        split=line.strip().split(" ")
        print(split[0])
        train_accuracy.append(float(split[1]))
        dev_accuracy.append(float(split[2]))
        line= f.readline()
    f.close()
    return train_loss,dev_loss,train_accuracy,dev_accuracy

# creates a plot for the loss and accuracy
def plotdata(data,x_axis_name,color,y_axis_name,title):
    size=len(data)
    x=[]
    for i in range(size):
        x.append(i+1)
    pylab.plot(x,data,color)
    pylab.title(title)
    pylab.xlabel(x_axis_name)
    pylab.ylabel(y_axis_name)
    pylab.show()

if __name__ == '__main__':
    train_loss, dev_loss, train_accuracy, dev_accuracy=read_loss_and_accuracy("tagger2_pos_loss_and_accuracy.txt")
    # plotdata(dev_loss,"Iteration number","-r","Loss","part 1 - POS (Dev Loss)")
    plotdata(dev_accuracy, "Iteration number", "-b", "Accuracy", "part 1 - POS (Dev Accuracy)")