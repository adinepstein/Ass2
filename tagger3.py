import numpy as np
import random
import torch
import tagger1
import torch.nn as nn
import utils
import windowTaggerModel as wtm
import torch.nn.functional as F
import torch.optim as optim
import tagger2
torch.manual_seed(0)

PAD ="__PAD__"
UNKOWN = "__UNKOWN__"
DIM_EMBEDDING= 50
CONTEXT_SIZE = 5
DIM_LAYER = 100
LR = 0.04
BATCH_SIZE= 30
EPOCHS = 100

# choose what to clasify POS-True or NER-False
POS=False
# choose embedding vectors, pretranined- True or untrained- False
PRETRAINED= True
# file paths
train_path="C:\\DeepLearning\\ner\\train.txt"
dev_path="C:\\DeepLearning\\ner\\dev.txt"
test_path ="C:\\DeepLearning\\ner\\test.txt"

#pretrained embedding files
word_path="C:\\DeepLearning\\pretrained\\vocab.txt"
vectors_path="C:\\DeepLearning\\pretrained\\wordVectors.txt"


save_model_path= "model1_ner.pt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def add_prefix_suffix_to_word_index(word_to_index,index_to_word,word_to_label):
    temp_index_to_word=[]
    for word in index_to_word:
        temp_index_to_word.append(word)
    for word in temp_index_to_word:
        if len(word)>=3:
            prefix=word[:3]
            suffix = word[-3:]
            if prefix not in word_to_label:
                word_to_index[prefix]=len(word_to_index)
                index_to_word.append(prefix)
            if suffix not in word_to_label:
                word_to_index[suffix]= len(word_to_index)
                index_to_word.append(suffix)
    return word_to_index,word_to_label,index_to_word


def main():
    word_to_index = {PAD: 0, UNKOWN: 1}
    index_to_word = [PAD, UNKOWN]
    label_to_index = {PAD: 0}
    index_to_label = [PAD]
    word_to_label = {}
    word_and_label_structures=[word_to_index,index_to_word,label_to_index,index_to_label,word_to_label]
    # upload train data
    word_to_index, index_to_word, label_to_index, index_to_label, word_to_label=utils.set_data_and_indexs(train_path, word_and_label_structures)
    word_to_index, word_to_label, index_to_word = add_prefix_suffix_to_word_index(word_to_index,index_to_word,word_to_label)
    #create embedder
    vocab_size = len(word_to_index)
    if PRETRAINED:
        pretrained_dic = tagger2.load_pretrained_embedding_vectors(word_path, vectors_path)
        pretrained_emnedding_list = tagger2.set_pretrained_list(pretrained_dic, index_to_word)
        pretrained_tensor = torch.FloatTensor(pretrained_emnedding_list)
        embedder = nn.Embedding.from_pretrained(pretrained_tensor, freeze=False)
    else:
        embedder = nn.Embedding(vocab_size, DIM_EMBEDDING)

    train_sequences = utils.sentences_to_sequences(utils.data_to_sentences(train_path),word_to_label,index_to_label)
    dev_sequences = utils.sentences_to_sequences(utils.data_to_sentences(dev_path),word_to_label,index_to_label)
    test_sequences = utils.sentences_to_sequences(utils.data_to_sentences(test_path),word_to_label,index_to_label)
    target_size = len(index_to_label)
    model = wtm.WindowTaggerModeler(target_size, DIM_EMBEDDING, CONTEXT_SIZE, DIM_LAYER, embedder,True)
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        random.shuffle(train_sequences)
        model.train()
        model.zero_grad()
        train_loss, train_acc = utils.run_single_epoch_with_prex_suffix(train_sequences, model, optimizer,BATCH_SIZE,CONTEXT_SIZE,word_to_index,label_to_index,index_to_label, True, POS)
        model.eval()
        dev_loss, dev_acc = utils.run_single_epoch_with_prex_suffix(dev_sequences, model, optimizer,BATCH_SIZE,CONTEXT_SIZE,word_to_index,label_to_index,index_to_label, False, POS)
        print("{} - train loss {} train-accuracy {} dev loss {}  dev-accuracy {}".format(epoch, train_loss, train_acc,
                                                                                         dev_loss, dev_acc))
    torch.save(model.state_dict(), save_model_path)
    # # Load model
    # model.load_state_dict(torch.load(model_save_path))
    #
    # # Evaluation pass.
    # _, test_acc = utils.run_single_epoch_with_prex_suffix(test_sequences, model,optimizer,optimizer,BATCH_SIZE,CONTEXT_SIZE,word_to_index,label_to_index,index_to_label, False,POS)
    # print("Test Accuracy: {:.3f}".format(test_acc))




if __name__ == '__main__':
    main()