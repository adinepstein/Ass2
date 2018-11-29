import utils
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import windowTaggerModel  as wtm
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
POS=True
# file paths
train_path="C:\\DeepLearning\\ner\\train.txt"
dev_path="C:\\DeepLearning\\ner\\dev.txt"
test_path ="C:\\DeepLearning\\ner\\test.txt"

save_model_path= "model1_ner.pt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def main():
    word_to_index = {PAD: 0, UNKOWN: 1}
    index_to_word = [PAD, UNKOWN]
    label_to_index = {PAD: 0}
    index_to_label = [PAD]
    word_to_label = {}
    word_and_label_structures=[word_to_index,index_to_word,label_to_index,index_to_label,word_to_label]
    # upload train data
    word_to_index, index_to_word, label_to_index, index_to_label, word_to_label=utils.set_data_and_indexs(train_path, word_and_label_structures)
    #create embedder
    vocab_size = len(word_to_index)
    embedder = nn.Embedding(vocab_size, DIM_EMBEDDING)

    train_sequences = utils.sentences_to_sequences(utils.data_to_sentences(train_path),word_to_label,index_to_label)
    dev_sequences = utils.sentences_to_sequences(utils.data_to_sentences(dev_path),word_to_label,index_to_label)
    test_sequences = utils.sentences_to_sequences(utils.data_to_sentences(test_path),word_to_label,index_to_label)
    target_size = len(index_to_label)
    model = wtm.WindowTaggerModeler(target_size, DIM_EMBEDDING, CONTEXT_SIZE, DIM_LAYER, embedder,False)
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        random.shuffle(train_sequences)
        model.train()
        model.zero_grad()
        train_loss, train_acc = utils.run_single_epoch(train_sequences, model, optimizer,BATCH_SIZE,CONTEXT_SIZE,word_to_index,label_to_index,index_to_label, True, POS)
        model.eval()
        dev_loss, dev_acc = utils.run_single_epoch(dev_sequences, model, optimizer,BATCH_SIZE,CONTEXT_SIZE,word_to_index,label_to_index,index_to_label, False, POS)
        print("{} - train loss {} train-accuracy {} dev loss {}  dev-accuracy {}".format(epoch, train_loss, train_acc,
                                                                                         dev_loss, dev_acc))
    torch.save(model.state_dict(), save_model_path)
    # # Load model
    # model.load_state_dict(torch.load(model_save_path))
    #
    # # Evaluation pass.
    # _, test_acc = run_single_epoch(test_sequences, model,optimizer,optimizer,BATCH_SIZE,CONTEXT_SIZE,word_to_index,label_to_index,index_to_label, False,POS)
    # print("Test Accuracy: {:.3f}".format(test_acc))




if __name__ == '__main__':
    main()
