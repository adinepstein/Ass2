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
DIM_LAYER = 150
LR = 0.045
BATCH_SIZE= 60
EPOCHS = 60

# choose what to clasify POS-True or NER-False
POS=False
#choose if you want to train the model(True) or load the trained model and predict(False)
TRAIN_MODEL=False
#choose if adding prefix and suffix(True) or not (False)
PREFIX_SUFFIX=False

# file paths
train_path="ner\\train.txt"
dev_path="ner\\dev.txt"
test_path ="ner\\test.txt"

save_model_path= "model12_ner.pt"

prediction_results_path="test1.ner"
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
    if TRAIN_MODEL:
        utils.train_model(model,train_sequences,dev_sequences,word_to_index,label_to_index,index_to_label,EPOCHS,LR,POS,BATCH_SIZE,CONTEXT_SIZE,DIM_LAYER,save_model_path,PREFIX_SUFFIX)
    else:
        utils.predict_test(test_sequences,model,len(test_sequences),CONTEXT_SIZE,word_to_index,index_to_label,save_model_path,prediction_results_path,PREFIX_SUFFIX)



if __name__ == '__main__':
    main()
