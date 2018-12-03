import numpy as np
import random
import torch
import tagger1
import torch.nn as nn
import utils
import windowTaggerModel as wtm
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(0)

PAD ="__PAD__"
UNKOWN = "__UNKOWN__"
DIM_EMBEDDING= 50
CONTEXT_SIZE = 5
DIM_LAYER = 100
LR = 0.01
BATCH_SIZE= 10
EPOCHS = 40

# choose what to clasify POS-True or NER-False
POS=True
#choose if you want to train the model(True) or load the trained model and predict(False)
TRAIN_MODEL=False
#choose if adding prefix and suffix(True) or not (False)
PREFIX_SUFFIX=False

# data paths
train_path="pos\\train.txt"
dev_path="pos\\dev.txt"
test_path ="pos\\test.txt"

#pretrained embedding files
word_path="vocab.txt"
vectors_path="wordVectors.txt"

#model and results file path
save_model_path= "model2_3.pt"
prediction_results_path="test3.pos"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_pretrained_embedding_vectors(words_file,vectors_file):
    vecs=np.loadtxt(vectors_file,dtype=float)
    vocabulary=np.loadtxt(words_file,dtype=str)
    pretrained_dic= {}
    for i in range(0,len(vocabulary)):
        pretrained_dic[vocabulary[i]] = vecs[i]
    return pretrained_dic

def set_pretrained_list(pretained_dic,index_to_word):
    pretrained_emnedding_list = []
    scale = np.sqrt(3/DIM_EMBEDDING)
    for word in index_to_word:
        if word.lower() in pretained_dic:
            pretrained_emnedding_list.append(np.array(pretained_dic[word.lower()]))
        else:
            random_vector= np.random.uniform(-scale,scale,DIM_EMBEDDING)
            pretrained_emnedding_list.append(random_vector)
    return pretrained_emnedding_list



def main():
    word_to_index = {PAD: 0, UNKOWN: 1}
    index_to_word = [PAD, UNKOWN]
    label_to_index = {PAD: 0}
    index_to_label = [PAD]
    word_to_label = {}

    word_and_label_structures = [word_to_index, index_to_word, label_to_index, index_to_label, word_to_label]
    # upload train data
    word_to_index, index_to_word, label_to_index, index_to_label, word_to_label = utils.set_data_and_indexs(train_path,
                                                                                                            word_and_label_structures)

    pretrained_dic=load_pretrained_embedding_vectors(word_path,vectors_path)
    pretrained_emnedding_list = set_pretrained_list(pretrained_dic,index_to_word)
    pretrained_tensor= torch.FloatTensor(pretrained_emnedding_list)
    embedder= nn.Embedding.from_pretrained(pretrained_tensor,freeze=False)
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

    # # Load model
    # model.load_state_dict(torch.load(model_save_path))
    #
    # # Evaluation pass.
    # _, test_acc = run_single_epoch(test_sequences, model,optimizer,optimizer,BATCH_SIZE,CONTEXT_SIZE,word_to_index,label_to_index,index_to_label, False,POS)
    # print("Test Accuracy: {:.3f}".format(test_acc))

if __name__=='__main__':
    main()


