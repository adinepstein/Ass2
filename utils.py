import sys

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

# def set_data_and_indexs(filename):
#
#     data= read_data(filename)
#     word_to_index= {PAD:0,UNKOWN:1}
#     index_to_word= [PAD, UNKOWN]
#     label_to_index = {PAD:0}
#     index_to_label = [PAD]
#     word_to_label = {}
#
#     for word , label in data:
#         word_to_label[word]=label
#         if word not in word_to_index:
#             word_to_index[word]=len(word_to_index)
#             index_to_word.append(word)
#         if label not in label_to_index:
#             label_to_index[label]= len(label_to_index)
#             index_to_label.append(label)
#
#     return word_to_label, word_to_index,index_to_word,label_to_index,index_to_label




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




