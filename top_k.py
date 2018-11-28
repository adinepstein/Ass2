import numpy as np
import math
import scipy.spatial.distance as distance

word_to_index = {}
index_to_word = []
global vecs

def read_words_from_file(fileName):
    with open(fileName) as dataSource:
        i = 0
        for line in dataSource.readlines():
            word = line.rstrip()
            word_to_index[word] = i
            index_to_word.append(word)
            i += 1

def most_similar(word, k):
    k += 1
    indexWord = word_to_index[word]
    vectorWord = vecs[indexWord]

    distsArray = [math.inf] * k
    wordsArray = [""] * k

    for i in range(0, len(word_to_index)):
        dist = distance.cosine(vectorWord, vecs[i])

        indexInsert = k
        for j in range(k - 1, -1, -1):
            if dist >= distsArray[j]:
                indexInsert = j + 1
                break
            if j == 0: indexInsert = 0

        if indexInsert < k:
            distsArray.insert(indexInsert, dist)
            del distsArray[-1]

            wordsArray.insert(indexInsert, index_to_word[i])
            del wordsArray[-1]

    del distsArray[0]
    del wordsArray[0]
    return wordsArray, distsArray

def write_to_file(file, wordsArray, distsArray, word):
    file.write("the 5 most similar words for word - " + str(word) + ":\n")
    for i in range(0, len(wordsArray)):
        file.write(str(i) + ": " + wordsArray[i] + ", " + str(distsArray[i]) + "\n")
    file.write("\n")

def main():

    read_words_from_file("words.txt")
    global vecs
    vecs = np.loadtxt("vectors_file_name.txt")

    k = 5
    dog_wordsArray, dog_distsArray = most_similar("dog", k)
    england_wordsArray, england_distsArray = most_similar("england", k)
    john_wordsArray, john_distsArray = most_similar("john", k)
    explode_wordsArray, explode_distsArray = most_similar("explode", k)
    office_wordsArray, office_distsArray = most_similar("office", k)

    file = open("part2.txt", "w")

    write_to_file(file, dog_wordsArray, dog_distsArray, "dog")
    write_to_file(file, england_wordsArray, england_distsArray, "england")
    write_to_file(file, john_wordsArray, john_distsArray, "john")
    write_to_file(file, explode_wordsArray, explode_distsArray, "explode")
    write_to_file(file, office_wordsArray, office_distsArray, "office")

    file.close()

if __name__ == '__main__':
    main()