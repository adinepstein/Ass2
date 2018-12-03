Aviad Fux 302593421 and Adin Epstein 021890876

Our model runs on Pytorch using windows

Part 1,3,4 run the same way (all 3 use the same model and utils with a different embedder and small other changes).

There are 3 py files you need for running a training or for predicting using a trained model.

main file: tagger1/2/3.py
addition files: 1- windowTaggerModel.py - extends the pytorch Module and
                2- utils.py

In each file that you want to run you must set the following parameters:

1) POS= (choose the task type - POS-True or NER-False)

2) PRETRAINED=  (choose embedding vectors type, pretranined- True or untrained- False) - only in tagger 4

3) TRAIN_MODEL= (choose if you want to train the model-True, or load the trained model and predict-False)

4) Insert the file paths of the train,dev and test data
    train_path="train.txt"
    dev_path="dev.txt"
    test_path ="test.txt"

5) Insert the file paths of the pretrained embedding files
    word_path="vocab.txt"
    vectors_path="wordVectors.txt"

6) Insert the path where to save the parameters of the module
    save_model_path= "model1_ner.pt"

7) Insert the file path of where to save the prediction results
    prediction_results_path="test4.pos"

8) Insert the parameters of the model (learning rate, epochs, dim layer,batch size)

9) Very important - For running a prediction you must insert the DIM_LAYER of the model you want to use for production

