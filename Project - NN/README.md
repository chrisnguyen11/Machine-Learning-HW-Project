## Files
*herg_features.tsv* - dataset from PKB lab 

*Neural_Net.py* - script that builds, test, and evaluates the neural net models using the *herg_features.tsv* dataset

*neural_net_output.txt* - output file from *Neural_Net.py* script with each models' accuracy, F1 score, percision, recall


## Prediction of point mutation for long QT syndrome using Machine Learning
By Geraldine San Ramon (SVM) and Chris Nguyen (NN)

### Introduction
Long QT syndrome (LQTS) is a cardiac condition characterized by a prolonged QT-interval. In other words, LQTS is a cardiovascular disorder and one risk of type 2 LQTS is sudden death. The KCNH2 gene encodes for potassium channels, however, mutations in this gene can cause type 2 LQTS. Only a small number of mutations in the KCNH2 gene are pathogenic. 
	
### Dataset description
These mutations are either classified as benign - do not cause disease - or are pathogenic - causing disease. The data set contains 38 mutations classified as trafficking and 166 mutations that are non-trafficking. There are two key features that are important to evaluating whether the mutation causes disease, hydrophobicity change, and conversation score. The conservation score is related to the sequence conservation, in other words, it characterizes the functionally and structurally residues of the mutation to the non-mutation sequence. This is critical for proteins as different amino acids have different properties which may be the source of the functional differences of these mutations. The hydrophobicity change refers to a change in free energy for the transfer of an amino acid from a solvent. These features were used as they have been found to have significant functional effects to classify mutations as benign or pathogenic. 

### Neural Network
For the neural network models, the data set contains 2 features, so the input layer needs to contain 2 units for each feature. These units utilize the rectified linear unit activation function which is a non-linear function and compresses all inputs less than 0 to 0; while for inputs greater than 0, the function is the identity function. When compared to other activations, such as logistic sigmoid or inverse trigonometric functions, this activation function does not have a maximum and near 0, the function does not drastically change. This allows for better gradient propagation, ease of computation, and sparse activation of units. The output of the neural network is the classification of the mutation so if the mutation causes pathogenesis. Therefore, the output layer has one unit to classify the mutation into causing pathogenesis or does not cause pathogenesis. The activation function for this single classifier is a logistic sigmoid to compress the input into the space between 0 and 1 for a single classification. When compiling this neutral network framework, the optimizer chosen was rmsprop ??? which relies on gradient descent to perform the optimization of the loss function. The lost function used when compiling the model is ???binary cross-entropy which calculates the cross-entropy loss between the true and predicted labels. To attempt to make a more accurate model, we performed a grid search, varying the number of epochs, batch sizes, and learning rate. Using grid search the optimal hyperparameters were 2 epochs, 1 batch size, and a learning rate of 0.001. 

### Evaluation/Discussion
Compared to the dummy classifier accuracy as baseline, the accuracy on the test data  using LinearSVM is 0.82 which is the same as the dummy accuracy for most frequent. Also, the precision, recall & f1 score for 0 class is 0.00 which means that it???s not classifying the 0 label correctly if not at all. The SVC gridsearch for different kernels gave an output of 0.66 accuracy which is the same as the dummy for stratified. The precision, recall & f1 score for 0 class is still too low.  
When testing the baseline network model in the test and entire dataset, the statistics were accuracy: 0.7096, f1: 0.8301, precision: 0.7096, recall: 1.0. A recall of 1 and the accuracy indicates that the model was not able to distinguish between the two classes. Similarly, when looking at the tuned neural network model using grid search, this model was also unable to classify the mutations, the statistics were accuracy: 0.8137, f1: 0.8972, precision: 0.8137, recall: 1.0. 

## [Presentation](https://docs.google.com/presentation/d/1b7Wr27tGzMrxmreThVKl4bLdyqjhh2YwEwrAf-lig-4/edit?usp=sharing)
