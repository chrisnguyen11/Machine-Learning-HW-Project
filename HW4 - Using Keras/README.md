## Files 
*heart.csv* - the dataset used for this assignment 

*hw_v2.py* - the python script used for the assignment, outputs the results written in the docx file

## Written Report 

The data set I chose for this homework set is the heart-failure-prediction from Kaggle, this is the same dataset I used from the last homework. If we recall, this data set contains 11 features and the label: heart failure. The features were converted from strings to integers, for example, the feature chest pain type has multiple classes: TA (Typical Angina), ATA (Atypical Angina), NAP (Non-Anginal Pain), and ASY (Asymptomatic) which were encoded into integers using the apply and map functions.
 
1. Randomly Splitting the Data into Training and Test
To split the data set, I used the sklearn function, train_test_split which outputs two data sets, the training and testing. Using the ratios outlined in the homework directions, the training data set contains 80% while the test data set contains 20% of the original dataset.
 
2. n-fold Cross Validation
To implement the n-fold cross validation, I began by creating a new function nfold_cross_valid which takes in the training features and class, the number of folds – k, and two hyperparameters c and p. The numpy function array_split was used to split the training features and class into k folds. With k-groups we iterate over the folds and concatenate the k-1 folds into that fold’s ‘training data set’. Then using this fold’s ‘training data set’ we train a logistic regression model. The accuracy of the iterations is appended into the list. This process is performed using each fold as the testing data set. The average of all the iterations is taken a which is the output of the nfold_cross_valid function. When using the heart dataset, the average accuracy over all the iterations was 85.28%.  
 
3. Grid Search
To implement the grid search, I created a new function grid_search which takes in the training’s features and class, and the number of folds – k. The logistic classifier has two primary hyperparameters: c and p. The first hyperparameter, c, refers to the inverse of the regularization strength. The second hyperparameter, p, is the mixing parameter unique to elastic net as it is the ratio at which the penalties L1 and L2 are combined. A list of values for c is explicitly written out the inverse of the regularization strength ranges from 0 to infinity, so I chose10 values:  [0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000]. Similarly, a list of values for p was explicitly written as it ranges from 0 to 1. I chose 11 values: [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]. Then for each value for c and p, we call the function from part 2, nfold_cross_valid to demine the accuracy of using the specific combination of c and p values. Then the accuracy of the combinations of c and p are used to determine the most accurate model with the range of hyperparameters. For the heart dataset, the most accurate model had a c value of 0.1 and a p value of 0.4. Interestingly, the accuracy of this model was 85.42% which is greater than the accuracy from part 2.  
 
4. Evaluating the Performance of the Best Model in Part 3
Using the hyperparameters from part 3, a c value of 0.1 and a p value of 0.4, a model was trained on the entire testing data set and tested in the testing data set. The accuracy of this model was 84.23%.
