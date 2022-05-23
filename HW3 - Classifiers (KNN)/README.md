## Files 
*heart.csv* - conatins the data useed for the homework

*hw3.py* - is the executable python script which builds, test, and analyze the heart.csv data. prints to screen the accuracies of the models

## Written Report

The data set I chose for this homework set is the heart-failure-prediction from Kaggle. This data set contains 11 features and the label: heart failure. These features are age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting electrocardiogram, maximum heart rate, exercise-induced angina, oldpeak (an ECG finding reliable for the diagnosis of heart disease), and ST_Slope (the slope of peak exercise). Some of these features need to be converted from strings to integers, for example, the feature chest pain type has multiple classes: TA (Typical Angina), ATA (Atypical Angina), NAP (Non-Anginal Pain), and ASY (Asymptomatic) which were encoded into integers using the apply and map functions.
 
1. Default Logistic Regression Classifier
 
From sklearn, I used the logistic regression classifier function which takes in the two data frames: the features and labels of the training data set. With the default hyper-parameters, I tested this model on the development data set, the accuracy of the default model is 83.3%. When looking more at the statistics, the default model has an F1 score of 90.3%. F1 score is a better scoring metric for data sets that are imbalanced, however, this data is balanced as it contains about an equal number of samples with and without heart disease. Therefore, accuracy may be a better evaluation metric as it considers predicting samples of the negative class, samples without heart disease.
 
2. Tweaked Logistic Regression Classifier
 
Tweaking the default logistic regression classifier model, I tested multiple values of the hyper-parameter C, which is the inverse of regularization strength. A small value of C increases the amount of regularization, regularization reduces overfitting/underfitting by adding another term into the cost function equation. This term penalizes large weights by decaying proportional to their size. I tested values multiple values for C, between 0.00001 and 10000 to find the most accurate model in the development data set. With a C value of 0.005, the model is the most accurate with an accuracy of 83.3%, which is the same as in part one. The model built in part one has a C value of 1 so these two C values are comparable.
 
3. K-nearest Neighbor
 
To implement the k-nearest neighbors classifier, I modeled my code after the code from the previous homework. I started by defining a new class KNN, which will house the functions needed to implement k-nearest neighbors. Then I created the constructors, training_X and training_y, which will house the training data’s features and labels respectively. For this algorithm, the training data is used as a reference to determine the labels of new data. The fit function is used to get the training data into the class KNN which will be used by the other function, nearest_k_neighbors. This function loops through each unknown sample then calculates the difference between the features of training data and the unknown features. The difference for each feature is squared, summed together, then the square root is this value is taken. This is equivalent to the Euclidian distance between the unknown sample and the samples of the training data set. Sorting these distances, we can then find the nearest k neighbors to the unknown sample. The majority label of the k nearest training samples will determine the label of the unknown sample.
Using the training dataset, I use the fit function of KNN then tested this model on the development data set. Similar to how the hyper-parameter was determined for the logistic regression model, I also tested a series of odd integers to find the most accurate model on the devotement data set. Only odd integers were chosen because even neighbors can cause a tie. The most accurate KNN model has a k a value of 13 and an accuracy of 85.5%.  
 
4. Baseline Classifiers
 
From scikit-learn, I imported the function DummyClassifer and created two models. These models have different values for the hyper-parameter ‘strategy’, one using ‘stratified’ while the other used ‘most_frequent’. The first model using ‘stratified’ classifies samples following the distribution of labels in the training set. While the ‘most_frequent’ model classifies all samples as the most frequent label in the training set.
When testing the performance of these two models on the development data set, the accuracies are 53.6% and 48.6% respectively.
 
5. Testing using the Test Data Set  
 
When evaluating all the models built on the testing data set, the first model created in the first part of this assignment was the default logistic regression model which has an accuracy of 85.5 %. The second model built in the second part of this assignment was the tweaked logistic regression model which has an accuracy of 84.1 %. The third model built was the k-nearest- neighbors classifier which has an accuracy of 87.0%. From part 4, the dummy classifiers ‘stratified’ and ‘most_frequent’ were 66.1% and 63.8%. The most accurate model on the test data set was the k-nearest- neighbors classifier.
