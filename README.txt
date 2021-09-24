Data structure for Decision Tree
Tree is created by defining class Tree with two features name and children
The name variable hold values of type Node( label, feature and theta) and children is list of other trees.
——————
DT_train_binary(X,Y,max_depth)
 1). Calls function -> best_features(X,Y) : to get best feature that have maximum information gain, when portion the samples.
 2)  the label for this node is calculated by taking maximum label values.
 3) the tree is created by seeting the feature and label values
 4) for the Children nodes the label and feature are partitioned into two group based on feature value.
 5) the DT_train_binary(X,Y,max_depth-1) is called recursively for both children
 6) this top down process continue until we have max_depth =0 or when best_feature return -1 ( in case we have all same label for feature)

Function used :
    best_feature(X,Y)
    it runs through all feature and divide the sample in two part based on the feature value
    calculate information gain on each feature and return the maximum for the given X: samples

    hypertropy(Y)
    it finds count of each unique label values and return the sum of probability and log of probability

DT_test_binary(X,Y,DT)
    it loops through each  test sample in  X and make its prediction by calling DT_make_prediction(x,DT)
    checks i=the prediction with the given test labels and increment accuracy varible if it matches
    returns the ratio of accuracy and total samples

DT_make_prediction(x,DT)
    here the Tree DT is iterated from top to bottom.
    the next child is chosen depending on the feature value of test data.
    when the tree reaches feature -1 , the label of the tree is returned

DT_train_real(X,Y,max_depth)
    it is similar to binary problem except it find best theta , the value to which feature is compared to partiton the sample instead 0/1
    best_feature : its modified to include function best_theta for each feature.
                    it then set value of tree theta accordingly.
    best_theta(X,Y,feature): it loop through all sample and check all the values of X as theta ,
                   and find the theta for that feature , which result in maximum information gain

DT_test_real(X,Y,DT)
  it is similar to binary counterpart expect it compare values feature to theta from the tree.


———————————————————————————————————————————————————————————————————————————————————————————————————————————————————

nearest_neighbors.py

KNN_test(X_train,Y_train,X_test,Y_test,K)
for each sample in test data
    it calculates square distance from each training data
    it find the smallest K distance using np.argsort
        and take sum of training labels corresponding to k  smallest distance
    according to sum it assign the test data to a label
    Accuracy: if the predicted label is same as test label, it increments the accuracy variable
    finally return ratio of accuracy and total sample size

choose_K(X_train,Y_train,X_val,Y_val)
    in for loop get accuracy from KNN_test by giving differnt K values
    it also keep tract of maximum accuracy and its K value
    returns the best K

———————————————————————————————————————————————————————————————————————————————————————————————————————————————————


Clustering.py
K_Means(X,K,mu)
1) in case the mu is not given, it generate  'K' distinct samples from the given Samples X
2) initiate cluster set list of list , with length of number of means
3) for each sample it calculate distance from both the mu values and append in the perticular index of cluster_set
4) calculates new means by taking mean of each cluster_set
5) it returns new mean if convergences ie new mean is equal to previous mean , if not it repeats the process with new Mean in while loop

K_Means_better(X,K)

it run the K_means in For loop for 1000 times stores each result in list
it then calculate the maximum occurrence of means by taking unique values and count using np.unique on axis =0
and return values with maximum count.
