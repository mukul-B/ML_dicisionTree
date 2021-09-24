Data structure for Decision Tree
Tree is created by defining class Tree with two features name and children
The name variable hold values of type Node( label, feature and theta) and children is list of other trees.
——————
DT_train_binary(X,Y,max_depth)
 1). Calls function -> best_features(X,Y) : to get best feature that have maximum information gain, when portion the samples.
 2)
DT_test_binary(X,Y,DT)
DT_make_prediction(x,DT)
DT_train_real(X,Y,max_depth)
DT_test_real(X,Y,DT)


———————————————————————————————————————————————————————————————————————————————————————————————————————————————————

nearest_neighbors.py

KNN_test(X_train,Y_train,X_test,Y_test,K)

choose_K(X_train,Y_train,X_val,Y_val)

———————————————————————————————————————————————————————————————————————————————————————————————————————————————————


Clustering.py
K_Means(X,K,mu)

K_Means_better(X,K)
