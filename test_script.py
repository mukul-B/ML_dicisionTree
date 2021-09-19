import numpy as np
import decision_trees2 as dt
import nearest_neighbors as nn
import clustering as kmeans

def load_data(fname):
  f = open(fname, 'r')
  ctr = 0
  y_str = ''
  for line in f:
    line = line.strip().split(';')
    if ctr == 0:
      x_str = line
    else:
      y_str = line
    ctr+=1
  f.close()
  X = []
  Y = []
  for item in x_str:
    temp = [float(x) for x in item.split(',')]
    X.append(temp)
  if len(y_str)>0:
    for item in y_str:
      temp = int(item)
      Y.append(temp)
  X = np.array(X)
  Y = np.array(Y)
  return X, Y


X,Y = load_data("data_3.txt")
print(X,Y)
max_depth = 3

DT = dt.DT_train_binary(X,Y,max_depth)

test_acc = dt.DT_test_binary(X,Y,DT)
print("DT:",test_acc)
#
# X,Y = load_data("data_4.txt")
# best_k=nn.choose_K(X,Y,X,Y)
# print("KNN:", best_k)
# acc = nn.KNN_test(X,Y,X,Y,1)
#print("KNN:", acc)


# X,Y = load_data("data_5.txt")
# kmeans.K_Means_better(X,2)
# mu = np.array([[1],[5]])
# mu = kmeans.K_Means(X,2,mu)
# print("KMeans:",mu)

# X,Y = load_data("data_6.txt")
# mu = kmeans.K_Means(X,2,[])
# print("KMeans:",mu)
