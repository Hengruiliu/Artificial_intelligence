import numpy as np
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
#import data
wine = load_wine()

#Select the first two attributes for a 2D representation of the image.
X = wine.data[:,:2]
y = wine.target

#Randomly split data into train, validation and test sets in proportion 5:2:3
from sklearn.model_selection import train_test_split
seed=4 #for obtained the same result
X_train, X_rest, y_train, y_rest = train_test_split(X,y,test_size=0.5,random_state=seed)
X_val,X_test,y_val,y_test= train_test_split(X_rest,y_rest,test_size=0.6,random_state=seed)

# plot the decision boundry (data,method)
def plot_decision_regions(X, y, clf):
    # define the limit of polt
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    # define the boundry of different target
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots()
    colors = ['r', 'g', 'b']
    ax.contourf(xx, yy, Z, alpha=0.4, levels=[-0.5, 0.5, 1.5, 2.5], colors=colors)
    #classify the differnet target
    for lbl in range(max(y) + 1):
        mask = y == lbl
        ax.scatter(X[mask][:, 0], X[mask][:, 1], s=20, edgecolor="k", color=colors[lbl], label=f"{lbl}")
        pass
    ax.legend()
    plt.show()
    pass

# Apply K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
val_accuracy = []
best_k = 0
max_val_acc = 0
for k in [1, 3, 5, 7]:
    # Apply K-Nearest Neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    # Evaluate the method on the validation set
    y_pred = knn.predict(X_val)
    accuracy = sum(y_pred == y_val) / len(y_val)
    val_accuracy.append(accuracy)
    if accuracy > max_val_acc:
        max_val_acc = accuracy
        best_k = k
        pass
    print(f"k-NN with k={k}. Accuracy on val: {accuracy * 100} %")
    # Plot the data and the decision boundaries
    plot_decision_regions(X_train, y_train, clf=knn)
    pass

#Plot a graph showing how the accuracy on the validation set varies when changing K
plt.figure()
plt.xlabel('K')
plt.ylabel('val accuracy')
plt.plot([1,3,5,7], val_accuracy)
plt.title('Accuracy trend with respect to K')
plt.show()
# best result
print('best k:',best_k)