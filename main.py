import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

gnb = GaussianNB()

translatorX = {
    'x': 0,
    'o': 1,
    'b': 2
}

translatorY = {
    'positive': 1,
    'negative': -1
}

datasetTicTacToe = open("data/Tic_tac_toe.txt", "r")

X = []
y = []

test_val = 958

for i in range(0, test_val):
    line = datasetTicTacToe.readline()
    line = line.rstrip("\n")
    arr = line.split(",")
    currX = list(map(lambda x: translatorX[x], arr[:9]))
    currY = list(map(lambda y: translatorY[y], arr[9:]))
    X.append(currX)
    y.extend(currY)

x = []
steps = []
step = 0.1
for i in range(0, 10):
    globalX, testX, globalY, testY = train_test_split(X, y, train_size=step)

    gnb.fit(globalX, globalY)
    y_pred = gnb.predict(testX)

    res = 1 - accuracy_score(testY, y_pred)
    x.append(res)
    print(res)
    steps.append(step)
    step += 0.1

datasetTicTacToe.close()

plt.plot(steps, x)
plt.title("Погрешность в зависимости от соотнешения\nтестовой и обучающей выборок")
plt.show()




from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import csv
import numpy as np
import matplotlib.pyplot as plt

translatorSpam = {
    'spam': 0,
    'nonspam': 1
}

def getValuesReady(arr):
    coreX = []
    coreY = []

    for line in arr:
        tmp = line
        tmp = tmp.rstrip("\n")
        tmp = tmp.replace('"', "")
        arr = tmp.split(",")
        coreX.append(arr[1:-1])
        coreY.append(translatorSpam[arr[-1]])

    return (np.array(coreX).astype(np.float), coreY)

f = open("data/spam.csv", "r")
answ = f.readlines()
f.close()
answ = answ[1:]
gnb = GaussianNB()
X_data, y_data = getValuesReady(answ)

xArr = []
steps = []
step = 0.05
for i in range(0, 19):
    x, testX, y, testY = train_test_split(X_data, y_data, test_size=step)
    gnb.fit(x, y)
    y_pred = gnb.predict(testX)
    res = 1 - accuracy_score(testY, y_pred)
    print(res)
    step += 0.05
    xArr.append(res)
    steps.append(step)

plt.plot(steps, xArr)
plt.ylim(0.10, 0.25)
plt.title("Погрешность классификатора в зависимости от соотнешения\nтестовой и обучающей выборок")
plt.grid()
plt.show()



from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
import numpy as np
import random
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x1_1 = np.random.normal(10, 5, 80) # класс 1
    x1_2 = np.random.normal(16, 5, 80)

    x2_1 = np.random.normal(20, 4, 20) # класс -1 (мат ожидание, дисперсия)
    x2_2 = np.random.normal(4, 4, 20)
    ax = plt.subplots()
    plt.scatter(x1_1, x1_2, c='r')
    plt.scatter(x2_1, x2_2, c='b')
    plt.title("Генерация точек")
    plt.grid()
    plt.show()

    one_class = [1] * 80
    minus_one_class = [-1] * 20

    print(minus_one_class)
    train = []
    test = []
    expect = []
    for i in range(0, 10):
        train.append([x1_1[i], x1_2[i], one_class[i]])
        train.append([x2_1[i], x2_2[i], minus_one_class[i]])

    for i in range(11, 20):
        test.append([x1_1[i], x1_2[i], one_class[i]])
        test.append([x2_1[i], x2_2[i], minus_one_class[i]])

    for i in range(21, 80):
        test.append([x1_1[i], x1_2[i], one_class[i]])
        # test.append([x2_1[i], x2_2[i], minus_one_class[i]])

    random.shuffle(test)
    random.shuffle(train)

    train = np.array(train)
    test = np.array(test)
    expect = test[:, 2]

    gnb = GaussianNB()
    gnb.fit(train[:, :2], train[:,2])
    y_pred = gnb.predict(test[:, :-1])
    print(accuracy_score(expect, y_pred))
    print("***")
    c_matrix = confusion_matrix(expect, y_pred)
    print(c_matrix)
    TP = c_matrix[0][0]
    FP = c_matrix[0][1]
    FN = c_matrix[1][0]
    TN = c_matrix[1][1]
    print("***")
    fpr, tpr, thresholds = metrics.roc_curve(y_pred, expect)
    plt.plot(fpr,tpr)
    plt.title("ROC-кривая")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.grid()
    plt.show()

    precision, recall, thresholds = precision_recall_curve(y_pred, expect)
    plt.plot(recall, precision)
    plt.title("PR-кривая")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid()
    plt.show()




from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import csv
import numpy as np
import matplotlib.pyplot as plt

def readFromFile(fileName):
    f = open(fileName, "r")
    rr = csv.reader(f)
    answ = [row for row in rr]
    f.close()
    return answ

def getValuesReady(arr):
    coreX = []
    coreY = []

    for line in arr:
        coreX.append(line[1:-1])
        coreY.append(line[-1])

    return (np.array(coreX).astype(np.float), np.array(coreY).astype(np.int))

f = open("data/glass.csv", "r")
answ = f.readlines()
f.close()
answ = answ[1:]

for i in range(0, len(answ)):
    line = answ[i]
    line = line.rstrip("\n")
    line = line.replace('"', '')
    arr = line.split(",")
    answ[i] = arr

dataX, dataY = getValuesReady(answ)

x, testX, y, testY = train_test_split(dataX, dataY, train_size=0.8)

classes = [0] * 7

xArr = []
for neighbors in range(1, 25):
    knc = KNeighborsClassifier(n_neighbors=neighbors)
    knc.fit(x, y)
    y_pred = knc.predict(testX)
    res = 1 - accuracy_score(testY, y_pred)
    print(res)
    xArr.append(res)
    classes[
        knc.predict(
            [[1.51,11.7,1.01,1.19 ,72.59 ,0.43, 11.44, 0.02 ,0.1]]
        )[0] - 1] += 1

for metric in ['minkowski','chebyshev','euclidean', 'manhattan']:
    knc = KNeighborsClassifier(metric=metric)
    knc.fit(x, y)
    y_pred = knc.predict(testX)
    print(metric,  1 - accuracy_score(testY, y_pred))
    classes[knc.predict([[1.51,11.7,1.01,1.19 ,72.59 ,0.43, 11.44, 0.02 ,0.1]])[0] - 1] += 1

print(classes)


plt.plot(xArr)
plt.ylim(0.2, 0.5)
plt.title("Ошибка в зависимости от количества соседей")
plt.grid()
plt.show()





from sklearn import svm
from random import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
# BEGIN A
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

f = open("data/svmdata_a_test.txt", "r")
a_test_mass = f.readlines()
f.close()
a_test_mass = a_test_mass[1:]
shuffle(a_test_mass)

v = open("data/svmdata_a.txt", "r")
a_mass = v.readlines()
v.close()
a_mass = a_mass[1:]
shuffle(a_mass)

translator = {
    "red" : 0,
    "green" : 1
}

test_val = len(a_mass)
# test_val = 3
globalX = []
globalY = []

for i in range(0, test_val):
    line = a_mass[i]
    line = line.rstrip("\n")
    arr = line.split("\t")
    currX = arr[1:3]
    currY = translator[arr[3]]
    globalX.append(currX)
    globalY.append(currY)

testX = []
testY = []
for i in range(0, len(a_test_mass)):
    line = a_test_mass[i]
    line = line.rstrip("\n")
    arr = line.split("\t")
    currX = arr[1:3]
    currY = translator[arr[3]]
    testX.append(currX)
    testY.append(currY)

globalX = np.array(globalX)
globalY = np.array(globalY)

globalX = globalX.astype(np.float)
globalY = globalY.astype(np.int)

clf1 = svm.SVC(kernel='linear')
clf1.fit(globalX, globalY)

clf_predictions = clf1.predict(testX)
print("*TEST*")
print(accuracy_score(testY, clf_predictions))
print("***")
c_matrix = confusion_matrix(testY, clf_predictions)
print(c_matrix)

clf_predictions = clf1.predict(globalX)
print("*TRAIN*")
print(accuracy_score(globalY, clf_predictions))
print("***")
c_matrix = confusion_matrix(globalY, clf_predictions)
print(c_matrix)
print("Support vec " , clf1.n_support_)

X = globalX
y = globalY

C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C, max_iter=10000),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))
models = (clf.fit(X, y) for clf in models)

# title for the plots
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()


