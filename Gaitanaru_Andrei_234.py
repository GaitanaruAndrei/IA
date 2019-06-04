import numpy as np
import pandas as pd
from collections import Counter
import glob
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
import os

import io
startpath = "C:/Users/GaitanaruAndrei/Desktop/Data/Data/"


def citire_train():
    nr = 0
    data_nou = 0
    lista = []  # lsita in care salvam datele din csv sub forma de matrici ( o sa fie de dimensiuni 9000 x 450
    # pentru a se potrivii cu  dimenisunea listei ce va contine id-urile
    for root, dirs, files in os.walk(startpath + "train/"):
        for x in files:  # parcurgem fiecare csv si accesam numele acestora
            ok = 0
            nr = nr + 1
            data = np.array(pd.read_csv(startpath + "train/" + x, header=None))  # salvam fiecare csv intr o matrice
            if data.size > 450:
                data_nou = np.array(data[:150])

            if data.size < 450:  # verificam daca depaseste dimensiunea de 450 ( 150 linii si 3 coloane)
                while (
                        ok == 0):  # daca depaseste eliminam ultimele randuri in plus, daca are mai putin adaugam o linie formata din media valorilor din csv
                    medie = np.mean(data, axis=0)
                    data = np.concatenate((data, [medie]), axis=0)
                    if data.size == 450:
                        data_nou = (data[:])
                        ok = 1
            if data.size == 450:
                data_nou = (data[:])
            data_nou.flatten()  # transformam in vector ca sa putem adauga la lista
            lista.append(data_nou)
        #             if nr == 1:
        #                 train_label=np.array(data_nou[:])

        #             else:
        #                 train_label=np.append(train_label,data_nou,axis=0)
        return lista  # returnam lista
    # data_nou = data_nou.flatten()
    # train.append(data_nou)
    #  print(data.shape)
#         medie = [np.mean((data),axis=0)]
#         for i in range(data.shape[0], 150):

# #             mean_row = np.mean(data, axis = 0)
# #             data = np.concatenate((data, [mean_row]), axis = 0)


#            # data = data.flatten()
#             train.append(data)
def citire_test():  # facem acelasi lucru ca si pentru fisierele de tip train
    nr = 0
    data_nou = 0
    lista = []
    for root, dirs, files in os.walk(startpath + "test/"):
        for x in files:
            ok = 0
            nr = nr + 1
            data = np.array(pd.read_csv(startpath + "test/" + x, header=None))
            if data.size > 450:
                data_nou = np.array(data[:150])

            if data.size < 450:
                while (ok == 0):
                    medie = np.mean(data, axis=0)
                    data = np.concatenate((data, [medie]), axis=0)
                    if data.size == 450:
                        data_nou = (data[:])
                        ok = 1
            if data.size == 450:
                data_nou = (data[:])
            data_nou.flatten()
            lista.append(data_nou)
        #             if nr == 1:
        #                 train_label=np.array(data_nou[:])

        #             else:
        #                 train_label=np.append(train_label,data_nou,axis=0)
        return lista

lista_train = citire_train()
train = np.array(lista_train[:]) # dam reshape la lista deoarece este de 3d iar  noi avem nevoie de 2d
train=train.reshape(9000,450)

lista_test= citire_test()

test = np.array(lista_test[:]) # dam reshape la lista deoarece este de 3d iar  noi avem nevoie de 2d
test=test.reshape(5000,450)

train_labels = pd.read_csv('C:/Users/GaitanaruAndrei/Desktop/Data/Data/train_labels.csv', usecols=[1], header=0)
labels =np.asarray(train_labels).ravel() ## salvam intr o lista numai coloana id-urilor
clf = SVC(gamma='auto')
print(labels)


X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.20) # Am impartit datele
svm_classifier = SVC(C=20, kernel='rbf', gamma=0.001)
svm_classifier.fit(X_train, y_train)  # apelam functia de antrenare a datelor

y_pred = svm_classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))   # pemntr a vizualiza performanta
print(classification_report(y_test, y_pred))
svm_classifier= SVC(C=20, kernel='rbf', gamma=0.001)
svm_classifier.fit(train, labels)
y_final = svm_classifier.predict(test)
test_lista = []
for root, dirs, files in os.walk(startpath+"test/"):
        for x in files:
            s=x[0:-4]
            test_lista.append(s)
print(len(test_lista))
print(y_final)
with open("sumbm.csv", "w") as f:
    f.write('id,class\n')
    for i in range(len(test_lista)):
        buffer1 = test_lista[i]
        buffer2 = str(int(y_final[i]))
        f.write(buffer1)
        f.write(',')
        f.write(buffer2)
        f.write('\n')
