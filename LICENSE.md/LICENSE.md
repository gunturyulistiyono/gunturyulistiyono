import matplotlib.pyplot as plt
from subprocess import call
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics, tree
import pandas as pd

#load dataset ke dalam Pandas Dataframe
dat = pd.read_csv('tic_tac_toe.csv')
cols = ['top-left-square', 'top-middle-square', 'top-right-square', 'middle-left-square', 'middle-middle-square', 'middle-right-square', 'bottom-left-square', 'bottom-middle-square', 'bottom-right-square']
x = dat[cols]
y = dat['Class']

#Pisah, Test_size 20% dan training set 80%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
dtree = DecisionTreeClassifier()
dtree = dtree.fit(x_train, y_train)

#Melakukan prediksi terhadap data testing dan probabilitasnya
y_predict = dtree.predict(x_test)
dtree.predict_proba(x_test)

#tampilkan akurasi klasifikasi
print("Akurasi klasifikasi Machine Learning : " + str(metrics.accuracy_score(y_test, y_predict))+"\n")

#tampilkan confusion matrix
print(metrics.confusion_matrix(y_test, y_predict))

#tampilkan precision,recall,fmeasure
print(metrics.classification_report(y_test, y_predict))

#Export as dot file
export_graphviz(dtree, out_file='tree.dot',
                feature_names=['top-left-square', 'top-middle-square', 'top-right-square', 'middle-left-square', 'middle-middle-square',
                               'middle-right-square', 'bottom-left-square', 'bottom-middle-square', 'bottom-right-square'],
                class_names='Class',
                rounded=True, proportion=False,
                precision=2, filled=True)
