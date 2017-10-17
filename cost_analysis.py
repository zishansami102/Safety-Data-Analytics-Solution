import numpy as np
from sklearn import preprocessing,cross_validation, svm
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sn

plt.style.use('seaborn-darkgrid')
plt.rcParams['axes.facecolor'] = '#f7f1eb'

df = pd.read_csv('newccd.csv')
df.drop(['Id'], 1, inplace=True)

# df = df[['', '', '', '']]
X = df.drop(['churn'],1)
y = list(df.churn)



## Model Deciding through highest AUC(Area Under ROC curve)
print "Churn percentage in full data:",np.sum(1*(np.array(y)==1))/5000.0

## Train Test Split
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.4)
X_val, X_test, y_val, y_test = cross_validation.train_test_split(X_test,y_test,test_size=0.5)
print "Churn percentage in training data:",np.sum(1*(np.array(y_train)==1))/float(len(y_train))
print "Churn percentage in val data:",np.sum(1*(np.array(y_val)==1))/float(len(y_val))
print "Churn percentage in test data:",np.sum(1*(np.array(y_test)==1))/float(len(y_test))



## Model Selection
print "\nNeural Network Analysis:"
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,), random_state=1,activation='relu',max_iter=300)
## Training
clf.fit(X_train, y_train)
prob = clf.predict_proba(X_val)[:,1]
fpr5, tpr5, thresholds = metrics.roc_curve(np.array(y_val), prob, pos_label=1,drop_intermediate=False)
AUC5 = metrics.auc(fpr5, tpr5)
print "AUC : ", AUC5,"\n"
## Confusion Matrix at initial Threshold=0.5
pred = (prob>=0.5)*1
tn, fp, fn, tp = metrics.confusion_matrix(np.array(y_val), pred).ravel()
conf_matrix = np.array([[tp,fp],[fn,tn]])

sensitivity = tp/float(fn+tp)
specificity = tn/float(fp+tn)
precision = tp/float(tp+fp)
df_cm = pd.DataFrame(conf_matrix,  index=["1","0"],columns=["1","0"])
print "Confusion Matrix at default Threshold=0.5"
print df_cm
print "Specificity:", np.round(specificity,2),"| Sensitivity/Recall:", np.round(sensitivity,2),"| Precision:",np.round(precision,2),"\n"
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True,annot_kws={"size": 16},fmt='g')
plt.title("Confusion Matrix with Neural Network (threshold="+str(0.5)+")")
plt.xlabel("Correct Labels")
plt.ylabel("Predicted Labels")
plt.show()


## Choosing the suitable Threshold Value at point with lowest cost value
print "Threshold selection:"
cost_list = []	## list initialization
tn, fp, fn, tp = metrics.confusion_matrix(np.array(y_val), (prob>=0.01)*1).ravel()
ratio = 6	## ratio = (Cost of aquisition per person)/(Cost of retention per person)
lowest_cost = 10*(tp+fp)+ratio*10*fn
x = np.arange(0.0,1.0,0.01) ## Threshold stretch
for threshold in x:
	pred = (prob>=threshold)*1
	tn, fp, fn, tp = metrics.confusion_matrix(np.array(y_val), pred).ravel()
	cost = 10*(tp+fp)+ratio*10*fn
	if cost<=lowest_cost:
		lowest_cost=cost
		selected_threshold=threshold
	cost_list.append(cost)

plt.plot(x, cost_list,color='#62bba4',label="Total Cost",linewidth=3)
plt.legend(loc='lower right')
plt.title("Total Cost vs Threshold")
plt.xlabel("---- Threshold  --->")
plt.ylabel("--- Total Cost-->")
plt.show()

## 	Confusion Matrix at optimum threshold value
pred = (prob>=selected_threshold)*1
tn, fp, fn, tp = metrics.confusion_matrix(np.array(y_val), pred).ravel()
conf_matrix = np.array([[tp,fp],[fn,tn]])
df_cm = pd.DataFrame(conf_matrix, index=["1","0"],columns=["1","0"])
print "Confusion Matrix at selected threshold=", selected_threshold
print df_cm

sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True,annot_kws={"size": 16},fmt='g')
plt.xlabel("Correct Labels")
plt.ylabel("Predicted Labels")
plt.title("Confusion Matrix at selected threshold("+str(selected_threshold)+")")
plt.show()


sensitivity = tp/float(fn+tp)
specificity = tn/float(fp+tn)
precision = tp/float(tp+fp)
acc = (tp+tn)/float(tp+tn+fn+fp)

prob = clf.predict_proba(X_train)[:,1]
fpr, tpr, thresholds = metrics.roc_curve(np.array(y_train), prob, pos_label=1)
Training_AUC = metrics.auc(fpr, tpr)

print "\nSpecificity:", np.round(specificity,2),"| Sensitivity/Recall:", np.round(sensitivity,2),"| Precision:",np.round(precision,2)
print "val AUC:",AUC5,"| Training AUC:",Training_AUC,"| val. Accuracy:",acc


print "\nTest set Analysis:"
prob = clf.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = metrics.roc_curve(np.array(y_test), prob, pos_label=1)
Val_AUC = metrics.auc(fpr, tpr)

pred = (prob>=selected_threshold)*1
tn, fp, fn, tp = metrics.confusion_matrix(np.array(y_test), pred).ravel()
conf_matrix = np.array([[tp,fp],[fn,tn]])
df_cm = pd.DataFrame(conf_matrix, index=["1","0"],columns=["1","0"])
print "Confusion Matrix of Test Set:"
print df_cm

sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True,annot_kws={"size": 16},fmt='g')
plt.xlabel("Correct Labels")
plt.ylabel("Predicted Labels")
plt.title("Confusion Matrix of Test Set")
plt.show()

sensitivity = tp/float(fn+tp)
specificity = tn/float(fp+tn)
precision = tp/float(tp+fp)
acc = (tp+tn)/float(tp+tn+fn+fp)
print "\nSpecificity:", np.round(specificity,2),"| Sensitivity/Recall:", np.round(sensitivity,2),"| Precision:",np.round(precision,2)
print "Test AUC:",Val_AUC,"| Test Accuracy:",acc

