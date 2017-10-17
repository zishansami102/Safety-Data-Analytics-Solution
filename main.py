import numpy as np
from sklearn import preprocessing,cross_validation, svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sn

plt.style.use('seaborn-darkgrid')
plt.rcParams['axes.facecolor'] = '#f7f1eb'

df = pd.read_csv('newccd.csv')
df.drop(['Id','account_length'], 1, inplace=True)

# df = df[['', '', '', '']]
X = df.drop(['churn'],1)
y = list(df.churn)



## Model Deciding through highest AUC(Area Under ROC curve)
print "Churn percentage in full data:",np.sum(1*(np.array(y)==1))/5000.0

## Train Test Split
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.3)
X_val, X_test, y_val, y_test = cross_validation.train_test_split(X_test,y_test,test_size=0.5)
print "Churn percentage in training data:",np.sum(1*(np.array(y_train)==1))/float(len(y_train))
print "Churn percentage in val data:",np.sum(1*(np.array(y_val)==1))/float(len(y_val))
print "Churn percentage in test data:",np.sum(1*(np.array(y_test)==1))/float(len(y_test))



## Model Selection
print "\nLogistic Regression Analysis:"
clf = LogisticRegression(solver='sag',max_iter=500)
## Training
clf.fit(X_train, y_train)
prob = clf.predict_proba(X_val)[:,1]
fpr1, tpr1, thresholds = metrics.roc_curve(np.array(y_val), prob, pos_label=1)
AUC1 = metrics.auc(fpr1, tpr1)
print "AUC: ", AUC1
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
plt.title("Confusion Matrix with Log Reg (threshold="+str(0.5)+")")
plt.xlabel("Correct Labels")
plt.ylabel("Predicted Labels")
plt.show()


## Model Selection
print "\nGaussian Naive Bayes Analysis:"
clf = GaussianNB()
## Training
clf.fit(X_train, y_train)
prob = clf.predict_proba(X_val)[:,1]
fpr2, tpr2, thresholds = metrics.roc_curve(np.array(y_val), prob, pos_label=1)
AUC2 = metrics.auc(fpr2, tpr2)
print "AUC : ", AUC2
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
plt.title("Confusion Matrix with GaussianNB (threshold="+str(0.5)+")")
plt.xlabel("Correct Labels")
plt.ylabel("Predicted Labels")
plt.show()

## Model Selection
print "\nSVM Analysis:"
clf = svm.SVC(probability=True,kernel='rbf')
## Training
clf.fit(X_train, y_train)
prob = clf.predict_proba(X_val)[:,1]
fpr3, tpr3, thresholds = metrics.roc_curve(np.array(y_val), prob, pos_label=1)
AUC3 = metrics.auc(fpr3, tpr3)
print "AUC : ", AUC3
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
plt.title("Confusion Matrix with SVM (threshold="+str(0.5)+")")
plt.xlabel("Correct Labels")
plt.ylabel("Predicted Labels")
plt.show()


## Model Selection
print "\nGradient Boosting Analysis:"
clf = GradientBoostingClassifier()
## Training
clf.fit(X_train, y_train)
prob = clf.predict_proba(X_val)[:,1]
fpr4, tpr4, thresholds = metrics.roc_curve(np.array(y_val), prob, pos_label=1)
AUC4 = metrics.auc(fpr4, tpr4)
print "AUC : ", AUC4
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
plt.title("Confusion Matrix with Gradient Boosting (threshold="+str(0.5)+")")
plt.xlabel("Correct Labels")
plt.ylabel("Predicted Labels")
plt.show()


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



## plotting ROC curve for all the models
x = np.arange(0.0,1.0,0.01)
plt.plot(x,x,'--',color="#00a2d9",linewidth=1)
plt.plot(fpr1,tpr1,label="Logistic Regression(Area:"+str(round(AUC1,2))+")", color="cyan",linewidth=1)
plt.plot(fpr2,tpr2,label="Gaussian Naive Bayes(Area:"+str(round(AUC2,2))+")", color="black",linewidth=1)
plt.plot(fpr3,tpr3,label="SVM RBF(Area:"+str(round(AUC3,2))+")", color="red",linewidth=1)
plt.plot(fpr4,tpr4,label="Gradient Boosting(Area:"+str(round(AUC4,2))+")", color="#ffb74d",linewidth=1)
plt.plot(fpr5,tpr5,label="Neural Nework(Area:"+str(round(AUC5,2))+")", color="#62bba4",linewidth=1)

plt.legend(loc='lower right')
plt.title("ROC Curve for different classifiers")
plt.xlabel("---- False Positive Rate  --->")
plt.ylabel("---- True Positive Rate --->")
plt.show()


## Choosing the suitable Threshold Value at point with lowest distance from the upper left corner on the ROC curve
x = np.arange(0.0,1.0,0.01)
print "Choosing threshold at point closer to upper left corner in ROC curve:"
YIndexList = []
dList = []
fb = []
Highest_YIndex=0 # Initializing it to zero
Lowest_d = 1
highest_Fbeta=0
for threshold in x:
	pred = (prob>=threshold)*1
	tn, fp, fn, tp = metrics.confusion_matrix(np.array(y_val), pred).ravel()
	sensitivity = tp/float(fn+tp)
	specificity = tn/float(fp+tn)
	d = np.sqrt(np.power(1-sensitivity,2)+np.power(1-specificity,2))
	YIndex=specificity+sensitivity-1
	fbeta = metrics.fbeta_score(y_val, pred,beta=2)
	if fbeta>=highest_Fbeta:
		highest_Fbeta=fbeta
		selected_threshold=threshold
	YIndexList.append(YIndex)
	dList.append(d)
	fb.append(fbeta)

plt.plot(x,YIndexList,color='#62bba4',label="Youden Index",linewidth=3)
plt.plot(x,dList,color='#ffb74d',label="Dist from upper Left Corner",linewidth=3)	
plt.plot(x, fb,color='#816E94',label="Fbeta",linewidth=3)
plt.legend(loc='lower right')
plt.title("Different Parameters vs Threshold")
plt.xlabel("---- Threshold  --->")
plt.ylabel("--- Index -->")
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

