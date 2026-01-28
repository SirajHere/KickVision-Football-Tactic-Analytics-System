import numpy as np
import math
from sklearn.ensemble import VotingClassifier
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import  accuracy_score,precision_score,recall_score,f1_score,confusion_matrix 
test=pd.read_csv('/Users/shubh/Downloads/testing.csv')
X_test=test.drop('prognosis', axis=1)
print(X_test)
y_test=test['prognosis']
print(y_test)
with open('random_forest_model1.pkl', 'rb') as f:
        ge= pickle.load(f) 
with open('multinomial_nb_model1.pkl', 'rb') as f:
        mnb= pickle.load(f) 
with open('svm_model1.pkl', 'rb') as f:
        final_svm_model= pickle.load(f) 
with open('logistic_regression_model1.pkl', 'rb') as f:
        lrrs= pickle.load(f) 
with open('voting_classifier_model1.pkl', 'rb') as f:
        voting_classifier= pickle.load(f) 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Encode the string labels in y_test into integer labels
label_encoder = LabelEncoder()
y_test_encoded = label_encoder.fit_transform(y_test)

# Assuming you have a trained model named 'model'
# and test data X_test and y_test_encoded

# Initialize a list to store the fpr and tpr for each class
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Initialize empty lists to store fpr, tpr, and auc for each class
all_fpr = []
all_tpr = []
all_auc = []
num_classes = 41  # Assuming num_classes is 41

# Iterate over each class
for i in range(num_classes):
    # Treat the current class as positive, and all others as negative
    y_true = (y_test_encoded == i)
    
    # Predict probabilities for the positive class
    y_scores = ge.predict_proba(X_test.values)[:, i]
    
    # Compute ROC curve and AUC for the current class
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Store fpr, tpr, and auc for plotting
    all_fpr.append(fpr)
    all_tpr.append(tpr)
    all_auc.append(roc_auc)

# Plot ROC curves for all classes
plt.figure(figsize=(12, 8))

# Plot ROC curves for each class
for i in range(num_classes):
    plt.plot(all_fpr[i], all_tpr[i], label='Class {} (AUC = {:.2f})'.format(i, all_auc[i]))

plt.plot([0, 1], [0, 1], linestyle='--', color='k', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Class randomforest')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='x-small')
plt.xticks(fontsize='small')
plt.yticks(fontsize='small')
plt.show()


# Initialize a list to store the fpr and tpr for each class
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Initialize empty lists to store fpr, tpr, and auc for each class
all_fpr = []
all_tpr = []
all_auc = []
num_classes = 41  # Assuming num_classes is 41

# Iterate over each class
for i in range(num_classes):
    # Treat the current class as positive, and all others as negative
    y_true = (y_test_encoded == i)
    
    # Predict probabilities for the positive class
    y_scores =mnb.predict_proba(X_test.values)[:, i]
    
    # Compute ROC curve and AUC for the current class
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Store fpr, tpr, and auc for plotting
    all_fpr.append(fpr)
    all_tpr.append(tpr)
    all_auc.append(roc_auc)

# Plot ROC curves for all classes
plt.figure(figsize=(12, 8))

# Plot ROC curves for each class
for i in range(num_classes):
    plt.plot(all_fpr[i], all_tpr[i], label='Class {} (AUC = {:.2f})'.format(i, all_auc[i]))

plt.plot([0, 1], [0, 1], linestyle='--', color='k', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Class multinomialnb')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='x-small')
plt.xticks(fontsize='small')
plt.yticks(fontsize='small')
plt.show()

# Initialize a list to store the fpr and tpr for each class
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Initialize empty lists to store fpr, tpr, and auc for each class
all_fpr = []
all_tpr = []
all_auc = []
num_classes = 41  # Assuming num_classes is 41

# Iterate over each class
for i in range(num_classes):
    # Treat the current class as positive, and all others as negative
    y_true = (y_test_encoded == i)
    
    # Predict probabilities for the positive class
    y_scores = final_svm_model.predict_proba(X_test.values)[:, i]
    
    # Compute ROC curve and AUC for the current class
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Store fpr, tpr, and auc for plotting
    all_fpr.append(fpr)
    all_tpr.append(tpr)
    all_auc.append(roc_auc)

# Plot ROC curves for all classes
plt.figure(figsize=(12, 8))

# Plot ROC curves for each class
for i in range(num_classes):
    plt.plot(all_fpr[i], all_tpr[i], label='Class {} (AUC = {:.2f})'.format(i, all_auc[i]))

plt.plot([0, 1], [0, 1], linestyle='--', color='k', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Class svm')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='x-small')
plt.xticks(fontsize='small')
plt.yticks(fontsize='small')
plt.show()


# Initialize a list to store the fpr and tpr for each class
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Initialize empty lists to store fpr, tpr, and auc for each class
all_fpr = []
all_tpr = []
all_auc = []
num_classes = 41  # Assuming num_classes is 41

# Iterate over each class
for i in range(num_classes):
    # Treat the current class as positive, and all others as negative
    y_true = (y_test_encoded == i)
    
    # Predict probabilities for the positive class
    y_scores = lrrs.predict_proba(X_test.values)[:, i]
    
    # Compute ROC curve and AUC for the current class
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Store fpr, tpr, and auc for plotting
    all_fpr.append(fpr)
    all_tpr.append(tpr)
    all_auc.append(roc_auc)

# Plot ROC curves for all classes
plt.figure(figsize=(12, 8))

# Plot ROC curves for each class
for i in range(num_classes):
    plt.plot(all_fpr[i], all_tpr[i], label='Class {} (AUC = {:.2f})'.format(i, all_auc[i]))

plt.plot([0, 1], [0, 1], linestyle='--', color='k', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Class loogistic regression')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='x-small')
plt.xticks(fontsize='small')
plt.yticks(fontsize='small')
plt.show()

# Initialize a list to store the fpr and tpr for each class
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Initialize empty lists to store fpr, tpr, and auc for each class
all_fpr = []
all_tpr = []
all_auc = []
num_classes = 41  # Assuming num_classes is 41

# Iterate over each class
for i in range(num_classes):
    # Treat the current class as positive, and all others as negative
    y_true = (y_test_encoded == i)
    
    # Predict probabilities for the positive class
    y_scores = voting_classifier.predict_proba(X_test.values)[:, i]
    
    # Compute ROC curve and AUC for the current class
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Store fpr, tpr, and auc for plotting
    all_fpr.append(fpr)
    all_tpr.append(tpr)
    all_auc.append(roc_auc)

# Plot ROC curves for all classes
plt.figure(figsize=(12, 8))

# Plot ROC curves for each class
for i in range(num_classes):
    plt.plot(all_fpr[i], all_tpr[i], label='Class {} (AUC = {:.2f})'.format(i, all_auc[i]))

plt.plot([0, 1], [0, 1], linestyle='--', color='k', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Class soft voting')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='x-small')
plt.xticks(fontsize='small')
plt.yticks(fontsize='small')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp

# Assuming y_true contains the true labels and y_scores contains the predicted probabilities for each class

# Convert true labels to one-hot encoded format
y_true_binary = label_binarize(y_true, classes=np.arange(num_classes))

# Initialize lists to store FPR and TPR for each class
all_fpr = []
all_tpr = []
all_auc = []
y_scores = ge.predict_proba(X_test.values)[:, i]
# Calculate ROC curve for each class
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_true_binary[:, i], y_scores[:, i])
    roc_auc = auc(fpr, tpr)
    all_fpr.append(fpr)
    all_tpr.append(tpr)
    all_auc.append(roc_auc)

# Compute micro-average ROC curve and AUC
fpr_micro, tpr_micro, _ = roc_curve(y_true_binary.ravel(), y_scores.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)

# Plot ROC curve for each class
plt.figure(figsize=(10, 8))
for i in range(num_classes):
    plt.plot(all_fpr[i], all_tpr[i], label='Class {} ROC curve (AUC = {:.2f})'.format(i, all_auc[i]))

# Plot micro-average ROC curve
plt.plot(fpr_micro, tpr_micro, label='Micro-average ROC curve (AUC = {:.2f})'.format(roc_auc_micro), color='deeppink', linestyle=':')

plt.plot([0, 1], [0, 1], linestyle='--', color='k', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curve')
plt.legend()
plt.show()