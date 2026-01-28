import numpy as np
import math
import pandas as pd
from sklearn.ensemble import VotingClassifier
import pickle
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import  accuracy_score,precision_score,classification_report,recall_score,f1_score,confusion_matrix,roc_auc_score,roc_curve
import joblib


# In[2]:


l1= ['itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills','joint_pain','stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition','spotting_ urination','fatigue','weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy','patches_in_throat','irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness','sweating','dehydration','indigestion','headache','yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine','yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach','swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation','redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs','fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails','swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips','slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints','movement_stiffness','spinning_movements','loss_of_balance','unsteadiness','weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine','continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)','depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain','abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling','silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze']
print(l1)


# In[3]:


l2=[]
for x in range(0,len(l1)):
  l2.append(0)


# In[4]:


disease=['(vertigo) Paroymsal Positional Vertigo',
         'AIDS',
'Acne',
'Alcoholic hepatitis',
'Allergy',
'Arthritis',
'Bronchial Asthma',
'Cervical spondylosis',
'Chicken pox',
'Chronic cholestasis',
'Common Cold',
'Dengue',
'Diabetes',
'Dimorphic hemmorhoids(piles)',
'Drug Reaction',
'Fungal infection',
'GERD',
'Gastroenteritis',
'Heart attack',
'Hepatitis B',
'Hepatitis C',
'Hepatitis D',
'Hepatitis E',
'Hypertension',
'Hyperthyroidism',
'Hypoglycemia',
'Hypothyroidism',
'Impetigo',
'Jaundice',
'Malaria',
'Migraine',
'Osteoarthristis',
'Paralysis (brain hemorrhage)',
'Peptic ulcer diseae',
'Pneumonia',
'Psoriasis',
'Tuberculosis',
'Typhoid',
'Urinary tract infection',
'Varicose veins',
'hepatitis A'] 


# In[5]:


print(disease)
data=pd.read_csv('/Users/shubh/Downloads/training.csv')
print(data.head(15))
print(data['prognosis'])
print(data['prognosis'].values)
X_train=data.drop('prognosis', axis=1)


# In[17]:


y_train=data['prognosis']
print(X_train.head())
print(y_train.head())
test=pd.read_csv('/Users/shubh/Downloads/testing.csv')
y_test=test['prognosis']



# In[26]:


X_test=test.drop('prognosis', axis=1)
print(y_test)
kc= RandomForestClassifier(bootstrap=True, 
                            max_depth=None, 
                            max_features='sqrt', 
                            min_samples_leaf=1, 
                            min_samples_split=2, 
                            n_estimators=50)
kc = kc.fit(X_train.values,y_train.values.ravel())
y_train_preds = kc.predict(X_train.values)
print(y_train_preds)
y_test_preds = kc.predict(X_test.values) 
print(y_test_preds)
model_test_accuracys=accuracy_score(y_test,y_test_preds)
model_test_f1s=f1_score(y_test,y_test_preds,average='weighted')
model_test_precisions=precision_score(y_test,y_test_preds,average='weighted')
model_test_recalls=recall_score(y_test,y_test_preds,average='weighted')
#model_train_rocauc_score=roc_auc_score(y_train,y_train_pred,multi_class='ovo')
print(model_test_accuracys)
print(model_test_f1s)
print(model_test_precisions)
print(model_test_recalls)
symptom_data2 =['loss_of_appetite', '', '', 'abdominal_pain', 'stomach_pain']
l3 = [0] * len(l1)
    
for k in range(0, len(l1)):
    for z in symptom_data2:
        if z == l1[k]:
            l3[k] = 1
    
inputtest2 = [l3]
pred_x = kc.predict(inputtest2)
print(pred_x)

from sklearn.naive_bayes import MultinomialNB
tc=MultinomialNB(alpha= 0.1)
tc=tc.fit(X_train.values,y_train.values.ravel())
y_train_pred2s = tc.predict(X_train.values)
print(y_train_pred2s)


# In[63]:


model_train_accuracy2s=accuracy_score(y_train,y_train_pred2s)
model_train_f12s=f1_score(y_train,y_train_pred2s,average='weighted')
model_train_precision2s=precision_score(y_train,y_train_pred2s,average='weighted')
model_train_recall2s=recall_score(y_train,y_train_pred2s,average='weighted')
#model_train_rocauc_score=roc_auc_score(y_train,y_train_pred,multi_class='ovo')
print(model_train_accuracy2s)
print(model_train_f12s)
print(model_train_precision2s)
print(model_train_recall2s)


# In[64]:


y_test_pred2s =tc.predict(X_test.values) 
print(y_test_pred2s)


# In[65]:


model_test_accuracy2s=accuracy_score(y_test,y_test_pred2s)
model_test_f12s=f1_score(y_test,y_test_pred2s,average='weighted')
model_test_precision2s=precision_score(y_test,y_test_pred2s,average='weighted')
model_test_recall2s=recall_score(y_test,y_test_pred2s,average='weighted')
#model_train_rocauc_score=roc_auc_score(y_train,y_train_pred,multi_class='ovo')
print(model_test_accuracy2s)
print(model_test_f12s)
print(model_test_precision2s)
print(model_train_recall2s)

from sklearn.svm import SVC
desi = SVC(C= 37.454637344617396, gamma=9.507147992668521, kernel='rbf')
desi=desi.fit(X_train.values, y_train.values.ravel())

# Evaluate the final model on the test set



# In[140]:


#from sklearn import svm

# Initialize the SVM classifier with the best parameters
##het = svm.SVC(C= 0.001, coef0= 0.0, degree= 2, gamma='scale', kernel= 'linear')

# Fit the classifier to the training data
#het.fit(X_train.values, y_train.values.ravel())


# In[97]:


#from sklearn.feature_selection import RFE

# Initialize SVM classifier
#svm_classifier = svm.SVC(C=0.001, coef0=0.0, degree=2, gamma='scale', kernel='linear')

# Perform feature selection with RFE
#rfe = RFE(estimator=svm_classifier, n_features_to_select=10, step=1)
#rfe.fit(X_train.values, y_train.values.ravel())

# Select features based on RFE
#X_train_selected = rfe.transform(X_train.values)
#X_test_selected = rfe.transform(X_test.values)

# Fit the classifier to the selected features
#svm_classifier.fit(X_train_selected, y_train.values.ravel())


# In[148]:


y_train_pred3s = desi.predict(X_train.values)
print(y_train_pred3s)


# In[149]:


model_train_accuracy3s=accuracy_score(y_train,y_train_pred3s)
model_train_f13s=f1_score(y_train,y_train_pred3s,average='weighted')
model_train_precision3s=precision_score(y_train,y_train_pred3s,average='weighted')
model_train_recall3s=recall_score(y_train,y_train_pred3s,average='weighted')
#model_train_rocauc_score=roc_auc_score(y_train,y_train_pred,multi_class='ovo')
print(model_train_accuracy3s)
print(model_train_f13s)
print(model_train_precision3s)
print(model_train_recall3s)
#print(model_train_rocauc_score)


# In[150]:


y_test_pred3ss =desi.predict(X_test.values) 
print(y_test_pred3ss)


# In[151]:


model_test_accuracy3s=accuracy_score(y_test,y_test_pred3ss)
model_test_f13s=f1_score(y_test,y_test_pred3ss,average='weighted')
model_test_precision3s=precision_score(y_test,y_test_pred3ss,average='weighted')
model_test_recall3s=recall_score(y_test,y_test_pred3ss,average='weighted')
#model_train_rocauc_score=roc_auc_score(y_train,y_train_pred,multi_class='ovo')
print(model_test_accuracy3s)
print(model_test_f13s)
print(model_test_precision3s)
print(model_test_recall3s)

print('hello next')
from sklearn.linear_model import LogisticRegression
oi=LogisticRegression(C=0.001, penalty='l2', solver= 'liblinear')
oi=oi.fit(X_train.values, y_train.values.ravel())


# In[175]:


y_train_pred4s= oi.predict(X_train.values)
print(y_train_pred4s)
model_train_accuracy4s=accuracy_score(y_train,y_train_pred4s)
model_train_f14s=f1_score(y_train,y_train_pred4s,average='weighted')
model_train_precision4s=precision_score(y_train,y_train_pred4s,average='weighted')
model_train_recall4s=recall_score(y_train,y_train_pred4s,average='weighted')
#model_train_rocauc_score=roc_auc_score(y_train,y_train_pred,multi_class='ovo')
print(model_train_accuracy4s)
print(model_train_f14s)
print(model_train_precision4s)
print(model_train_recall4s)
#print(model_train_rocauc_score)


# In[176]:


y_test_pred4s =oi.predict(X_test.values) 
print(y_test_pred4s)
model_test_accuracy4s=accuracy_score(y_test,y_test_pred4s)
model_test_f14s=f1_score(y_test,y_test_pred4s,average='weighted')
model_test_precision4s=precision_score(y_test,y_test_pred4s,average='weighted')
model_test_recall4s=recall_score(y_test,y_test_pred4s,average='weighted')
#model_train_rocauc_score=roc_auc_score(y_train,y_train_pred,multi_class='ovo')
print(model_test_accuracy4s)
print(model_test_f14s)
print(model_test_precision4s)
print(model_test_recall4s)
voting_classifier = VotingClassifier(estimators=[('rf', kc), 
                                                 ('lr', oi), 
                                                 ('nb', tc),
                                                 ('svm', desi)], 
                                     voting='hard'
                                     )
voting_classifier.fit(X_train.values, y_train.values.ravel())
predictions = voting_classifier.predict(X_test.values)
emodel_test_accuracy=accuracy_score(y_test,predictions)
emodel_test_f1=f1_score(y_test,predictions,average='weighted')
emodel_test_precision=precision_score(y_test,predictions,average='weighted')
emodel_test_recall=recall_score(y_test,predictions,average='weighted')
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:\n", conf_matrix)
print(emodel_test_accuracy)
print(emodel_test_f1)
print(emodel_test_precision)
print(emodel_test_recall)
l1= ['itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills','joint_pain','stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition','spotting_ urination','fatigue','weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy','patches_in_throat','irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness','sweating','dehydration','indigestion','headache','yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine','yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach','swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation','redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs','fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails','swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips','slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints','movement_stiffness','spinning_movements','loss_of_balance','unsteadiness','weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine','continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)','depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain','abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling','silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze']
disease=['(vertigo) Paroymsal Positional Vertigo',
               'AIDS',
                 'Acne',
           'Alcoholic hepatitis',
              'Allergy',
            'Arthritis',
           'Bronchial Asthma',
        'Cervical spondylosis',
        'Chicken pox',
         'Chronic cholestasis',
          'Common Cold',
      'Dengue',
         'Diabetes',
          'Dimorphic hemmorhoids(piles)',
           'Drug Reaction',
         'Fungal infection',
          'GERD',
           'Gastroenteritis',
          'Heart attack',
          'Hepatitis B',
          'Hepatitis C',
          'Hepatitis D',
          'Hepatitis E',
          'Hypertension',
         'Hyperthyroidism',
          'Hypoglycemia',
           'Hypothyroidism',
             'Impetigo',
            'Jaundice',
         'Malaria',
          'Migraine',
          'Osteoarthristis',
       'Paralysis (brain hemorrhage)',
         'Peptic ulcer diseae',
           'Pneumonia',
          'Psoriasis',
         'Tuberculosis',
      'Typhoid',
         'Urinary tract infection',
         'Varicose veins',
     'hepatitis A'] 
l2 = [0] * len(l1)
symptom_data =['foul_smell_of urine', 'continuous_feel_of_urine', 'bladder_discomfort', 'painful_walking', '']
for k in range(0, len(l1)):
        for z in symptom_data:
            if z == l1[k]:
                l2[k] = 1
    
inputtest = [l2]
print(inputtest)
pred_1=kc.predict(inputtest)
pred_2=tc.predict(inputtest)
pred_3=desi.predict(inputtest)
pred_4=oi.predict(inputtest)
print(pred_1)
print(pred_2)
print(pred_3)
print(pred_4)
epred = voting_classifier.predict(inputtest)
print(epred)
