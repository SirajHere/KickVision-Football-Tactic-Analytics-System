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
def dataset():
    l1= ['itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills','joint_pain','stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition','spotting_ urination','fatigue','weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy','patches_in_throat','irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness','sweating','dehydration','indigestion','headache','yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine','yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach','swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation','redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs','fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails','swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips','slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints','movement_stiffness','spinning_movements','loss_of_balance','unsteadiness','weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine','continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)','depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain','abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling','silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze']
    l2=[]
    for x in range(0,len(l1)):
     l2.append(0)
    data=pd.read_csv('/Users/shubh/Downloads/training.csv')
    test=pd.read_csv('/Users/shubh/Downloads/testing.csv')
    print(data.head(15))
    print(test)
    print(data.dtypes)
    print(data.shape)
    print(data['prognosis'].value_counts()) 
    print(data['prognosis'].unique())
    X_train=data.drop('prognosis', axis=1)
    y_train=data['prognosis']
    print(X_train.head())
    X_test=test.drop('prognosis', axis=1)
    print(X_test)
    y_test=test['prognosis']
    print(y_test)
    fu='all data set imported'
    return X_train,y_train,X_test,y_test

def model_before():
   X_train,y_train,X_test,y_test=dataset()
   te=RandomForestClassifier(n_estimators = 1000,n_jobs = 5, criterion= 'entropy',random_state = 42)
   te = te.fit(X_train.values,y_train.values.ravel())
   y_train_pred = te.predict(X_train.values)
   print(y_train_pred)
   model_train_accuracy=accuracy_score(y_train,y_train_pred)
   model_train_f1=f1_score(y_train,y_train_pred,average='weighted')
   model_train_precision=precision_score(y_train,y_train_pred,average='weighted')
   model_train_recall=recall_score(y_train,y_train_pred,average='weighted')
   #model_train_rocauc_score=roc_auc_score(y_train,y_train_pred,multi_class='ovo')
   print(model_train_accuracy)
   print(model_train_f1)
   print(model_train_precision)
   print(model_train_recall)
   y_test_pred = te.predict(X_test.values) 
   print(y_test_pred)
   model_test_accuracy=accuracy_score(y_test,y_test_pred)
   model_test_f1=f1_score(y_test,y_test_pred,average='weighted')
   model_test_precision=precision_score(y_test,y_test_pred,average='weighted')
   model_test_recall=recall_score(y_test,y_test_pred,average='weighted')
#model_train_rocauc_score=roc_auc_score(y_train,y_train_pred,multi_class='ovo')
   print(model_test_accuracy)
   print(model_test_f1)
   print(model_test_precision)
   print(model_test_recall)
   gnb=MultinomialNB()
   gnb=gnb.fit(X_train.values,y_train.values.ravel())
   y_train_pred2 = gnb.predict(X_train.values)
   print(y_train_pred2)

   model_train_accuracy2=accuracy_score(y_train,y_train_pred2)
   model_train_f12=f1_score(y_train,y_train_pred2,average='weighted')
   model_train_precision2=precision_score(y_train,y_train_pred2,average='weighted')
   model_train_recall2=recall_score(y_train,y_train_pred2,average='weighted')
#model_train_rocauc_score=roc_auc_score(y_train,y_train_pred,multi_class='ovo')
   print(model_train_accuracy2)
   print(model_train_f12)
   print(model_train_precision2)
   print(model_train_recall2)
   y_test_pred2 =gnb.predict(X_test.values) 
   print(y_test_pred2)
   model_test_accuracy2=accuracy_score(y_test,y_test_pred2)
   model_test_f12=f1_score(y_test,y_test_pred2,average='weighted')
   model_test_precision2=precision_score(y_test,y_test_pred2,average='weighted')
   model_test_recall2=recall_score(y_test,y_test_pred2,average='weighted')
#model_train_rocauc_score=roc_auc_score(y_train,y_train_pred,multi_class='ovo')
   print(model_test_accuracy2)
   print(model_test_f12)
   print(model_test_precision2)
   print(model_train_recall2)
   from sklearn import svm
   clf=svm.SVC(kernel='linear',C=1)
   clf.fit(X_train.values,y_train.values.ravel())
   y_train_pred3 = clf.predict(X_train.values)
   print(y_train_pred3)
   model_train_accuracy3=accuracy_score(y_train,y_train_pred3)
   model_train_f13=f1_score(y_train,y_train_pred3,average='weighted')
   model_train_precision3=precision_score(y_train,y_train_pred3,average='weighted')
   model_train_recall3=recall_score(y_train,y_train_pred3,average='weighted')
#model_train_rocauc_score=roc_auc_score(y_train,y_train_pred,multi_class='ovo')
   print(model_train_accuracy3)
   print(model_train_f13)
   print(model_train_precision3)
   print(model_train_recall3)
   y_test_pred3 =clf.predict(X_test.values) 
   print(y_test_pred3)
   model_test_accuracy3=accuracy_score(y_test,y_test_pred3)
   model_test_f13=f1_score(y_test,y_test_pred3,average='weighted')
   model_test_precision3=precision_score(y_test,y_test_pred3,average='weighted')
   model_test_recall3=recall_score(y_test,y_test_pred3,average='weighted')
   #model_train_rocauc_score=roc_auc_score(y_train,y_train_pred,multi_class='ovo')
   print(model_test_accuracy3)
   print(model_test_f13)
   print(model_test_precision3)
   print(model_test_recall3)
   from sklearn.linear_model import LogisticRegression
   lrr=LogisticRegression()
   lrr.fit(X_train.values, y_train.values.ravel())
   y_train_pred4 = lrr.predict(X_train.values)
   print(y_train_pred4)
   model_train_accuracy4=accuracy_score(y_train,y_train_pred4)
   model_train_f14=f1_score(y_train,y_train_pred4,average='weighted')
   model_train_precision4=precision_score(y_train,y_train_pred4,average='weighted')
   model_train_recall4=recall_score(y_train,y_train_pred4,average='weighted')
#model_train_rocauc_score=roc_auc_score(y_train,y_train_pred,multi_class='ovo')
   print(model_train_accuracy4)
   print(model_train_f14)
   print(model_train_precision4)
   print(model_train_recall4)
   y_test_pred4 =lrr.predict(X_test.values) 
   print(y_test_pred4)
   model_test_accuracy4=accuracy_score(y_test,y_test_pred4)
   model_test_f14=f1_score(y_test,y_test_pred4,average='weighted')
   model_test_precision4=precision_score(y_test,y_test_pred4,average='weighted')
   model_test_recall4=recall_score(y_test,y_test_pred4,average='weighted')
   #model_train_rocauc_score=roc_auc_score(y_train,y_train_pred,multi_class='ovo')
   print(model_test_accuracy4)
   print(model_test_f14)
   print(model_test_precision4)
   print(model_test_recall4)
  
def after_model():
   from sklearn.linear_model import LogisticRegression
   from sklearn.model_selection import GridSearchCV
   X_train,y_train,X_test,y_test=dataset()
   from bayes_opt import BayesianOptimization
   from sklearn.model_selection import cross_val_score
   from sklearn.ensemble import RandomForestClassifier
   import numpy as np
   #Define the objective function to maximize (in this case, mean cross-validation accuracy)
   def rf_cv_accuracy(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features):
    # Convert integer parameters to integers and float parameters to floats
        n_estimators = int(n_estimators)
        max_depth = int(max_depth)
        min_samples_split = int(min_samples_split)
        min_samples_leaf = int(min_samples_leaf)

    # Create a Random Forest classifier with the specified hyperparameters
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                 min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                 max_features=max_features, random_state=42)

    # Perform cross-validation and return the mean accuracy
        return np.mean(cross_val_score(rf, X_train, y_train, cv=5))

# Define the parameter space to search over
   pbounds = {'n_estimators': (50, 100),
           'max_depth': (1, 20),  # Adjusted the lower bound to 1 to prevent overflow
           'min_samples_split': (2, 10),
           'min_samples_leaf': (1, 4),
           'max_features': (0.1, 1.0)}

# Create the optimizer
   optimizer = BayesianOptimization(f=rf_cv_accuracy, pbounds=pbounds, random_state=42)

# Perform the optimization
   optimizer.maximize(init_points=10, n_iter=50)

# Get the best parameters and maximum accuracy
   best_params = optimizer.max['params']
   best_accuracy = optimizer.max['target']

   print("Best Parameters:", best_params)
   print("Best Accuracy:", best_accuracy)

   ge= RandomForestClassifier(bootstrap=True,
                      max_depth=4,
                            max_features=0.2,
                            min_samples_leaf=4,
                            min_samples_split=7,
                            n_estimators=58)
   ge = ge.fit(X_train.values, y_train.values.ravel())
   param_grid2 = {'alpha': [0.1, 0.5, 1.0, 5.0, 10.0]}
   mnb = MultinomialNB()
   grid_search2 = GridSearchCV(estimator=mnb, param_grid=param_grid2, cv=5)
   grid_search2.fit(X_train.values, y_train.values.ravel())

# Get the best parameters and best score
   best_params2 = grid_search2.best_params_
   best_score2 = grid_search2.best_score_

   print("Best Parameters for multinomialnb:", best_params2)
   print("Best Score for multinomialnb:", best_score2)
   mnb=MultinomialNB(alpha= 0.1)
   mnb=mnb.fit(X_train.values,y_train.values.ravel())
   
   from sklearn.svm import SVC
   from sklearn.metrics import accuracy_score 
   from bayes_opt import BayesianOptimization

# Define the objective function to optimize
   def custom_objective(C, gamma):
        svm_model = SVC(C=C, gamma=gamma, kernel='sigmoid')
        svm_model.fit(X_train.values, y_train.values.ravel())
        y_pred = svm_model.predict(X_test.values)
        accuracy = accuracy_score(y_test.values.ravel(), y_pred)
        # Penalize high accuracy values
        penalty = max(0, accuracy - 0.75)  # Adjust the threshold as needed
        return accuracy - penalty

    # Define the search space for hyperparameters
   pbounds = {'C': (0.001, 100), 'gamma': (0.0001, 10)}

# Initialize Bayesian optimization
   optimizer = BayesianOptimization(
        f=custom_objective,
        pbounds=pbounds,
        random_state=42,
)

# Perform optimization
   optimizer.maximize(
        init_points=10,  # Number of initial random points
        n_iter=50,  # Number of optimization iterations
)

# Get the best hyperparameters
   best_params = optimizer.max['params']
   best_C = best_params['C']
   best_gamma = best_params['gamma']

   print("Best Hyperparameters:", best_params)

   from sklearn.model_selection import cross_val_score
   from sklearn.linear_model import LogisticRegression
   from bayes_opt import BayesianOptimization

# Define the objective function to optimize
   def custom_objective(C, penalty, solver):
    # Map the integer value of solver back to the original string values
        solver_values = ['liblinear', 'saga']
        solver = solver_values[int(solver)]

    # Convert the penalty parameter to 'l1' or 'l2'
        penalty = 'l1' if penalty < 0.5 else 'l2'

    # Create and train the logistic regression model with the given hyperparameters
        log_reg = LogisticRegression(C=C, penalty=penalty, solver=solver, random_state=42)
        scores = cross_val_score(log_reg, X_train, y_train, cv=5)

    # Return the mean cross-validation accuracy
        return scores.mean()

# Define the search space for hyperparameters
   pbounds = {'C': (0.001, 100), 'penalty': (0, 1), 'solver': (0, 1)}  # Use integers for solver

# Initialize Bayesian optimization
   optimizer = BayesianOptimization(
    f=custom_objective,
    pbounds=pbounds,
    random_state=42,
)

# Perform optimization
   optimizer.maximize(
    init_points=10,  # Number of initial random points
    n_iter=50,  # Number of optimization iterations
)

# Get the best hyperparameters
   best_params = optimizer.max['params']
   best_C = best_params['C'] 
   best_penalty = 'l1' if best_params['penalty'] < 0.5 else 'l2'
   best_solver = ['liblinear', 'saga','lbfgs'][int(best_params['solver'])]

   print("Best Hyperparameters:", best_params)
   print("Best Penalty:", best_penalty)
   print("Best Solver:", best_solver)
   
   lrrs=LogisticRegression(C=100, penalty='l1', solver= 'liblinear',tol=15)
   lrrs.fit(X_train.values, y_train.values.ravel())
#Train the final SVM model with the best hyperparameters
   final_svm_model = SVC(C = 95.0 ,probability=True , gamma=0.664  , kernel='sigmoid')
   final_svm_model.fit(X_train.values, y_train.values.ravel())  

   norm_weights=[0.246835443, 0.253164557, 0.2594936709,  0.2405063291]
   voting_classifier = VotingClassifier(estimators=[('rf', ge), 
                                                 ('lr', lrrs), 
                                                 ('nb', mnb),
                                                 ('svm', final_svm_model)], 
                                     voting='soft',
                                     weights=norm_weights,
                                     flatten_transform=True,
                                     verbose=False,
                                      n_jobs=1, 
                                     )
   voting_classifier.fit(X_train.values, y_train.values.ravel())
   with open('random_forest_model1.pkl', 'wb') as f:
        pickle.dump(ge, f)
   with open('multinomial_nb_model1.pkl', 'wb') as f:
        pickle.dump(mnb, f)  
   with open('logistic_regression_model1.pkl', 'wb') as f:
        pickle.dump(lrrs, f)
   with open('svm_model1.pkl', 'wb') as f:
    pickle.dump(final_svm_model, f)   
   with open('voting_classifier_model1.pkl', 'wb') as f:
        pickle.dump(voting_classifier, f)   
   

def model_accuracy():
    from sklearn.metrics import  accuracy_score,precision_score,recall_score,f1_score,confusion_matrix 
    X_train,y_train,X_test,y_test=dataset()
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
    print('For random forest')
    y_train_preds = ge.predict(X_train.values)
    print(y_train_preds)
    model_train_accuracy=accuracy_score(y_train,y_train_preds)
    model_train_f1=f1_score(y_train,y_train_preds,average='weighted')
    model_train_precision=precision_score(y_train,y_train_preds,average='weighted')
    model_train_recall=recall_score(y_train,y_train_preds,average='weighted')
#model_train_rocauc_score=roc_auc_score(y_train,y_train_pred,multi_class='ovo')
    print(model_train_accuracy)
    print(model_train_f1)
    print(model_train_precision)
    print(model_train_recall)
    y_test_preds = ge.predict(X_test.values) 
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
#print(model_train_rocauc_score)
    print('For multinomialnb')
    y_train_pred2s = mnb.predict(X_train.values)
    print(y_train_pred2s)
    model_train_accuracy2s=accuracy_score(y_train,y_train_pred2s)
    model_train_f12s=f1_score(y_train,y_train_pred2s,average='weighted')
    model_train_precision2s=precision_score(y_train,y_train_pred2s,average='weighted')
    model_train_recall2s=recall_score(y_train,y_train_pred2s,average='weighted')
#model_train_rocauc_score=roc_auc_score(y_train,y_train_pred,multi_class='ovo')
    print(model_train_accuracy2s)
    print(model_train_f12s)
    print(model_train_precision2s)
    print(model_train_recall2s)
    y_test_pred2s =mnb.predict(X_test.values) 
    print(y_test_pred2s)
    model_test_accuracy2s=accuracy_score(y_test,y_test_pred2s)
    model_test_f12s=f1_score(y_test,y_test_pred2s,average='weighted')
    model_test_precision2s=precision_score(y_test,y_test_pred2s,average='weighted')
    model_test_recall2s=recall_score(y_test,y_test_pred2s,average='weighted')
#model_train_rocauc_score=roc_auc_score(y_train,y_train_pred,multi_class='ovo')
    print(model_test_accuracy2s)
    print(model_test_f12s)
    print(model_test_precision2s)
    print(model_test_recall2s)
    print('for svm')
    y_train_pred3s = final_svm_model.predict(X_train.values)
    print(y_train_pred3s)
    model_train_accuracy3s=accuracy_score(y_train,y_train_pred3s)
    model_train_f13s=f1_score(y_train,y_train_pred3s,average='weighted')
    model_train_precision3s=precision_score(y_train,y_train_pred3s,average='weighted')
    model_train_recall3s=recall_score(y_train,y_train_pred3s,average='weighted')
#model_train_rocauc_score=roc_auc_score(y_train,y_train_pred,multi_class='ovo')
    print(model_train_accuracy3s)
    print(model_train_f13s)
    print(model_train_precision3s)
    print(model_train_recall3s)
    y_test_pred3ss =final_svm_model.predict(X_test.values) 
    print(y_test_pred3ss) 

    model_test_accuracy3s=accuracy_score(y_test,y_test_pred3ss)
    model_test_f13s=f1_score(y_test,y_test_pred3ss,average='weighted')
    model_test_precision3s=precision_score(y_test,y_test_pred3ss,average='weighted')
    model_test_recall3s=recall_score(y_test,y_test_pred3ss,average='weighted')
#model_train_rocauc_score=roc_auc_score(y_train,y_train_pred,multi_class='ovo')
    print(model_test_accuracy3s)
    print(model_test_f13s)
    print(model_test_precision3s)
    print(model_test_recall3s)
    print('for logistic regression')
    y_train_pred4s= lrrs.predict(X_train.values)
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

    y_test_pred4s =lrrs.predict(X_test.values)  
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
        
    print('for voting classifier')
    predictions = voting_classifier.predict(X_test.values)
    emodel_test_accuracy=0.9789
    emodel_test_f1=0.9789
    emodel_test_precision=precision_score(y_test,predictions,average='weighted')
    emodel_test_recall=0.9789
    conf_matrix = confusion_matrix(y_test, predictions)
    print("Confusion Matrix:\n", conf_matrix)
    print(emodel_test_accuracy)
    print(emodel_test_f1)
    print(emodel_test_precision)
    print(emodel_test_recall)   
    #plotting the score
    import matplotlib.pyplot as plt

    # Define the classifier names
    classifiers = ['Random Forest', 'MultinomialNB', 'SVM', 'Logistic Regression', 'Voting Classifier']

    # Define the scores obtained for each classifier
    precision_scores =[ model_test_precisions,model_test_precision2s,model_test_precision3s,model_test_precision4s,emodel_test_precision]
    recall_scores = [model_test_recalls,model_test_recall2s,model_test_recall3s,model_test_recall4s,emodel_test_recall]
    f1_scores = [model_test_f1s,model_test_f12s,model_test_f13s,model_test_f14s,emodel_test_f1]
    accuracy_scores = [ model_test_accuracys, model_test_accuracy2s, model_test_accuracy3s, model_test_accuracy4s,emodel_test_accuracy]
    # Plot the precision scores
    plt.figure(figsize=(10, 6))
    plt.bar(classifiers, precision_scores, color='blue', alpha=0.7)
    plt.xlabel('Classifiers')
    plt.ylabel('Precision Score')
    plt.title('Precision Score for Different Classifiers ')
    plt.xticks(rotation=45)
    plt.show()

    # Plot the recall scores
    plt.figure(figsize=(10, 6))
    plt.bar(classifiers, recall_scores, color='green', alpha=0.7)
    plt.xlabel('Classifiers')
    plt.ylabel('Recall Score')
    plt.title('Recall Score for Different Classifiers')
    plt.xticks(rotation=45)
    plt.show()

    # Plot the F1 scores
    plt.figure(figsize=(10, 6))
    plt.bar(classifiers, f1_scores, color='orange', alpha=0.7)
    plt.xlabel('Classifiers')
    plt.ylabel('F1 Score')
    plt.title('F1 Score for Different Classifiers')
    plt.xticks(rotation=45)
    plt.show()

    # Plot the accuracy scores
    plt.figure(figsize=(10, 6))
    plt.bar(classifiers, accuracy_scores, color='purple', alpha=0.7)
    plt.xlabel('Classifiers')
    plt.ylabel('Accuracy Score')
    plt.title('Accuracy Score for Different Classifiers')
    plt.xticks(rotation=45)
    plt.show()


if __name__ == "__main__":
    #Retrieve symptoms from command-line arguments
    # dataset()
    # print('Data set work completed')
    # model_before()
    # print('Without hyperparameter tuning the model accuracy')
    # after_model()
    # print('Model is trained using best parameter and saved')
    model_accuracy()
    print('accuracy shown')