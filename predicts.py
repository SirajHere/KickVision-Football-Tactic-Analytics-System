import sys
import json
def predict(symptom_data):
    import joblib
    import numpy as np
    import math
    import pickle
    print(symptom_data)
    l1= ['itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills','joint_pain','stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition','spotting_ urination','fatigue','weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy','patches_in_throat','irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness','sweating','dehydration','indigestion','headache','yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine','yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach','swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation','redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs','fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails','swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips','slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints','movement_stiffness','spinning_movements','loss_of_balance','unsteadiness','weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine','continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)','depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain','abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling','silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze']

    l2 = [0] * len(l1)
    
    for k in range(0, len(l1)):
        for z in symptom_data:
            if z == l1[k]:
                l2[k] = 1
    
    inputtest = [l2]
    
    
    with open('random_forest_model1.pkl', 'rb') as f:
        ge= pickle.load(f) 
    with open('multinomial_nb_model1.pkl', 'rb') as f:
        mnb= pickle.load(f) 
    with open('svm_model1.pkl', 'rb') as f:
        svm= pickle.load(f) 
    with open('logistic_regression_model1.pkl', 'rb') as f:
        lr= pickle.load(f) 
    
# load voting classifier model
    with open('voting_classifier_model1.pkl', 'rb') as f:
        voting_classifier= pickle.load(f) 

    predicted_disease=voting_classifier.predict(inputtest)
    pred_1=ge.predict(inputtest)
    pred_2=mnb.predict(inputtest)
    pred_3=svm.predict(inputtest)
    pred_4=lr.predict(inputtest)
    print( pred_1)
    print( pred_2)
    print( pred_3)
    print( pred_4)
    
   
    return predicted_disease


if __name__ == "__main__":
    # Retrieve symptoms from command-line arguments
    symptom_data =['watering_from_eyes','continuous_sneezing','headache','runny_nose','cough']
    # Predict disease based on symptoms
    predicted_disease = predict(symptom_data)
    # Print the predicted disease (this will be captured by shell_exec in PHP)
    print(predicted_disease)
