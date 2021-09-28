import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
COLUMNS=[]


'''
ref_PS Optician ID
nb_lignes_PEC Number of reimbursement requests
dep_identique Number of prescription from a geographical area identical to the optician area (%)
dep_limit Number of prescription from a geographical area bordering to the optician area (%)
dep_non_limit Number of prescription from a geographical area very distant to the optician area (%)
PEC_CRE Number of reimbursement requests in the ""created"" state (%)
PEC_ANN Number of reimbursement requests in the ""canceled"" state (%)
PEC_ACC Number of reimbursement requests in the ""granded"" state (%)
PEC_FAC Number of reimbursement requests in the ""invoiced"" state (%)
PEC_REF Number of reimbursement requests in the ""refused"" state (%) taux_non_ventes no sale rate (%)
Prix_monture means of price of the frame (euro)
Prix_verres means of final price of the 2 glasses (right + left) in euro
Prix_equipement means of total price
Pourcent_remise_verre means gross price of glasses
Pourcent_remise_monture means gross price of frame
Pourcent_remise_total means gross price of total equipment
cor_sphere_VD means of correction vision for sphere right
cor_cylindre_VD means of correction vision for cylinder right
cor_addition_VD means of correction vision for addition right
cor_sphere_VG means of correction vision for sphere left
cor_cylindre_VG means of correction vision for cylinder left
cor_addition_VG means of correction vision for addition left
facteur_correction all correction vision factor
age_moyen patient age
nb_prescripteurs number of differents prescriber (ophthalmologist)
nb_PEC_par_prescripteurs number of reimbursement requests by prescriber'''



# 1 --> Fraud
# 0 --> Not Fraud



# Loading the Dataset and Fraud IDs
df= pd.read_csv('optical_care_transaction_opticiens.csv', delimiter=';')
df_opticianID= pd.read_csv('almerys_fraudulent_optician_id.csv').values
df['fraud'] = 0

for i in range(0,len(df)) :
    if df.iloc[i, 0] in df_opticianID:
        df.at[i, 'fraud']=1  

        
####### Exploratory Data Analysis
fraud=df[df['fraud']==1]
not_fraud=df[df['fraud']==0]

def EDA(dataset): 
    profile =ProfileReport(dataset, title="Reimbursement Dataset Profile Report", explorative= True)
    profile.to_file("report.html")
    # profile.to_widgets()

####### Checking the VIF Factor###############





# Reducing the Majority class to 900 rows
fraud_df= df[df['fraud']==1]
normal_df= df[df['fraud']==0]
normal_df= normal_df.sample(900)

frames=[normal_df, fraud_df]
df=pd.concat(frames, axis=0).reset_index(drop= True)

# Defining the Dependent and Independent Variables        
X = df.iloc[:, 1:-1].values
y= df['fraud'].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)


# Handling Imbalanced Data
from imblearn.combine import SMOTEENN
sm = SMOTEENN()
X_train, y_train =  sm.fit_resample(X_train, y_train)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


############################SVM#####################################
# # Training the Kernel SVM model on the Training set
# from sklearn.svm import SVC
# classifier = SVC(kernel = 'rbf', random_state = 0, gamma=0.09)
# classifier.fit(X_train, y_train)

# # Applying k-Fold Cross Validation
# from sklearn.model_selection import cross_val_score
# accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
# print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
# print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

# # Applying Grid Search to find the best model and the best parameters
# from sklearn.model_selection import GridSearchCV
# parameters = [{'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']},
#               {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['rbf'], 'gamma': [0.03, 0.04, 0.05, 0.01, 0.02, 0.06, 0.07, 0.08, 0.09]}]
# grid_search = GridSearchCV(estimator = classifier,
#                            param_grid = parameters,
#                            scoring = 'accuracy',
#                            cv = 10,
#                            n_jobs = -1)
# grid_search.fit(X_train, y_train)
# best_accuracy = grid_search.best_score_
# best_parameters = grid_search.best_params_
# print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
# print("Best Parameters:", best_parameters)
############################SVM######################################



#########################RANDOM FOREST#######################################
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 300, criterion = 'gini', random_state = 0)
classifier.fit(X_train, y_train)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("10 Fold Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [100, 200, 300, 400,450, 500, 600], 'criterion': ['gini']},
              {'n_estimators': [100, 200, 300, 400,450, 500, 600], 'criterion': ['entropy']}]
grid_search = GridSearchCV(estimator = classifier,
                            param_grid = parameters,
                            scoring = 'recall',
                            cv = 10,
                            n_jobs = -1)

grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)


######################### RANDOM FOREST #######################################
# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, plot_roc_curve, fbeta_score
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()


# Evaluating the Performance
accuracy= (tp+tn)/(tp+tn+fp+fn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score= (2* precision * recall) / (precision + recall)


plot_roc_curve(classifier, X_test, y_test) 
print(f"Precision : {precision*100}\n Recall: {recall*100}\n F1 Score: {f1_score*100}\n Accuracy: {accuracy*100}\n")


if __name__ == "__main__":
    pass
    # EDA(df)