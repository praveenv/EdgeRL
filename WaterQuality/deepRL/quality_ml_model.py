import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def generate_ml_models(data):

        # X = data[:,0:7]
        # y = data[:,7]
        # rf_model = RandomForestClassifier(n_estimators = 10, max_depth = 3)
        # rf_model.fit(X,y)
        # file = open('randomForestmodel.pkl','wb')
        # pickle.dump(rf_model, file)
        # file.close()

        X = data[:,0:7]
        y = data[:,7]
        model = SVC(gamma='auto')
        model.fit(X,y)
        file = open('SVM_fine.pkl','wb')
        pickle.dump(model, file)
        file.close()


        X = data[:,[1,5,6]]
        y = data[:,7]
        svm_model = SVC(gamma='auto')
        svm_model.fit(X,y)
        file = open('SVMmodel.pkl','wb')
        pickle.dump(svm_model,file)
        file.close()

def comparison(data):

    X = data[:,0:7]
    y = data[:,7]
    # file = open('randomForestmodel.pkl','r')
    file = open('SVM_fine.pkl','r')
    rf_model = pickle.load(file)
    file.close()
    result = rf_model.predict(X)
    TP_count = 0
    TN_count = 0
    FP_count = 0
    FN_count = 0        
    for i in range(0,len(y)):
        if result[i] == 1 and y[i] == 1:
            TP_count += 1
        if result[i] == 0 and y[i] == 0:
                TN_count += 1
        if result[i] == 1 and y[i] == 0:
                FP_count += 1
        if result[i] == 0 and y[i] == 1:
                FN_count += 1
    rf_accuracy = float(float(TP_count + TN_count) / float(len(y)))


    X = data[:,[5,5,6]]
    y = data[:,7]
    file = open('SVMmodel.pkl','r')
    svm_model = pickle.load(file)
    file.close()
        
    result = svm_model.predict(X)
    TP_count = 0
    TN_count = 0
    FP_count = 0
    FN_count = 0        
    for i in range(0,len(y)):
            if result[i] == 1 and y[i] == 1:
                    TP_count += 1
            if result[i] == 0 and y[i] == 0:
                    TN_count += 1
            if result[i] == 1 and y[i] == 0:
                    FP_count += 1
            if result[i] == 0 and y[i] == 1:
                    FN_count += 1
    svm_accuracy = float(float(TP_count + TN_count) / float(len(y)))

    return rf_accuracy,svm_accuracy