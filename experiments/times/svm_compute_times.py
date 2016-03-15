import time
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '',''))
import numpy as np
from sklearn import svm
from svmlight_loader import load_svmlight_file
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score

if len(sys.argv)<4:
    sys.exit("python compute_times.py matrix_dir shape outfile")

shape=int(sys.argv[2])
output=sys.argv[3]

cs=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
rs=42

times_file = open(output)
times_file.write("r,l,c,time\n")

for r in range(1,11):
    for l in [0.1, 0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.8]:
        # kODDSTC.r3.l1.0.mtx.svmlight
        fname = sys.argv[1]+".r"+str(r)+".l"+str(l)+".mtx.svmlight"

        km, target_array = load_svmlight_file(fname, shape, zero_based=True)

        for c in cs:

            start = time.clock()

            kf = cross_validation.StratifiedKFold(target_array, n_folds=10, shuffle=True,random_state=rs)
            gram=km.todense()

            for train_index, test_index in kf:

                clf = svm.SVC(C=c, kernel='precomputed', probability=True)
                train_gram = []
                test_gram = []
                  
                index=-1    
                for row in gram:
                    index+=1
                    if index in train_index:
                        train_gram.append(np.array(row).take(train_index))
                    else:
                        test_gram.append(np.array(row).take(train_index))

                X_train, X_test, y_train, y_test = np.array(train_gram), np.array(test_gram), target_array[train_index], target_array[test_index]

                #COMPUTE INNERKFOLD
                kif = cross_validation.StratifiedKFold(y_train, n_folds=10, shuffle=True, random_state=rs)
                inner_scores = cross_validation.cross_val_score(clf, X_train, y_train, scoring='roc_auc', cv=kif)

                clf.fit(X_train, y_train)

                y_proba=clf.predict_proba(X_test)
                y_test_predicted = [yp[1] for yp in y_proba]
                score = roc_auc_score(y_test, y_test_predicted)

            end = time.clock()
            times_file.write(str(r)+","+str(l)+","+str(c)+","+str(end - start)+"\n")

times_file.close()


