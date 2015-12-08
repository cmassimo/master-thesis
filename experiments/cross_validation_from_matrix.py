import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '',''))
import numpy as np
from sklearn import svm

if len(sys.argv)<4:
    sys.exit("python cross_validation_from_matrix.py inputMatrix.libsvm C outfile")

c=float(sys.argv[2])

from sklearn.datasets import load_svmlight_file
km, target_array = load_svmlight_file(sys.argv[1])

from sklearn import cross_validation
sc=[]
for rs in range(42,52):
    f=open(str(sys.argv[3]+".seed"+str(rs)+".c"+str(c)),'w')

    kf = cross_validation.StratifiedKFold(target_array, n_folds=10, shuffle=True,random_state=rs)
    #print kf    
    #remove column zero because
    #first entry of each line is the index
    
    #gram=km[:,1:].todense()
    gram=km.todense()
    
    f.write("Total examples "+str(len(gram))+"\n")
    f.write("CV\t\t test_acc\n")
    #print gram
    
    #sc=[] TODO spostato sopra
    for train_index, test_index in kf:
        #print("TRAIN:", train_index, "TEST:", test_index)
    
        #generated train and test lists, incuding indices of the examples in training/test
        #for the specific fold. Indices starts from 0 now
        
        clf = svm.SVC(C=c, kernel='precomputed')
        train_gram = [] #[[] for x in xrange(0,len(train))]
        test_gram = []# [[] for x in xrange(0,len(test))]
          
        #generate train matrix and test matrix
        index=-1    
        for row in gram:
            index+=1
            if index in train_index:
                train_gram.append([gram[index,i] for i in train_index])
            else:
                test_gram.append([gram[index,i] for i in train_index])
    
    
    
        #print gram
        X_train, X_test, y_train, y_test = np.array(train_gram), np.array(test_gram), target_array[train_index], target_array[test_index]
        #COMPUTE INNERKFOLD
        kif = cross_validation.StratifiedKFold(y_train, n_folds=10, shuffle=True,random_state=rs)
        inner_scores= cross_validation.cross_val_score(
        clf, X_train, y_train, cv=kif)
        #print "inner scores", inner_scores
        print "Inner Accuracy: %0.8f (+/- %0.8f)" % (inner_scores.mean(), inner_scores.std())

        f.write(str(inner_scores.mean())+"\t")
    
        clf.fit(X_train, y_train)
    
        from sklearn.metrics import accuracy_score
        # predict on test examples
        y_test_predicted=clf.predict(X_test)
        sc.append(accuracy_score(y_test, y_test_predicted))
        f.write(str(accuracy_score(y_test, y_test_predicted))+"\t")
        
        f.write(str(inner_scores.std())+"\n")

    f.close()
scores=np.array(sc) #sc dovrebbe essere accuracy non nested sui vari random seed di 10-fold.
print "Accuracy: %0.8f (+/- %0.8f)" % (scores.mean(), scores.std())
