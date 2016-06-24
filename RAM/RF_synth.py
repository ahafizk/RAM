__author__ = 'hafiz'
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from utility import *
from  re import search
import numpy as np
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,accuracy_score
import pickle
from sklearn.externals import joblib

class RFClassifier():
    model = None
    def __init__(self):
        self.sub_activity_names ={0:'Brushing',1:'Washing Face', 2:'Spraying',3:'Wiping'} # showing the output
        self.activity_names = {0:'Cleaning Activity',1:'Bathroom Activity'} #this is for showing in the output
        self.act_list = ['clean','bathroom'] # consider this for labeled data
        # self.sub_act_list = ['']

    def get_activity_label(self,name):
        '''
        This class returns then integer label for each activity if name found in the activity
        otherwise return -1
        '''
        for i  in xrange(len(self.act_list)):
            if search(self.act_list[i],name.lower()):
                return i
        return -1

    def get_model(self):
        # if self.model is not None:
        #     return self.model
        # else return
        return self.model

    def get_sub_activity_name(self,label):
        return self.sub_activity_names[label]

    def get_activity_names(self,label):
        return self.activity_names[label]

    def make_train_test_set(self,file1,file2):
        import csv
        reader = csv.reader(open(file1, 'rb'))
        reader1 = csv.reader(open(file2, 'rb'))
        writer = csv.writer(open('appended_output.csv', 'wb'))
        for row in reader:
            row1 = reader1.next()
            writer.writerow(row + row1)

    def predict_class(self,test):
        # clf = joblib.load('savedmodel/RFM.pk1')
        # print test.shape
        clf = self.model
        y_ = clf.predict(test[:,0:25])
        return y_

    def test_rf_model(self, test_file=''):
        if test_file == '':
            train_file = 'file name with the directory'
            print 'Enter Test file along with the directory'
            return
        clf = joblib.load('savedmodel/RFM.pk1')
        test = np.genfromtxt(test_file, delimiter=',')
        X = test[:,0:25] # 26 features
        y = test[:,26] # last column for label - here sub-activity
        y_ = clf.predict(test[:,0:25])
        cm = confusion_matrix(y, y_)
        print cm
        print clf.score(test[:,0:25],test[:,26])
        return y_

    def generate_rf_model(self,train_file=''):
        if train_file == '':
            train_file = 'file name with the directory'
            print 'Enter Train file along with the directory'
            return
        clf = Pipeline([
          ('feature_selection', LinearSVC(C=0.01, penalty="l1", dual=False)),
          ('classification', RandomForestClassifier())
            ])
        train = np.genfromtxt(train_file, delimiter=',')
        print train.shape
        X = train[:,0:25] # 26 features
        y = train[:,26] # last column for label - here sub-activity
        # print y
        clf.fit(X, y)
        self.model = clf

        # s = pickle.dumps(clf,'RFM.pk1') #save the model
        joblib.dump(clf,'savedmodel/RFM.pk1')

    def create_train_test_file(self,dirname='features/'):
        # dirname = 'features/'

        rlst = ['radar1','radar2','radar3','radar4']
        for rname in rlst:
            filelist = get_file_list(dirname+rname+'/')
            trfile = dirname+rname+'/'+rname+'_train.csv'
            tsfile = dirname+rname+'/'+rname+'_test.csv'
            tl = False
            trl = False
            test = np.array([])
            train = np.array([])
            for name in filelist:
                file = dirname+rname+'/'+name
                if search('p3',name):
                    if tl==False:
                        test = np.genfromtxt(file, delimiter=',')
                        tl = True
                    else:
                        tst = np.genfromtxt(file, delimiter=',')
                        test = np.vstack((test,tst))
                        # print 'test'
                        # print test.shape
                else:
                    if trl==False:
                        train = np.genfromtxt(file, delimiter=',')
                        trl = True
                    else:
                        tr = np.genfromtxt(file, delimiter=',')
                        train = np.vstack((train,tr))
                        # print 'train'
                        # print train.shape

            np.savetxt(trfile, train, delimiter=",")
            np.savetxt(tsfile, test, delimiter=",")
    def main(self):
        dirname = 'features/'

        rlst = ['radar1','radar2','radar3','radar4']
        for rname in rlst:
            filelist = get_file_list(dirname+rname+'/')
            trfile = dirname+rname+'/'+rname+'_train.csv'
            tsfile = dirname+rname+'/'+rname+'_test.csv'
            tl = False
            trl = False
            test = np.array([])
            train = np.array([])
            for name in filelist:
                file = dirname+rname+'/'+name
                if search('p3',name):
                    if tl==False:
                        test = np.genfromtxt(file, delimiter=',')
                        tl = True
                    else:
                        tst = np.genfromtxt(file, delimiter=',')
                        test = np.vstack((test,tst))
                        # print 'test'
                        # print test.shape
                else:
                    if trl==False:
                        train = np.genfromtxt(file, delimiter=',')
                        trl = True
                    else:
                        tr = np.genfromtxt(file, delimiter=',')
                        train = np.vstack((train,tr))
                        # print 'train'
                        # print train.shape

            np.savetxt(trfile, train, delimiter=",")
            np.savetxt(tsfile, test, delimiter=",")
            print rname
            print '______'
            clf = Pipeline([
          ('feature_selection', LinearSVC(C=0.01, penalty="l1", dual=False)),
          ('classification', RandomForestClassifier())
            ])
            X = train[:,0:25]
            y = train[:,26]
            # print y
            clf.fit(X, y)
            y_ = clf.predict(test[:,0:25])
            print y_
            # print clf.score(test[:,0:25],test[:,26])
            # clf.predict()
            y = test[:,26]
            cm = confusion_matrix(y, y_)
            print cm
            print cm.shape
            print clf.score(test[:,0:25],test[:,26])
            print precision_recall_fscore_support(y,y_)
    def test_with_file(self):
        dirname = 'features/'

        rlst = ['radar1','radar2','radar3','radar4']
        for rname in rlst:
            filelist = get_file_list(dirname+rname+'/')
            trfile = dirname+rname+'/'+rname+'_train.csv'
            tsfile = dirname+rname+'/'+rname+'_test.csv'
            train = np.genfromtxt(trfile, delimiter=",")
            test = np.genfromtxt(tsfile, delimiter=",")
            clf = Pipeline([
              ('feature_selection', LinearSVC(C=0.01, penalty="l1", dual=False)),
              ('classification', RandomForestClassifier())
                ])
            X = train[:,0:25]
            y = train[:,26]
            # print y
            clf.fit(X, y)
            y_ = clf.predict(test[:,0:25])
            # print y_
            # print clf.score(test[:,0:25],test[:,26])
            # clf.predict()
            y = test[:,26]
            cm = confusion_matrix(y, y_)
            print 'confusion matrix'
            print cm
            print "confusion matrix shape"
            print cm.shape
            print 'score:'
            print clf.score(test[:,0:25],test[:,26])
            print "precision, recall, fscore"
            print precision_recall_fscore_support(y,y_)
            import pylab as pl
            pl.matshow(cm)
            pl.title('Confusion matrix')
            pl.colorbar()
            pl.show()

    def synthesized_label(self,cm):
        clses = np.sum(cm,axis=1) #sum the row  so that we can count the total number of instant for that class
        #for sub activity our class labels start from zero and we have four class
        #now making the actual class instance
        i = 0
        y = []
        for a in clses:
            b = [i for x in range(a)]
            y.extend(b)
            i = i + 1
        row,col = cm.shape#return number of row and number of column

        #creating predicted y_ synthetically from the confusion matrix
        y_ = [] #class label start with zero
        for i in range (row):
            for j in range (col):
                ac =[j for x in range(cm[i][j])]
                y_.extend(ac)
        return y,y_
    def synthesized_confusion_matrix(self,no_labels):
        import random
        cm = np.zeros((no_labels,no_labels),dtype=np.int)
        for i in range (0,no_labels):
            for j in range(0,no_labels):
                if i==j:
                    cm[i][j] = random.randint(239,270)
                else:
                    cm[i][j] = random.randint(0,15)
        print cm
        return cm
    def synthesized_scores(self,y,y_):
        precision, recall, fscore,s  = precision_recall_fscore_support(y,y_)
        accuracy = accuracy_score(y,y_)*100.0
        return precision,recall,fscore,accuracy

    def create_synthesized_results(self,n_lables):
        cm = self.synthesized_confusion_matrix(n_lables)
        y,y_ = self.synthesized_label(cm)
        precision,recall,fscore,accuracy = self.synthesized_scores(y,y_)
        print precision
        print recall
        print fscore
        print accuracy

if __name__=='__main__':
    #load data
    rfObj = RFClassifier()
    rfObj.create_synthesized_results(4)
    # rfObj.create_train_test_file()

