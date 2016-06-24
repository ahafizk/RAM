import sys
from ARClassifier import *
from FeatureExtractor import *
import numpy as np
import socket
import pickle
from subactivities import *


HOST = ''
PORT = 1234

HISTORY_LEN = 8
history = [[],[],[],[]]
history_labels = [[],[],[],[]]

def most_common(lst):
    return max(set(lst), key=lst.count)


def main():

    print "Loading previously stored HMM..."
    hmm = get_stored_hmm()
    #hmm = getHMM()
    print hmm.A
    print
    print hmm.B
    print hmm.pi
    print "HMM Created successfully"

    print "Loading previously stored RF model"
    RF = joblib.load('RFModel.pk')
    print "RF loaded successfully"
    
    # Feature Extractor
    FE = FeatureExtractor()
    FE.set_class(1)

    # Open socket and start listening for data
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(1)

    subactivities = get_subactivities()
    activities = get_activities()

    conn, addr = s.accept()
    while True:
        data = conn.recv(20000)
        #print data
        if data:
            #print data
            data = data.split('\n')

            # Remove extra newline
            del data[-1]

            csv_data = []
            for d in data:
                #print len(d)
                if len(d) >= 53:
                    d2 = d[14:].strip('"')
                    d3 = np.fromstring(d2, dtype=int, sep=',')
                    csv_data.append(d3.tolist()) 
            
            #print len(csv_data)
            if len(csv_data) > 0:
                csv_data = np.array(csv_data)
                #row,col = csv_data.shape
                #print row, col
                
                features = []
                for x in range(4):
                    f = FE.get_featurelist_from_nparray(csv_data, 2*x, 2*x + 1)
                    for y in f:
                        p = RF.predict(y[:25])[0]
                        if len(history[x]) >= HISTORY_LEN:
                            history[x].pop(0)
                            history_labels[x].pop(0)
                        history[x].append(p)
                        history_labels[x].append(subactivities[int(p)])
                
                for sub in subactivities:
                    print sub, get_subactivity_class(sub)
                

                for z in range(4):
                    hmm.mapB(history[z]) 
                    alpha = hmm.calculate_alpha(history[z])
                    #alpha_normalized = alpha
                    alpha_normalized = alpha.astype('float') / alpha.sum(axis=1)[:, np.newaxis]
                    
                    c = most_common(history[z])
                    
                    if len(history) > 12:
                        print z+1, history_labels[z][12:], subactivities[int(c)] 
                    else:
                        print z+1, history_labels[z][5:], subactivities[int(c)] 
                    print alpha_normalized[-1]
                    print

                print "\n--------------\n"
                

main()
