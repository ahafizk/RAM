__author__ = 'hafiz'
import sys
import os
from os import mkdir
from os.path import exists, abspath
from os.path import isfile, join
from os import listdir
import numpy as np
from Filter.datafilter import *
def get_file_list(dir_name):
        '''
        get_file_list returns name of the files of a directory
        '''
        files = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]
        return files
def main():

    if not len(sys.argv) == 3:
        print "\n Usage: python FeatureExtractor.py <data_directory> <train | test>"
        return


    dir_data = sys.argv[1].rstrip('/')
    sav_dir = sys.argv[2].rstrip('/')
    filelist = get_file_list(dir_data + "/")
    for name in filelist:
        file = dir_data+'/'+name
        data = np.genfromtxt(file, delimiter=',')
        row,col = data.shape
        fdata = np.zeros((row,col-1),dtype=float) #make the final array

        I = data[:,1]
        Q = data[:,2]
        filter_obj = LowPassFilter()
        I = filter_obj.lowpass_filter(I, 200, 10)
        Q = filter_obj.lowpass_filter(Q, 200, 10)
        index = 0
        fdata[:,index]=I
        index=index+1
        fdata[:,index]=Q


        I = data[:,3]
        Q = data[:,4]
        filter_obj = LowPassFilter()
        I = filter_obj.lowpass_filter(I, 200, 10)
        Q = filter_obj.lowpass_filter(Q, 200, 10)
        index=index+1
        fdata[:,index]=I
        index=index+1
        fdata[:,index]=Q

        I = data[:,5]
        Q = data[:,6]
        filter_obj = LowPassFilter()
        I = filter_obj.lowpass_filter(I, 200, 10)
        Q = filter_obj.lowpass_filter(Q, 200, 10)
        index=index+1
        fdata[:,index]=I
        index=index+1
        fdata[:,index]=Q


        I = data[:,7]
        Q = data[:,8]
        filter_obj = LowPassFilter()
        I = filter_obj.lowpass_filter(I, 200, 10)
        Q = filter_obj.lowpass_filter(Q, 200, 10)
        index=index+1
        fdata[:,index]=I
        index=index+1
        fdata[:,index]=Q
        np.savetxt(sav_dir+'/'+ name, fdata, delimiter=",")



if __name__=='__main__':
    main()