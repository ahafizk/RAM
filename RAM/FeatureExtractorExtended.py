__author__ = 'prune'
import sys
import numpy as np
from utility import *
import os
from os import mkdir
from os.path import exists, abspath
from math import log
from  re import search
from subactivities import *
from Filter.datafilter import *
#def get_file_list(mypath):
    #return [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]

class FeatureExtractor:

    
    def __init__(self):
        self.frame_length = 2000
        self.frame_step = 1000
        self.frequency = 200

    def set_frameslength(self,flength,step):
        self.frame_length = flength
        self.frame_step = step

    def get_velocity(self,data):
        '''this function calculate dtheta/dt which means the velocity of the object which moves
           infront of the radar.
        '''
        delta_s = np.fabs(np.diff(data))
        v = delta_s * self.frequency
        return v

    def get_range(self,data):
        min_v = np.min(data)
        max_v = np.max(data)
        rangev = np.fabs(max_v - min_v)
        return rangev

    def get_phase(self,data1,data2):
        """
        calculate phase from data1 and data2 and unwrap the phase [pi/2 to - pi/2]
        """
        val = data1/data2
        phase = np.arctan(val)
        unwrap_phase = np.unwrap(phase)
        return unwrap_phase

    # def get_velocity(self,data,dt):
    #     '''this function calculate dtheta/dt which means the velocity of the object which moves
    #        infront of the radar.
    #     '''
    #     delta_s = np.diff(data)
    #     # print delta_s
    #     v = delta_s * dt
    #     return v

    def mad(self,arr):
        """
            https://en.wikipedia.org/wiki/Median_absolute_deviation
        """
        arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
        med = np.median(arr)
        return np.median(np.abs(arr - med))

    def get_acceleration(self,v):
        '''
        Acceleration (a) = Change in Velocity (dv) / Time Interval (dt) = (v2 - v1) / (t1 - t2)
        dt = 1/fs
        acceleration, a = dv/dt = dv / (1/fs) = dv*fs
        '''

        dv = np.fabs(np.diff(v))
        a = dv * self.frequency
        return a

    def get_entropy(self,arr):
        '''
        calculate entropy of an array
        '''
        disct_arr = np.unique(arr)

        n_labels = len(disct_arr)

        if n_labels <= 1:
            return 0

        counts = np.bincount(arr)
        probs = counts / n_labels
        n_classes = np.count_nonzero(probs)

        if n_classes <= 1:
            return 0

        ent = 0.

        # Compute standard entropy.
        for i in probs:
            ent -= i * log(i, base=n_classes)
        return ent

    def spectral_centroid(wavedata, window_size, sample_rate):

        magnitude_spectrum = stft(wavedata, window_size)

        timebins, freqbins = np.shape(magnitude_spectrum)

        # when do these blocks begin (time in seconds)?
        timestamps = (np.arange(0,timebins - 1) * (timebins / float(sample_rate)))

        sc = []

        for t in range(timebins-1):

            power_spectrum = np.abs(magnitude_spectrum[t])**2

            sc_t = np.sum(power_spectrum * np.arange(1,freqbins+1)) / np.sum(power_spectrum)

            sc.append(sc_t)


        sc = np.asarray(sc)
        sc = np.nan_to_num(sc)

        return sc, np.asarray(timestamps)

    def get_max_energy_fft(self,data):
        p = (np.abs(np.fft.rfft(data)))
        e = np.sum(np.square(p))/len(data)
        max_mag = np.max(p)
        return max_mag,e

    def get_featurelist_helper(self, I, Q):
        
        filter_obj = LowPassFilter()
        I = filter_obj.lowpass_filter(I, 200, 10)
        Q = filter_obj.lowpass_filter(Q, 200, 10)

        I_frames = framesig(I,self.frame_length,self.frame_step)
        Q_frames = framesig(Q,self.frame_length,self.frame_step)
        phi = np.unwrap(np.arctan(Q/I)) #unwrap the phase values
        phi_frames = framesig(phi,self.frame_length,self.frame_step) #making phase frames

        # print(phi_frames)
        # print (phi_frames.shape)
        # print (I_frames.shape)
        # print(Q_frames.shape)
        
        row,col = I_frames.shape
        flist = []
        for i in range(row):
            I1 = I_frames[i,:]
            Q1 = Q_frames[i,:]
            phi1 = phi_frames[i,:]
            v = self.get_velocity(phi1) # get velocity
            a = self.get_acceleration(v)

            # fmag_I,fe_I = self.get_max_energy_fft(I)
            # fmag_Q,fe_Q = self.get_max_energy_fft(I)

            #calculate  range
            I_range = self.get_range(I1)
            Q_range = self.get_range(Q1)
            phi_range = self.get_range(phi1)
            v_range = self.get_range(v)
            a_range = self.get_range(a)

            #calculate mean
            I_avg = np.average(I1)
            Q_avg = np.average(Q1)
            phi_avg = np.average(phi1)
            v_avg = np.average(v)
            a_avg = np.average(a)

            #calculate average energy
            i_2 = np.square(I1)
            q_2 = np.square(Q1)
            E = np.average(i_2+q_2) # energy e = i*i + q*q

            #calculate  standard deviation
            i_std = np.std(I1)
            q_std = np.std(Q1)
            phi_std = np.std(phi1)
            v_std = np.std(v)
            a_std = np.std(a)

            #calculate variance
            i_var = np.var(I1)
            q_var = np.var(Q1)
            phi_var = np.var(phi1)
            v_var = np.var(v)
            a_var = np.var(a)

            #calculate median absolute difference
            i_mad = self.mad(I1)
            q_mad = self.mad(Q1)
            phi_mad = self.mad(phi1)
            v_mad = self.mad(v)
            a_mad = self.mad(a)

            #calculate entropy
            # ent = self.get_entropy(phi1)

            #fl = [I_range,I_avg,i_std,i_var,i_mad,Q_range,Q_avg,q_std,q_var,q_mad,E,phi_range,phi_avg,phi_std,phi_var,phi_mad,v_range,v_avg,
                #v_std,v_var,v_mad,a_avg,a_range,a_std,a_var,a_mad, self.cls]
            
            fl = [I_range,
                I_avg,
                i_std,
                i_var,
                i_mad,
                Q_range,
                Q_avg,
                q_std,
                q_var,
                q_mad,
                E,
                # fmag_I,
                # fe_I,
                # fmag_Q,
                # fe_Q,
                phi_range,
                phi_avg,
                phi_std,
                phi_var,
                phi_mad,
                v_range,
                v_avg,
                v_std,
                v_var,
                v_mad,
                a_avg,
                a_range,
                a_std,
                a_var,
                a_mad,
                self.cls]

            flist+=[fl]

        # d = Q_frames[0,:]
        # print d
        d = np.array(flist)
        # print "d.shape",d.shape
        return d


    def get_featurelist(self,file,index1=0,index2=1):
        #print file
        data = np.genfromtxt(file, delimiter=',')
        I = data[:,index1]
        Q = data[:,index2]

        return self.get_featurelist_helper(I,Q)


    def get_featurelist_from_nparray(self, data, index1=0, index2=1):

       I = data[:,index1]
       Q = data[:,index2]
       return self.get_featurelist_helper(I,Q)


    def get_velocity_from_frequency(self,data,f):
        '''this function calculate dtheta/dt which means the velocity of the object which moves
           infront of the radar.
        '''
        delta_s = np.diff(data)
        v = delta_s * f
        return v

    def make_csv(self,dirname):
        import re
        import csv
        dir_data = dirname
        filelist = get_file_list(dir_data)
        for name in filelist:
            data = open(dir_data+name).read()
            lines_of_data = data.splitlines()
            # data.replace('"',' ')
            # print data
            print name
            target = open(dir_data+'csv/'+name.split('.')[0]+'.csv', 'w')
            for line in lines_of_data:
                if len(line)>2:
                    # print line
                    ln = line.split('"')
                    l1 = ln[0]
                    l2 = ln[1]
                    l = l1+l2
                    # print l
                    # target.write(line.split('"')[1]) #commented for considering timestamps
                    target.write(l)
                    target.write('\n')
            target.close()


    def make_gesture_csv(self,dirname='Data/gestures/'):
        # dirname = 'Data/gestures/'
        self.make_csv(dirname)

    def make_activity_csv(self,dirname='Data/activity/'):
        # dirname = 'Data/activity/'
        self.make_csv(dirname)


    def set_class(self,cls):
        self.cls = cls

    def get_radar_index(self,rad_no):
        index1 = 1
        index2 = 2
        if rad_no==1:
            index1= 1
            index = 2
        elif rad_no==2:
            index1 = 3
            index2 = 4
        elif rad_no ==3:
            index1 = 5
            index=6
        elif rad_no==4:
            index1 = 7
            index2 = 8
        return index1,index2

    def make_fustion_data(self,lines_of_data,file,index1,index2,index3,index4):
        target = open(file, 'w')
        for line in lines_of_data:
            l = line.split(",")
            ml = l[index1]+','+l[index2]
            target.write(ml)
            target.write('\n')
            ml = l[index3]+','+l[index4]
            target.write(ml)
            target.write('\n')
        target.close()

    def make_all_radar_fusion(self,lines_of_data,file):
        target = open(file, 'w')

        for line in lines_of_data:
            l = line.split(",")
            index = 1
            # print len(l)
            ml = l[index]+','+l[index+1]
            target.write(ml)
            target.write('\n')
            index = index+2
            ml = l[index]+','+l[index+1]
            target.write(ml)
            target.write('\n')
            index = index+2
            ml = l[index]+','+l[index+1]
            target.write(ml)
            target.write('\n')
            index = index+2
            ml = l[index]+','+l[index+1]
            target.write(ml)
            target.write('\n')
        target.close()

    def data_fusion(self,dir_data):

        csv_dir = dir_data + '/csv'
        filelist = get_file_list(csv_dir + "/")
        for name in filelist:
            data = open(csv_dir + "/"+name).read()
            lines_of_data = data.splitlines()

            #creating fusion 1
            radar1=1
            radar2=3
            print '\n Create data fusion with radar %d, radar %d\n'%(radar1,radar2)
            fusion_dir1 = dir_data+'/fusion1'
            if not os.path.exists(fusion_dir1):
                os.mkdir(fusion_dir1)
            filename = fusion_dir1+'/'+name
            index1,index2 = self.get_radar_index(radar1) #radar 1
            index3,index4 = self.get_radar_index(radar2) # radar 2
            self.make_fustion_data(lines_of_data,filename,index1,index2,index3,index4)

            # #fuse data for radar 2 and radar 4
            # radar1=3
            # radar2=4
            # print '\n Create data fusion with radar %d, radar %d'%(radar1,radar2)
            # fusion_dir2 = dir_data+'/fusion2'
            # if not os.path.exists(fusion_dir2):
            #     os.mkdir(fusion_dir2)
            # filename = fusion_dir2+'/'+name
            # index1,index2 = self.get_radar_index(radar1) #radar 1
            # index3,index4 = self.get_radar_index(radar2) # radar 2
            # self.make_fustion_data(lines_of_data,filename,index1,index2,index3,index4)


            # print '\n Create data fusion with all four radar'
            # fusion_dir3 = dir_data+'/fusion3'
            # if not os.path.exists(fusion_dir3):
            #     os.mkdir(fusion_dir3)
            # filename = fusion_dir3+'/'+name
            # self.make_all_radar_fusion(lines_of_data,filename)


    def extract_fusion_data_features(self,dir_data,feature_dir,fusion_name):
        filelist = get_file_list(dir_data + "/")
        for name in filelist:
            print "Processing file: ", name
            file = dir_data + "/" + name
            # print name
            #self.set_class(self.get_subacitivity_class(name))
            sub_class = get_subactivity_class(name)


            if sub_class == -1:
                print "Skipping subactivity: ", name
                continue

            self.set_class(sub_class)

            #read radar 1 data

            if not os.path.exists(feature_dir+ '/'+fusion_name):
                os.mkdir(feature_dir+ '/'+fusion_name)
            fet = self.get_featurelist(file,0,1)

            np.savetxt(feature_dir + '/'+fusion_name+'/'+ name, fet, delimiter=",")

    def main(self):

        if not len(sys.argv) == 3:
            print "\n Usage: python FeatureExtractor.py <data_directory> <train | test>"
            return


        dir_data = sys.argv[1].rstrip('/')

        # Make CSV
        csv_dir = dir_data + '/csv'
        if not os.path.exists(csv_dir):
            os.mkdir(csv_dir)

        print "\n Creating CSV files in directory", csv_dir
        self.make_csv(abspath(dir_data) + "/") 
        print "\n CSV generation complete...\n"

        print '\n Start Data Fusion....\n'
        self.data_fusion(dir_data)
        print '\n Data Fusion Complete..\n'

        # Create radar feature directories
        print "\n Creating feature directories for 4 radars \n"
        feature_dir = dir_data + '/features'
        if not os.path.exists(feature_dir):
            os.mkdir(feature_dir)
            os.mkdir(feature_dir + "/radar1")
            os.mkdir(feature_dir + "/radar2")
            os.mkdir(feature_dir + "/radar3")
            os.mkdir(feature_dir + "/radar4")


            os.mkdir(feature_dir + "/fusion1")
            os.mkdir(feature_dir + "/fusion2")
            os.mkdir(feature_dir + "/fusion3")


        print " Directory creation complete\n"


        print "\n Extracting Features ...\n"

        #Extract features and store them
        filelist = get_file_list(csv_dir + "/")
        for name in filelist:
            print "Processing file: ", name
            file = csv_dir + "/" + name
            # print name
            #self.set_class(self.get_subacitivity_class(name))
            sub_class = get_subactivity_class(name)


            if sub_class == -1:
                print "Skipping subactivity: ", name
                continue

            self.set_class(sub_class)
            
            #read radar 1 data
            fet = self.get_featurelist(file,1,2)
            np.savetxt(feature_dir + '/radar1/'+ name, fet, delimiter=",")

            #read radar 2 data
            fet = self.get_featurelist(file,3,4)
            np.savetxt(feature_dir + '/radar2/'+ name, fet, delimiter=",")

            #read radar 3 data
            fet = self.get_featurelist(file,5,6)
            np.savetxt(feature_dir + '/radar3/'+ name, fet, delimiter=",")

            #read radar 4 data
            fet = self.get_featurelist(file,7,8)
            np.savetxt(feature_dir + '/radar4/'+ name, fet, delimiter=",")

        print "\n Feature extraction complete. Features stored in directory: ", feature_dir 



        print '\n Creating Fusion Data features\n'
        fusion_dir1 = dir_data+'/fusion1'
        self.extract_fusion_data_features(fusion_dir1,feature_dir,'fusion1')
        # fusion_dir2 = dir_data+'/fusion2'
        # self.extract_fusion_data_features(fusion_dir2,feature_dir,'fusion2')
        # fusion_dir3 = dir_data+'/fusion3'
        # self.extract_fusion_data_features(fusion_dir3,feature_dir,'fusion3')
        print '\n Feature extraction complete for fusion data\n'


        # Combine features per radar and store them in Results directory
        print "Combining feature files by radar..."
        results_dir = "Results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Loop for all 4 radars
        for i in range(4):
            r = "radar" + str(i+1)
            filename = results_dir + "/" + r + "_" + sys.argv[2] + ".csv"  
            d = feature_dir + "/" + r + "/" 
            filelist = get_file_list(d)
            
            fp = open(filename, 'w')
            for f in filelist:
                print "f:",d,f
                fp2 = open(d + f, 'r')
                fp.write(fp2.read())
                fp2.close()
            fp.close()

        print '\n Combining feature files by for fusion data\n'
        results_dir = "Results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        if not os.path.exists(results_dir+'/fusion'):
            os.makedirs(results_dir+'/fusion')
        # if not os.path.exists(results_dir+'/fusion2'):
        #     os.makedirs(results_dir+'/fusion2')

        # Loop for all 2 fusion data
        for i in range(1):
            r = "radar" + str(i+1)
            fl = 'fusion'
            filename = results_dir + "/" +fl+'/'+ r + "_" + sys.argv[2] + ".csv"
            d = feature_dir + "/" + fl+str(i+1) + "/"
            filelist = get_file_list(d)

            fp = open(filename, 'w')
            for f in filelist:
                print "f:",d,f
                fp2 = open(d + f, 'r')
                fp.write(fp2.read())
                fp2.close()
            fp.close()

        print '\n\nCombining feature files complete!!!\n\n'

        subactivities = get_subactivities()
        print "\n Feature labels"
        for x in subactivities:
            print get_subactivity_class(x), "\t", x

        print "Exit FeatureExtractor.py"



if __name__ == '__main__':
    fobj = FeatureExtractor()
    fobj.main()
    # fobj.gen_activity_features()
