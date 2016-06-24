__author__ = 'hafiz'
import numpy as np
from FeatureExtractor import *
from drawfigure import *
from Filter.datafilter import *
feobj = FeatureExtractor()

def get_actual_velocity(frames):
    v = (50* frames * 3)/(3.14*58)
    return v

def check_fft(y):
    import matplotlib.pyplot as plt
    t = np.arange(len(y))
    sp = np.fft.fft(y)
    freq = np.fft.fftfreq(t.shape[-1])
    plt.plot(freq, sp.real, freq, sp.imag)

    plt.show()

def check_fft1(y):

    from pylab import plot, show, title, xlabel, ylabel, subplot
    from scipy import fft, arange


    import numpy as np
    import pylab as pl
    rate = 200.0
    t = np.arange(0, len(y), 1/rate)
    x = np.sin(2*np.pi*4*t) + np.sin(2*np.pi*7*t) + np.random.randn(len(t))*0.2
    # x = y
    p = 20*np.log10(np.abs(np.fft.rfft(x)))
    # p = (np.abs(np.fft.rfft(x)))
    f = np.fft.fftfreq(y.size,1/rate)
    # print f
    e = np.sum(np.square(p))/len(y)
    print 'energy: '
    print e
    print 'maximum: '
    print np.max(p)
    print 'mean: '
    print np.mean(p)
    r = np.max(p) - np.min(p)
    print 'range: '
    print r
    # f = np.linspace(0, rate/2, len(p))
    plot(f, p)

    # plotSpectrum(y,Fs)
    show()
    print '\n\n'
def make_frames_features(I,Q):
    # I_frames = framesig(I,feobj.frame_length,feobj.frame_step)
    # Q_frames = framesig(Q,feobj.frame_length,feobj.frame_step)
    phi = np.unwrap(np.arctan(Q/I)) #unwrap the phase values
    # phi_frames = framesig(phi,feobj.frame_length,feobj.frame_step) #making phase frames
    v = get_actual_velocity(phi)
    return v

if __name__=='__main__':
    csv_dir = 'Data/collection7-8/train/csv/'
    filelist = get_file_list(csv_dir)
    lst = []
    lst2 = []
    filter = LowPassFilter()
    for fname in filelist:
        print fname
        file = csv_dir+fname
        data = np.genfromtxt(file, delimiter=',')
        # print 'Radar 1'
        check_fft1(data[:,1])

        I = filter.lowpass_filter(data[:,0],cutoff=10,fs=200)
        Q = filter.lowpass_filter(data[:,1],cutoff=10,fs=200)
        # check_fft(I[0:800])
        #
        # phi = np.unwrap(np.arctan(Q/I))
        #
        # v = make_frames_features(I,Q)
        # v1=feobj.get_velocity(phi)
        # lst+=[[fname, np.min(v),np.max(v),'Radar1']]
        # lst2+=[[fname,np.min(v1),np.max(v1),'Radar1']]

if __name__=='__main1__':
    csv_dir = 'Data/collection7-8/test/train/csv/'
    # csv_dir = 'Data/Collection7-16/test1/train/csv/'
    filelist = get_file_list(csv_dir)
    lst = []
    lst2 = []
    filter = LowPassFilter()
    for fname in filelist:
        print fname
        file = csv_dir+fname
        data = np.genfromtxt(file, delimiter=',')
        # print 'Radar 1'
        I = filter.lowpass_filter(data[:,0],cutoff=10,fs=200)
        Q = filter.lowpass_filter(data[:,1],cutoff=10,fs=200)
        phi = np.unwrap(np.arctan(Q/I))

        v = make_frames_features(I,Q)
        v1=feobj.get_velocity(phi)
        lst+=[[fname, np.min(v),np.max(v),'Radar1']]
        lst2+=[[fname,np.min(v1),np.max(v1),'Radar1']]
        # print "min=%f,max=%f"%(np.min(v),np.max(v))
        dobj = DrawFigure(1,1)
        dobj.add_figure(phi[200:600],1,title=fname+'(velocity radar1)',xlabel='samples',ylabel='amplitude')
        # dobj.show_figure()
        # print 'Radar 2'

        # I = filter.lowpass_filter(data[:,2],cutoff=99,fs=200)
        # Q = filter.lowpass_filter(data[:,3],cutoff=99,fs=200)
        # phi = np.unwrap(np.arctan(Q/I))
        # v = make_frames_features(I,Q)
        # v1=feobj.get_velocity(phi)
        # # dobj = DrawFigure(1,1)
        # dobj.add_figure(v1[200:600],2,title=fname+'(velocity radar2)',xlabel='samples',ylabel='amplitude')
        # dobj.show_figure()
        # print v1
        # print "min=%f,max=%f"%(np.min(v),np.max(v))
        # lst+=[[fname, np.min(v),np.max(v),'Radar2']]
        # lst2+=[[fname,np.min(v1),np.max(v1),'Radar2']]
        # # print 'Radar 3'
        # I = filter.lowpass_filter(data[:,4],cutoff=99,fs=200)
        # Q = filter.lowpass_filter(data[:,5],cutoff=99,fs=200)
        # phi = np.unwrap(np.arctan(Q/I))
        # v = make_frames_features(I,Q)
        # v1=feobj.get_velocity(phi)
        # # dobj = DrawFigure(1,1)
        # dobj.add_figure(v1[200:600],3,title=fname+'(velocity radar3)',xlabel='samples',ylabel='amplitude')
        # # dobj.show_figure()
        # # print "min=%f,max=%f"%(np.min(v),np.max(v))
        # lst+=[[fname, np.min(v),np.max(v),'Radar3']]
        # lst2+=[[fname,np.min(v1),np.max(v1),'Radar3']]
        # # print 'Radar 4'
        # I = filter.lowpass_filter(data[:,6],cutoff=99,fs=200)
        # Q = filter.lowpass_filter(data[:,7],cutoff=99,fs=200)
        # phi = np.unwrap(np.arctan(Q/I))
        # v = make_frames_features(I,Q)
        # v1=feobj.get_velocity(phi)
        # # dobj2 = DrawFigure(1,1)
        # dobj.add_figure(v1[200:600],4,title=fname+'(velocity radar4)',xlabel='samples',ylabel='amplitude')
        # # dobj2.show_figure()
        # # print "min=%f,max=%f"%(np.min(v),np.max(v))
        # lst+=[[fname, np.min(v),np.max(v),'Radar4']]
        # lst2+=[[fname,np.min(v1),np.max(v1),'Radar4']]
        dobj.show_figure()
    # aa = np.array_str(lst)
    # print aa.shape
    # print lst
    # with open('velocity.csv','a') as f:
    #     for x in lst:
    #         print x
    #         f.write(x)
    #         f.write('\n')
    #         # np.savetxt(f,x,fmt='%s, %f, %f, %s\n',delimiter=',')
    #         # np.savetxt(f,x,fmt='%r',delimiter=', ')
    # np.savetxt('velocity.csv',aa,delimiter=',')
    # a = np.vstack(lst)
    va = np.vstack(lst2)
    print va
    # np.savetxt("velocity.csv", a, delimiter=",", fmt='%s,%f,%f,%s')
        # cls = get_subactivity_class(file)
        # print "activity name::%s, class=%d"%(file,cls)
