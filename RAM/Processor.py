__author__ = 'prune'
import numpy as np
from os.path import isfile, join
from os import listdir
import math

from sklearn import cross_validation

from drawfigure import *
from FeatureProcessor.FeatureExtractor import *


class Processor:
    def __init__(self):
        print "processor initialized"

    def get_file_list(self, dir_name):
        '''
        get_file_list returns name of the files of a directory
        '''
        files = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]
        return files

    def process(self):
        filename = 'subjectI/walking_backward_4.csv'
        # filename ='subjectII/walking_backward_4.csv'
        # filename ='subjectI/walking_forward_4.csv'
        # filename ='subjectII/walking_forward_4.csv'
        # filename ='subjectI/standing_2m_4.csv'
        # filename ='subjectII/standing_2m_4.csv'
        # filename ='subjectI/sitting_1m_4.csv'
        # filename ='subjectII/sitting_1m_4.csv'
        # show_Specgram(filename,0,1)
        # show_Specgram(filename,2,3)
        # calculate_phase(filename,2,3)
        # show_Specgram(filename,4,5)
        range1 = 300
        range2 = 700
        data = np.genfromtxt(filename, delimiter=',')
        I = np.trim_zeros(np.array(data[2]))  #remove trailing zeros
        Q = np.trim_zeros(np.array(data[3]))  #remove trailing zeros

        f = FeatureExtractor()
        angle = np.fabs(f.get_phase(I, Q))
        print angle.shape
        print (angle.T).shape
        row = angle.shape
        print row
        # np.savetxt("foo.csv", angle.T, delimiter=",")
        # dt = 1/330
        # v = f.get_velocity(angle, 330)
        # print(v)
        # fig = pylab.figure()
        # ax4 = fig.add_subplot(111)
        # ax4.plot(angle)
        # pylab.show()
        d = DrawFigure(1,1)
        d.add_figure(angle,1)
        d.show_figure()

    def framesig(self, sig, frame_len, frame_step):
        """Frame a signal into overlapping frames.

        :param sig: the audio signal to frame.
        :param frame_len: length of each frame measured in samples.
        :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
        :param winfunc: the analysis window to apply to each frame. By default no window is applied.
        :returns: an array of frames. Size is NUMFRAMES by frame_len.
        """
        slen = int(len(sig))
        frame_len = round(frame_len)
        frame_step = round(frame_step)
        if slen <= frame_len:
            numframes = 1
        else:
            numframes = 1 + math.ceil((1.0 * slen - frame_len) / frame_step)

        padlen = int((numframes - 1) * frame_step + frame_len)

        zeros = np.zeros((padlen - slen,))
        padsignal = np.concatenate((sig, zeros))

        indices = np.tile(np.arange(0, frame_len), (numframes, 1)) + np.tile(
            np.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
        indices = np.array(indices, dtype=np.int32)
        frames = padsignal[indices]
        # win = np.tile(winfunc(frame_len),(numframes,1))
        return frames
    def calculate_angles(self):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        dir_list = ['subjectI', 'subjectII']
        f = FeatureExtractor()
        for dir_name in dir_list:
            file_list = self.get_file_list(dir_name)
            pp = PdfPages(dir_name+'_velocity'+'.pdf')
            pp1 = PdfPages(dir_name+'_abs_velocity'+'.pdf')
            pp2 = PdfPages(dir_name+'_abs_angle'+'.pdf')
            pp3 = PdfPages(dir_name+'_abs_velocity_angle'+'.pdf')
            for file in file_list:
                filename = dir_name + '/' + file
                print (filename)
                data = np.genfromtxt(filename, delimiter=',')
                I = np.trim_zeros(np.array(data[2]))  #remove trailing zeros
                Q = np.trim_zeros(np.array(data[3]))  #remove trailing zeros

                angle = f.get_phase(I, Q)
                velocity = f.get_velocity(angle, 330)

                # print angle.shape
                # print (angle.T).shape
                save_ang_file = 'phi/' + dir_name + '/' + file
                np.savetxt(save_ang_file, angle, delimiter=",")

                save_v_file = 'velocity/' + dir_name + '/' + file
                np.savetxt(save_v_file, velocity, delimiter=",")

                plt.plot(velocity ,'b')
                nam = file.split('.')
                plt.title(dir_name+' '+nam[0])
                plt.xlabel('Number of Samples')
                plt.ylabel('Value')
                plt.savefig(pp, format='pdf')
                # plt.show()
                plt.close()

                plt.plot(np.fabs(velocity) ,'g')
                nam = file.split('.')
                plt.title(dir_name+' '+nam[0])
                plt.xlabel('Number of Samples')
                plt.ylabel('Value')
                plt.savefig(pp1, format='pdf')
                # plt.show()
                plt.close()

                plt.plot(np.fabs(angle) ,'r')
                nam = file.split('.')
                plt.title(dir_name+' '+nam[0])
                plt.xlabel('Number of Samples')
                plt.ylabel('Value')
                plt.savefig(pp2, format='pdf')
                # plt.show()
                plt.close()

                plt.plot(np.fabs(velocity) ,'g')
                plt.plot(np.fabs(angle) ,'r')
                nam = file.split('.')
                plt.title(dir_name+' '+nam[0])
                plt.xlabel('Number of Samples')
                plt.ylabel('Value')
                plt.savefig(pp3, format='pdf')
                # plt.show()
                plt.close()
                angle = None
            pp.close()
            pp1.close()
            pp2.close()
            pp3.close()

    def gen_features(self):
        # print ''

        dir_list = ['subjectI', 'subjectII']
        for dir_name in dir_list:
            file_list = self.get_file_list(dir_name)

            for file in file_list:
                filename = dir_name + '/' + file
                data = np.genfromtxt(filename, delimiter=',')
                I = np.trim_zeros(np.array(data[2]))  #remove trailing zeros
                Q = np.trim_zeros(np.array(data[3]))  #remove trailing zeros
                f = FeatureExtractor()
                angle = f.get_phase(I, Q)
                velocity = f.get_velocity(angle, 330)
                # print angle.shape
                # print (angle.T).shape
                save_ang_file = 'phi/' + dir_name + '/' + file
                np.savetxt(save_ang_file, angle, delimiter=",")
                save_v_file = 'velocity/' + dir_name + '/' + file
                np.savetxt(save_v_file, velocity, delimiter=",")


    def get_class(self, str):
        ret = -1
        if ('sitting' in str):
            ret = 1
        elif ('standing_1m' in str):
            ret = 2
        elif ('standing_2m' in str):
            ret = 3
        elif ('walking_backward' in str):
            ret = 4
        elif ('walking_forward' in str):
            ret = 5
        return ret

    def gen_feature_vector(self):
        dir_list = ['subjectI', 'subjectII']
        for dir_name in dir_list:
            file_list = self.get_file_list(dir_name)
            for file in file_list:
                save_ang_file = 'phi/' + dir_name + '/' + file
                save_v_file = 'velocity/' + dir_name + '/' + file
                angle = np.fabs(np.genfromtxt(save_ang_file, delimiter=','))
                velocity = np.fabs(np.genfromtxt(save_v_file, delimiter=','))
                lenth = len(angle)
                angle = angle[0:lenth - 1]
                angle_frames = self.framesig(angle, 990, 990)
                v_frames = self.framesig(velocity, 990, 990)
                mean_angles_list = np.mean(angle_frames, 1).tolist()
                mean_angles = np.array(mean_angles_list, ndmin=2)
                mean_v_list = np.mean(v_frames, 1).tolist()
                mean_v = np.array(mean_v_list, ndmin=2)
                # print mean_v.shape
                fet = np.append(mean_angles, mean_v, 0).T
                row, col = fet.shape
                fet = np.append(fet, np.zeros((row, 1), dtype=float), axis=1)  # now it has dimension row * (col+1)
                print file
                print 'class = ' + str(self.get_class(file))
                fet[:, 2] = self.get_class(file)  # class value assinged to the third column
                # print fet.shape
                # print fet
                save_dir = 'feature_vector/' + dir_name + '/' + file
                np.savetxt(save_dir, fet, delimiter=",")


    def check_crosvalid(self):
        dir_list = ['subjectI', 'subjectII']
        for dir_name in dir_list:
            file_list = self.get_file_list('feature_vector/' + dir_name)
            for file in file_list:
                data = np.genfromtxt('feature_vector/' + dir_name + '/' + file, delimiter=',')
                if len(data)>=10:
                    kf_total = cross_validation.KFold(len(data), n_folds=10, indices=True, shuffle=True, random_state=4)
                    for train, test in kf_total:
                        print train, '\n', test, '\n\n'
                    print data[train]
                    print data[test]

            break
    def draw_figure(self):
        print ''
        import matplotlib.pyplot as plt

        dir_list = ['subjectI', 'subjectII']
        i=0
        for dir_name in dir_list:

            # pp = PdfPages(dir_name+'.pdf')
            file_list = self.get_file_list(dir_name)
            for file in file_list:
                save_ang_file = 'phi/' + dir_name + '/' + file
                save_v_file = 'velocity/' + dir_name + '/' + file
                angle = (np.genfromtxt(save_ang_file, delimiter=','))
                velocity = (np.genfromtxt(save_v_file, delimiter=','))
                # print (angle)
                # plt.plot( velocity ,'b')
                # plt.plot(angle ,'r')
                # plt.plot(velocity,angle)
                plt.plot(velocity, angle, 'go-', label='line 1', linewidth=2)
                nam = file.split('.')
                plt.title(dir_name+' '+nam[0])
                plt.xlabel('Number of Samples')
                plt.ylabel('Absolute Value')
                # plt.savefig(pp, format='pdf')
                plt.show()
                break
            # pp.close()
        plt.close()
                # plt.show()
                # break
    def draw_sample(self):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        dir_list = ['subjectI', 'subjectII']

        for dir_name in dir_list:
            file_list = self.get_file_list('feature_vector/' + dir_name)
            print dir_name
            anttext = ''
            pp = PdfPages(dir_name+'_velocity_angle.pdf')
            isprint =np.array([1,1,1,1,1,1])
            clr = -1
            for file in file_list:
                print file
                data = np.genfromtxt('feature_vector/' + dir_name + '/' + file, delimiter=',')
                # print data.shape
                if data.ndim < 2:
                    x=data[1]
                    y=data[0]
                    point1 = x
                    point2 = y
                    clr = data[2]
                else:
                    y = data[:,0] # angle
                    x = data[:,1] #velocity
                    point1 = x[0]
                    point2 = y[0]
                    clr = data[0][2]
                # print clr
                # print x
                # print y
                point1 = np.min(x)
                point2 = np.min(y)
                c = 'ro' #default red
                fclr = 'red'
                if (int(clr)==1):

                    c = 'ro'
                    fclr ='red'
                    anttext ='sitting'
                elif (int(clr)==2):

                    anttext ='standing 1m'
                    fclr = 'blue'
                    c ='bo'
                elif (int(clr)==3):
                    anttext ='standing 2m'

                    fclr = 'magenta'
                    c ='mo'
                elif (int(clr)==4):
                    anttext ='walking backward'

                    fclr = 'green'
                    c ='go'
                elif (int(clr)==5):
                    anttext ='walking forward'

                    fclr = 'yellow'
                    c ='yo'
                plt.plot(x,y,c)
                plt.xlabel('Mean Velocity')
                plt.ylabel('Mean Angle')
                plt.title(dir_name)

                if int (clr)>0 and isprint[int(clr)]==1:
                    isprint[int(clr)]=0
                    clr = -1
                    plt.annotate(anttext, xy=(point1, point2), xytext=(1.5, point2),
            arrowprops=dict(facecolor=fclr, shrink=0.05),
            )
                    # plt.annotate(anttext, xy=(x, y), xytext=(3, 1.5), arrowprops=dict(facecolor='black', shrink=0.05),)
                # plt.show()
                # break
            plt.savefig(pp, format='pdf')
            # plt.show()
            pp.close()
            plt.close()

    def draw_phase(self,dirname):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        file_list = self.get_file_list(dirname)

        i = 1
        for file in file_list:
            f = file.split('.')
            pp = PdfPages(dirname+'phi/'+f[0]+'.pdf')
            data = np.genfromtxt(dirname+ file, delimiter=',')
            # fig = plt.figure(i)
            plt.plot(data)
            plt.savefig(pp, format='pdf')
            pp.close()
            plt.close()

    def gen_delta_theta(self,dirname):
        file_list = self.get_file_list(dirname)
        for file in file_list:
            f = file.split('.')
            data = np.genfromtxt(dirname+ file, delimiter=',')
            dif_data = np.diff(data)
            np.savetxt('analysis/deltaphi/'+file, dif_data, delimiter=",")
if __name__ == '__main__':
    p = Processor()
    # p.gen_features()
    # p.process()
    # p.calculate_angles()
    # p.gen_feature_vector()
    # p.check_crosvalid()
    p.draw_phase('analysis/phi/unwrap/')
    # p.gen_delta_theta('analysis/phi/')