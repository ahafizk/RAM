__author__ = 'hafiz'
from scipy.signal import butter, lfilter,filtfilt,buttord,freqs
from scipy.signal import freqz
from scipy.signal import firwin, remez, kaiser_atten, kaiser_beta
from scipy import signal
from drawfigure import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

class DataFilter(object):
    def __init__(self):
        print ' '
    def show_frequency_response(self,b,a,fs):

        plt.figure(1)
        plt.clf()
        for order in [3, 6]:

            w, h = freqz(b, a, worN=2000)
            plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.grid(True)
        plt.legend(loc='best')
        plt.show()

class BandPassFilter(DataFilter):
    def __init__(self):

        DataFilter.__init__(self)
        print 'band pass filter initialized'

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def bandpass_filter(self,data,samprate,lowcut,highcut):
        b,a = butter(6,[lowcut/(samprate/2.0),highcut/(samprate/2.0)],btype='bandpass',analog=0,output='ba')
        data_f = filtfilt(b,a,data)
        return data_f

    def butter_bandpass_filter(self,data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    # Several flavors of bandpass FIR filters.

    def bandpass_firwin(self, ntaps, lowcut, highcut, fs, window='hamming'):
        nyq = 0.5 * fs
        taps = firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero=False,
                      window=window, scale=False)
        return taps

    def bandpass_kaiser(self, ntaps, lowcut, highcut, fs, width):
        nyq = 0.5 * fs
        atten = kaiser_atten(ntaps, width / nyq)
        beta = kaiser_beta(atten)
        taps = firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero=False,
                      window=('kaiser', beta), scale=False)
        return taps

    def bandpass_remez(self, ntaps, lowcut, highcut, fs, width):
        delta = 0.5 * width
        edges = [0, lowcut - delta, lowcut + delta,
                 highcut - delta, highcut + delta, 0.5*fs]
        taps = remez(ntaps, edges, [0, 1, 0], Hz=fs)
        return taps

    def main_test(self):
        # Sample rate and desired cutoff frequencies (in Hz).
        fs = 200.0
        lowcut = 3.00
        highcut = 99.0

        ntaps = 201
        taps_hamming = self.bandpass_firwin(ntaps, lowcut, highcut, fs=fs)
        taps_kaiser16 = self.bandpass_kaiser(ntaps, lowcut, highcut, fs=fs, width=1.6)
        taps_kaiser10 = self.bandpass_kaiser(ntaps, lowcut, highcut, fs=fs, width=1.0)
        remez_width = 1.0
        taps_remez = self.bandpass_remez(ntaps, lowcut, highcut, fs=fs,
                                    width=remez_width)

        # Plot the frequency responses of the filters.
        plt.figure(1, figsize=(12, 9))
        plt.clf()
        pp = PdfPages('collection5-25/figures/different_filter_kaiser.pdf')
        # First plot the desired ideal response as a green(ish) rectangle.
        rect = plt.Rectangle((lowcut, 0), highcut - lowcut, 1.0,
                             facecolor="#60ff60", alpha=0.2)
        plt.gca().add_patch(rect)

        # Plot the frequency response of each filter.
        w, h = freqz(taps_hamming, 1, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="Hamming window")

        w, h = freqz(taps_kaiser16, 1, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="Kaiser window, width=1.6")

        # w, h = freqz(taps_kaiser10, 1, worN=2000)
        # plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="Kaiser window, width=1.0")

        # w, h = freqz(taps_remez, 1, worN=2000)
        # plt.plot((fs * 0.5 / np.pi) * w, abs(h),
        #          label="Remez algorithm, width=%.1f" % remez_width)

        plt.xlim(-2, 110.0)
        plt.ylim(0, 1.5)
        plt.grid(True)
        plt.legend()
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.title('Frequency response of several FIR filters, %d taps' % ntaps)
        plt.savefig(pp, format='pdf')
        plt.show()
        pp.close()

class HighPassFilter(DataFilter):
    def __init__(self):
        DataFilter.__init__(self)
        print 'high pass filter initialized!'

    def butter_highpass(self,cutoff,fs,order=5):
        b,a = butter(order,cutoff/(fs/2.0),btype='highpass',analog=0,output='ba')
        return b,a

    def highpass_filter(self,data,fs,cutoff,order=5):
        b,a = self.butter_highpass(cutoff,fs,order)
        data_f = filtfilt(b,a,data)
        return data_f

    def show_frequency_response(self,b,a,fs):
        cutoff = 100
        w, h = freqz(b, a, worN=8000)
        plt.subplot(2, 1, 1)
        plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
        plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
        plt.axvline(cutoff, color='k')
        plt.xlim(0, 0.5*fs)
        plt.title("Lowpass Filter Frequency Response")
        plt.xlabel('Frequency [Hz]')
        plt.grid()
        plt.show()

    def test_highpass(self,data,fs=200,cutoff=100,order=5):
        n = len(data)
        T = n//fs         # seconds
        # n = int(T * fs) # total number of samples
        t = np.linspace(0, T, n, endpoint=False)
        b,a = self.butter_highpass(cutoff,200,5)
        w, h = freqz(b, a, worN=8000)
        plt.subplot(2, 1, 1)
        plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
        plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
        plt.axvline(cutoff, color='k')
        plt.xlim(0, 0.5*fs)
        plt.title("Highpass Filter Frequency Response")
        plt.xlabel('Frequency [Hz]')
        plt.grid()

        #filter the data and plot to show
        y = self.highpass_filter(data,fs,cutoff,order)
        plt.subplot(2, 1, 2)
        plt.plot(t, data, 'b-', label='data')
        plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
        plt.xlabel('Time [sec]')
        plt.grid()
        plt.legend()

        plt.subplots_adjust(hspace=0.35)
        plt.show()

class LowPassFilter(DataFilter):
    def __init__(self):
        DataFilter.__init__(self)
        # print 'lowpass filter initialized!'

    def butter_lowpass(self,cutoff,fs,order=5):
        b,a = butter(order,cutoff/(fs/2.0),btype='low',analog=0,output='ba')
        return b,a

    def lowpass_filter(self,data,fs,cutoff,order=5):
        b,a = self.butter_lowpass(cutoff,fs,order)
        data_f = filtfilt(b,a,data)
        return data_f

    def show_frequency_response(self,b,a,fs):
        cutoff = 100
        w, h = freqz(b, a, worN=8000)
        plt.subplot(2, 1, 1)
        plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
        plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
        plt.axvline(cutoff, color='k')
        plt.xlim(0, 0.5*fs)
        plt.title("Lowpass Filter Frequency Response")
        plt.xlabel('Frequency [Hz]')
        plt.grid()
        plt.show()

    def test_lowpass(self,data,fs=200,cutoff=100,order=5):
        n = len(data)
        T = n//fs         # seconds
        # n = int(T * fs) # total number of samples
        t = np.linspace(0, T, n, endpoint=False)
        b,a = self.butter_lowpass(cutoff,200,5)
        w, h = freqz(b, a, worN=8000)
        plt.subplot(2, 1, 1)
        plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
        plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
        plt.axvline(cutoff, color='k')
        plt.xlim(0, 0.5*fs)
        plt.title("Lowpass Filter Frequency Response")
        plt.xlabel('Frequency [Hz]')
        plt.grid()


        #filter the data and plot to show
        y = self.lowpass_filter(data,fs,cutoff,order)
        plt.subplot(2, 1, 2)
        plt.plot(t, data, 'r-', label='data')
        plt.plot(t, y, 'b-', linewidth=2, label='filtered data')
        plt.xlabel('Time [sec]')
        plt.grid()
        plt.legend()

        plt.subplots_adjust(hspace=0.35)
        plt.show()



def test_filter():
        # file = 'Data/activity/csv/walking_complete_1.csv'
        # file='Data/gestures/csv/cooking_moving_utensils.csv'
        file ='../Data/gestures/csv/brushing_apply_toothpaste_1.csv'
        # file ='Data/gestures/csv/cooking_moving_utensils.csv'
        # from pylab import plot, show, title, xlabel, ylabel, subplot
        # from scipy import fft, arange
        # file = 'rawdata/csv/plain.csv'
        data = np.genfromtxt(file, delimiter=',')
        x = np.unwrap(data[:,0]/data[:,1])
        x = x[000:800]
        # return x
        return data[0:800,0]



if __name__=='__main__':
    # b,a = butter(2,100/(200/2.0),btype='low',analog=0,output='ba')
    # draw_frequency_response(b,a)
    lp = LowPassFilter()
    data = test_filter()
    lp.test_lowpass(data)
    # lp.show_frequency_response(b,a,200)
    # hp = HighPassFilter()
    # hp.test_highpass(data)