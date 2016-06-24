__author__ = 'hafiz'
#!/usr/bin/env python
# a bar plot with errorbars
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
def autolabel(rects,ax):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
                    ha='center', va='bottom')


def mean_std():
    activities= ['Entertainment','Cooking','Cleaning','Bathroom']
    subjects = ['sub1','sub2']
    dir_name = 'FullResults/'
    for act in activities:
        file1 = dir_name+'sub1/'+act+'.csv'
        data1 = np.genfromtxt(file1, delimiter=",")
        file2 = dir_name+'sub2/'+act+'.csv'
        data2 = np.genfromtxt(file2, delimiter=",")
        data = data1 + data2
        data = data/2.0
        row,col = data1.shape
        data_mean = []
        data_std = []
        for i in range(row):
            std = []
            mn = []
            for j in range(col):
                a = np.array([data1[i][j],data2[i][j]])
                mn.append(np.mean(a))
                std.append(np.std(a))
            data_mean+=[mn]
            data_std+=[std]
        data_mean = np.array(data_mean)
        print data
        print data_mean
        data_std = np.array(data_std)
        print data_std
        np.savetxt(dir_name+'mean_std/'+act+'_mean.csv',data_mean,delimiter=',')
        np.savetxt(dir_name+'mean_std/'+act+'_std.csv',data_std,delimiter=',')

def draw_framelength(filename,title):
    dir_name ='FullResults/mean_std/'

    # matplotlib.rcParams.update({'font.size': 15})
    pp = PdfPages(filename+"framelength.pdf")
    # mean_std()
    ent_mean = np.genfromtxt(dir_name+filename+'_mean.csv',delimiter=',')
    ent_std = np.genfromtxt(dir_name+filename+'_std.csv',delimiter=',')
    row,col = ent_mean.shape
    index = np.arange(row)
    width = 0.25       # the width of the bars
    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    # fig, ax = plt.subplots()
    fig, ax = plt.subplots()
    rects1 = ax.bar(index, ent_mean[:,1], width, alpha=opacity,color='r', yerr=ent_std[:,1],error_kw=error_config,label='Fusion1')
    rects2 = ax.bar(index+width, ent_mean[:,2], width,alpha=opacity, color='y', yerr=ent_std[:,2],error_kw=error_config,label='Fusion2')
    rects3 = ax.bar(index+2*width, ent_mean[:,3], width, color='c', yerr=ent_std[:,3],error_kw=error_config,label='Fusion3')
    ax.set_ylabel('Accuracy (%)')
    # ax.set_title(title)
    ax.set_xticks(index+width+width/2)
    # framelenghts=[1,2.5,5,7.5,10,12.5,15,17.5,20,22.5,25]
    ax.set_xticklabels( (1,2.5,5,7.5,10,12.5,15,17.5,20,22.5,25) )
    # ax.set_xticklabels( (1,2.5,5,7.5,10,12.5,15,17.5,20,22.5,25) )
    # ax.set_xticklabels( (200, 500, 1000, 1500, 2000,2500,3000,3500,4000,4500) )
    ax.set_xlabel('Frame Lengths (sec)')
    # ax.legend( (rects1[0], rects2[0],rects3[0]), ('Fusion1', 'Fusion2','Fusion3') )
    ax.legend(prop={'size':11})
    autolabel(rects1,ax)
    autolabel(rects2,ax)
    autolabel(rects3,ax)
    plt.tight_layout()
    # plt.show()
    pp.savefig(fig)
    pp.close()


if __name__=='__main__':
    activity = 'Entertainment'

    # draw_entertainment()
    activities= ['Entertainment','Cooking','Cleaning','Bathroom']
    for act in activities:
        title=act+' Activity Accuracy by Frame Length and Fusion'
        draw_framelength(act,title)



