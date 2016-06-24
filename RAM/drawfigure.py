__author__ = 'prune'
# import matplotlib.pyplot as plt
import pylab
import numpy as np

class DrawFigure:

    fig = None
    row = None
    col = None
    position = None
    last_index = None

    def __init__(self,row,col):
        # print 'initialize DrawFigure class'
        # self.num_fig = nfig
        self.fig = pylab.figure(figsize=(12,12))
        self.position = row*100+col*10
        self.last_index = 1
    def add_figure1(self,x,y,index=1,title='',xlabel='',ylabel=''):
        """
        add plot to a specific index of a figure. It takes title, xlabel and ylabel
        """
        self.last_index = index
        ax = self.fig.add_subplot(self.position+index)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.plot(x,y)

    def add_figure(self,sig,index,title='',xlabel='',ylabel=''):
        """
        add plot to a specific index of a figure. It takes title, xlabel and ylabel
        """
        self.last_index = index
        ax = self.fig.add_subplot(self.position+index)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.plot(sig)

    def show_figure(self):
        """
        must call this function to show all the plots.
        """
        pylab.show()