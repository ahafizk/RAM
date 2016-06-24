__author__ = 'prune'

from os.path import isfile, join
from os import listdir
import math
import numpy as np

def get_file_list(dir_name):
        '''
        get_file_list returns name of the files of a directory
        '''
        files = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]
        return files

def framesig( sig, frame_len, frame_step):
        """Frame a signal into overlapping frames.

        :param sig: signal to frame.
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