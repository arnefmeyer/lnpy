#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    Loading data from NeoHDF5 files (with some minor modifications)
"""

from os.path import join, split, splitext
import neo
import quantities as pq
from scipy.io import wavfile


class NeoHdf5Reader():
    """Read neo data from hdf5 files

    Parameters
    ----------
    h5_file : str
        The location of the hdf5 file

    """
    def __init__(self, h5_file):

        self.h5_file = h5_file

    def read(self, block_index=0):
        """read neo block from hdf5 file

        Parameters
        ----------
        block_index : int
            Index of the block in the file. Defaults to 0.
        """

        # Read neo block
        reader = neo.io.NeoHdf5IO()
        reader.connect(self.h5_file)
        block = reader.read()[block_index]
        reader.close()

        # Add wav file data to block
        h5path = split(self.h5_file)[0]

        for seg in block.segments:

            # Read data from wav file
            wav_file = seg.annotations['wav_file']
            wav_path = join(h5path, 'wav', wav_file)
            fs, data = wavfile.read(wav_path)

            # Convert int16 to normalized float64
            data = data / 2. ** 15

            # Add samples to segment
            sig = neo.AnalogSignal(data, copy=False, units=pq.V,
                                   t_start=0 * pq.s, file_origin=wav_file,
                                   sampling_rate=fs * pq.Hz,
                                   name=splitext(wav_file)[0])
            seg.annotate(wav_signal=sig)

        return block
