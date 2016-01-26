#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    Maximum informative dimensions (MID) solver (Sharpee et al. 2004)

    This is a wrapper around the MID solver by TO Sharpee available at:

    https://github.com/sharpee/mid

    You have to put the executable in the same directory as this file.

    Remark: Currently, only the 1D solver is working properly!
"""

import os
from os import listdir, remove, chdir, makedirs
from os.path import join, exists, split
from .base import LNPEstimator
import numpy as np
from xml.dom.minidom import Document
import time
import tempfile


class MID(LNPEstimator):
    """
        MID analysis (Sharpee et al. Neural Comput 2004)

        Receptive field estimation based on maximization of mutual
        information (MI) between stimulus and response

    """

    def __init__(self, bins=50, n_iter=50, tempdir=None, ndim=1,
                 n_iter2=None):
        """Constructor

            Inputs:
                size    - size of the receptive field (height, width)
                bins    - number of histogram bins
                n_iter  - number of iterations
                tempdir - temporary directory

        """
#        super(MID, self).__init__(self)
        (location, _) = os.path.split(__file__)
        self.bins = bins
        self.n_iter = n_iter
        self.tempdir = tempdir
        self.ndim = ndim
        self.n_iter2 = n_iter2
        if n_iter2 is None:
            self.n_iter2 = n_iter
        self.coef_ = None
        self.intercept_ = 0.
        self.__exe_1D = join(location, 'mid1d')
        self.__exe_2D = join(location, 'midnd')

    @property
    def name(self):
        return 'MID'

    def fit(self, X, Y):
        """Estimates RF from given data by MI maximization

            Inputs:
                X - array holding stimulus data with dimensions
                    (samples, features)
                y - array holding spike counts with dimesions (samples,)

        """
        # Temporary location of files passed to MID executable
        temp = self.tempdir
        if temp is None:
            temp = tempfile.mkdtemp()
        if not exists(temp):
            makedirs(temp)

        try:
            # Write stimulus (binary) and response data (text)
            _, base = tempfile.mkstemp(prefix='MID', dir=temp)
            _, prefix = split(base)

            stim_file = base + '.raw'
            resp_file = base + '.isk'
            param_file = base + '.xml'
            X.astype('float').tofile(stim_file)
            if Y.ndim > 1:
                np.savetxt(resp_file, np.sum(Y, axis=1).astype('int'),
                           fmt='%d')
            else:
                np.savetxt(resp_file, Y.astype('int'), fmt='%d')

            # Parameters have to be saved as xml file
            self.__params2xml__(prefix, param_file, stim_file, resp_file,
                                (X.shape[1], 1), self.bins, self.n_iter)

            # Run algorithm
            cmd = '%s %d %s 1' % (self._MID__exe_1D, 1, param_file)
            cwd = os.getcwd()
            os.chdir(temp)
            t0 = time.time()

            status = os.system(cmd)
            if self.ndim > 1:
                self.__params2xml__(prefix, param_file, stim_file, resp_file,
                                    (X.shape[1], 1), self.bins, self.n_iter2)
                cmd = '%s %d %s 1' % (self._MID__exe_2D, self.ndim,
                                      param_file)
                status = os.system(cmd)
            self.t_fit = time.time() - t0
            self.t_total = self.t_fit
            chdir(cwd)
            if status:
                print "An error occured while calling mid1d"

            # Read first MID from binary file
            if self.ndim == 1:
                dat_file = join(temp, prefix + '-1D-n1-v1-p1.dat')
                self.coef_ = np.fromfile(dat_file, dtype=np.double)

            elif self.ndim > 1:
                dat_file1 = join(temp, prefix + '-ND-n2-v1-p1.dat')
                dat_file2 = join(temp, prefix + '-ND-n2-v2-p1.dat')
                self.coef_ = [np.fromfile(dat_file1, dtype=np.double),
                              np.fromfile(dat_file2, dtype=np.double)]

        finally:
            # Clean up
            tmpfiles = [f for f in listdir(temp) if prefix in f]
            for f in tmpfiles:
                remove(join(temp, f))

    def __params2xml__(self, prefix, xmlfile, stim_file, resp_file, rfsize,
                       bins, n_iter):
        """Write estimation parameters to XML file

        """
        # Create document
        doc = Document()

        # Root element
        base = doc.createElement('Configuration')
        doc.appendChild(base)

        # Spike parameters
        params = doc.createElement('ParametersGroup')
        params.setAttribute('name', 'Spike Parameters')
        base.appendChild(params)
        cs = []
        cs.append(doc.createElement('StringListParameter'))
        cs[-1].setAttribute('name', 'spike files')
        cs[-1].setAttribute('value', resp_file)
        cs.append(doc.createElement('IntegerParameter'))
        cs[-1].setAttribute('name', 'number of parts')
        cs[-1].setAttribute('value', '4')
        cs.append(doc.createElement('IntegerParameter'))
        cs[-1].setAttribute('name', 'number of trials')
        cs[-1].setAttribute('value', '%d' % 0)
        for c in cs:
            params.appendChild(c)

        # Annealing parameters
        anneal = doc.createElement('ParametersGroup')
        anneal.setAttribute('name', 'Annealing Parameters')
        base.appendChild(anneal)
        ca = []
        ca.append(doc.createElement('IntegerParameter'))
        ca[-1].setAttribute('name', 'max annealing iterations')
        ca[-1].setAttribute('value', "1")
        ca.append(doc.createElement('DoubleParameter'))
        ca[-1].setAttribute('name', 'start temperature')
        ca[-1].setAttribute('value', "1")
        ca.append(doc.createElement('DoubleParameter'))
        ca[-1].setAttribute('name', 'stop temperature')
        ca[-1].setAttribute('value', "1.0e-5")
        ca.append(doc.createElement('DoubleParameter'))
        ca[-1].setAttribute('name', 'down temperature factor')
        ca[-1].setAttribute('value', "0.95")
        ca.append(doc.createElement('DoubleParameter'))
        ca[-1].setAttribute('name', 'up temperature factor')
        ca[-1].setAttribute('value', "10")
        ca.append(doc.createElement('DoubleParameter'))
        ca[-1].setAttribute('name', 'function tolerance')
        ca[-1].setAttribute('value', "5.0e-5")
        ca.append(doc.createElement('IntegerParameter'))
        ca[-1].setAttribute('name', 'updateFactor')
        ca[-1].setAttribute('value', "100")
        for c in ca:
            anneal.appendChild(c)

        # Movie parameters
        movie = doc.createElement('ParametersGroup')
        movie.setAttribute('name', 'Movie Parameters')
        base.appendChild(movie)
        cm = []
        cm.append(doc.createElement('StringListParameter'))
        cm[-1].setAttribute('name', 'movie files')
        cm[-1].setAttribute('value', stim_file)
        cm.append(doc.createElement('EnumeratorParameter'))
        cm[-1].setAttribute('name', 'data type')
        cm[-1].setAttribute('value', "2")
        cm[-1].setAttribute('values', "byte:1:double:2")
        cm.append(doc.createElement('IntegerParameter'))
        cm[-1].setAttribute('name', 'width')
        cm[-1].setAttribute('value', "%d" % rfsize[1])
        cm.append(doc.createElement('IntegerParameter'))
        cm[-1].setAttribute('name', 'x offset')
        cm[-1].setAttribute('value', "1")
        cm.append(doc.createElement('IntegerParameter'))
        cm[-1].setAttribute('name', 'sta width')
        cm[-1].setAttribute('value', "%d" % rfsize[1])
        cm.append(doc.createElement('IntegerParameter'))
        cm[-1].setAttribute('name', 'x downsample')
        cm[-1].setAttribute('value', "1")
        cm.append(doc.createElement('IntegerParameter'))
        cm[-1].setAttribute('name', 'height')
        cm[-1].setAttribute('value', "%d" % rfsize[0])
        cm.append(doc.createElement('IntegerParameter'))
        cm[-1].setAttribute('name', 'y offset')
        cm[-1].setAttribute('value', "1")
        cm.append(doc.createElement('IntegerParameter'))
        cm[-1].setAttribute('name', 'sta height')
        cm[-1].setAttribute('value', "%d" % rfsize[0])
        cm.append(doc.createElement('IntegerParameter'))
        cm[-1].setAttribute('name', 'y downsample')
        cm[-1].setAttribute('value', "1")
        cm.append(doc.createElement('IntegerParameter'))
        cm[-1].setAttribute('name', 'sta duration')
        cm[-1].setAttribute('value', "1")
        cm.append(doc.createElement('IntegerParameter'))
        cm[-1].setAttribute('name', 'skipped sta frames')
        cm[-1].setAttribute('value', "0")
        cm.append(doc.createElement('IntegerListParameter'))
        cm[-1].setAttribute('name', 'number of bins')
        cm[-1].setAttribute('value', "%d" % bins)
        cm.append(doc.createElement('IntegerListParameter'))
        cm[-1].setAttribute('name', 'number of iterations')
        cm[-1].setAttribute('value', "%d" % n_iter)
        for c in cm:
            movie.appendChild(c)

        # Output parameters
        output = doc.createElement('ParametersGroup')
        output.setAttribute('name', 'Output Parameters')
        base.appendChild(output)
        co = doc.createElement('StringParameter')
        co.setAttribute('name', 'prefix')
        co.setAttribute('value', prefix)
        output.appendChild(co)

        # Finally, write document to formatted xml file
        with open(xmlfile, 'w') as f:
            doc.writexml(f, indent="", addindent="   ", newl="\n",
                         encoding='UTF-8')
