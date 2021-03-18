## Copyright (c)
##    2017 by The University of Delaware
##    Contributors: Michael Wyatt
##    Affiliation: Global Computing Laboratory, Michela Taufer PI
##    Url: http://gcl.cis.udel.edu/, https://github.com/TauferLab
##
## All rights reserved.
##
## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are met:
##
##    1. Redistributions of source code must retain the above copyright notice,
##    this list of conditions and the following disclaimer.
##
##    2. Redistributions in binary form must reproduce the above copyright
##    notice, this list of conditions and the following disclaimer in the
##    documentation and/or other materials provided with the distribution.
##
##    3. If this code is used to create a published work, one or both of the
##    following papers must be cited.
##
##            M. Wyatt, T. Johnston, M. Papas, and M. Taufer.  Development of a
##            Scalable Method for Creating Food Groups Using the NHANES Dataset
##            and MapReduce.  In Proceedings of the ACM Bioinformatics and
##            Computational Biology Conference (BCB), pp. 1 - 10. Seattle, WA,
##            USA. October 2 - 4, 2016.
##
##    4.  Permission of the PI must be obtained before this software is used
##    for commercial purposes.  (Contact: taufer@acm.org)
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
## AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
## IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
## ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
## LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
## CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
## SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
## INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
## CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
## ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
## POSSIBILITY OF SUCH DAMAGE.

import numpy as np

'''
    Loads datafiles into a dictionary
    sc = SparkContext
    fs = List of files to load
    data_dir = directory with data files
    names = list of names for storing data in dictionary
'''
def LoadFiles(sc, fs, data_dir, names=False):
    data = {}
    for i, f in enumerate(fs):
        tmp = sc.textFile(data_dir+f)
        if names == False:
            data[f] = tmp.cache()
        else:
            data[names[i]] = tmp.cache()
    return data

''' Removes header from data RDD'''
def _RemoveHeader(rdd):
    header = rdd.first()
    rdd = rdd.filter(lambda x: x != header).cache()
    header = header.split(',')
    return header, rdd

''' Replaces empty values with None value '''
def _ReplaceEmpty(x, delim):
    x = x.split(delim)
    for i, v in enumerate(x):
        if v in ['', ' ']:
            x[i] = None
    return x

'''
    Splits data into header (column labels) and data then marks missing values
    with a None value
'''
def Process(rdd, header=True, empty=True, delim=','):
    if header == True:
        head, rdd = _RemoveHeader(rdd)
    if empty == True:
        rdd = rdd.map(lambda x: _ReplaceEmpty(x, delim))
    if header == True:
        return (head, rdd)
    else:
        return rdd

''' Returns only columns of x listed by index '''
def _MakeVector(x, index):
    x = np.array(x, dtype=object)
    x = list(x[index])
    return x

'''  Removes unwanted columns from an RDD '''
def FeatureVector(rdd, header, features):
    index = [header.index(feat) for feat in features if feat in header]
    rdd = rdd.map(lambda x: _MakeVector(x, index)).cache()
    header = [header[i] for i in index]
    return header, rdd

