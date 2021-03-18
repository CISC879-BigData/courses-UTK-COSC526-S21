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
from pyspark import SparkContext, SparkConf
import sys
import os
import argparse
import json
from scipy.stats import normaltest

# Import local libraries
from . import stats
from . import data_handler as dh

''' Gets a list of features from an input text file '''
def LoadFeatures(features_file):
    with open(features_file) as f:
        features = f.readlines()
    # Remove '\n' from end of each feature
    features = map(lambda x: x.strip(), features)
    return features

''' Replaces column labels for different years with most reason scheme '''
def ReplaceFeature(labels, feature):
    if feature in labels:
        return feature
    # special case for modification code
    elif feature == 'DRDMRUF':
        return 'DR1MC'
    elif feature.replace('DRX', 'DR1') in labels:
        return feature.replace('DRX', 'DR1')
    elif feature.replace('DRD', 'DR1') in labels:
        return feature.replace('DRD', 'DR1')
    elif feature.replace('DR2', 'DR1') in labels:
        return feature.replace('DR2', 'DR1')
    else:
       return feature

''' Counts the number of features in each data set '''
def CountFeatures(data, features):
    info = {}
    for key, value in data.items():
        info[key] = {}
        count = 0
        for f in features:
            if f in value[0]:
                count += 1
        info[key]['features'] = count
    return info

''' Returns a list of features which is present in every dataset '''
def RemoveFeatures(data, features):
    missing_feat = []
    for key, values in data.items():
        header = values[0]
        for feat in features:
            if feat not in header:
                missing_feat.append(feat)
    missing_feat = set(missing_feat) # unique missing features
    features = [feat for feat in features if feat not in missing_feat]
    return features

# Removes data with too many missing values or weight of 0
def FilterData(data, missing, no_mcf=False, no_gf=False):
    for key, values in data.items():
        header = values[0]
        gram = header.index('DR1IGRMS')
        try:
            mc = header.index('DR1MC')
        except:
            mc = False
        # filter by missing values
        tmp = values[1].filter(lambda x: list(x).count(None) <= missing)
        tmp = tmp.map(lambda x: [0 if v is None else v for v in x])
        # Filter by a weight of 0
        if not no_gf:
            tmp = tmp.filter(lambda x: float(x[gram]) != 0.0)
        # Filter by modification code
        if (not no_mcf) and mc:
            tmp = tmp.filter(lambda x: int(x[mc]) == 0)
        data[key] = (header, tmp)
    return data

def _KeyValue(x, size):
    key = x[0]
    value = np.array(x[-size:], dtype=object)
    return (key, value)

def MakeKeyValue(data, features):
    size = len(features)-3 # -3 to account for fdcd, seqn, and mdcd
    for key, values in data.items():
        tmp = values[1].map(lambda x: _KeyValue(x, size))
        data[key] = tmp
    return data

# Gets index of desired features for each year (may vary year to year)
# Returns an RDD with feature vectors
def GetFeatures(data, features):
    del features[2]
    for key, values in data.items():
        header = values[0]
        ind = [header.index(v) for v in features]
        tmp = values[1].map(lambda x: _MakeVector(x, ind))
        data[key] = tmp
    return data

# Concatenates the different years of data
def JoinYears(data):
    # Dummy variable to catch first instance, find a better way to do this
    concat_data = 0
    for value in data.values():
        if concat_data == 0:
            concat_data = value
        else:
            concat_data = concat_data.union(value)
    return concat_data

def _Half(x):
    if len(x) == 1:
        return x
    return x[-len(x)/2:]

def _Sort(x):
    x.sort(key=lambda y: y[0])
    return x

def AvgTopi(data, i):
    data = data.mapValues(lambda x: [x])
    data = data.reduceByKey(lambda x,y: x+y)
    data = data.mapValues(_Sort)
    if i == 'half':
        data = data.mapValues(_Half)
    else:
        data = data.mapValues(lambda x: x[-i:])
    data = data.mapValues(lambda x: np.array(x, dtype=float))
    data = data.mapValues(lambda x: np.average(x, axis=0))
    return data

if __name__ == '__main__':
    # Get input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, required=True,
        help='directory of raw NHANES data')
    parser.add_argument('-v', '--verbose', help='makes output verbose',
        action='store_true')
    parser.add_argument('-f', '--features', type=str, required=True,
        help='''file containing list of features to extract from data;
        1 feature per line;
        first feature should be key (food code);
        second feature should be weight of food entry
        third feature should be modification code of food entry''')
    parser.add_argument('-m', '--missing', type=int, default=2,
        help='number of missing features to drop a food entry')
    parser.add_argument('-s', '--start', type=int, default=1999,
        help='starting year of range of NHANES data to load')
    parser.add_argument('-e', '--end', type=int, default=2011,
        help='ending year of range of NHANES data to load')
    parser.add_argument('-o', '--output', type=str, required=True,
        help='file for processed data output')
    parser.add_argument('-p', '--parallel', type=int, default=16,
        help='number of partitions for spark RDD')
    parser.add_argument('--no-mcf', dest='no_mcf', action='store_true',
        help='boolean value for not filtering on non-zero mod codes')
    parser.add_argument('--no-gf', dest='no_gf', action='store_true',
        help='boolean value for not filtering on zero grams')
    parser.add_argument('--no-std', dest='no_std', help='dont standardize data columns',
            action='store_true')
    parser.set_defaults(mcf=True)
    args = parser.parse_args()

def cleanNHANESData(sc, data, features):
    # Processes data to split header from data and fill missing values
    for key, rdd in data.items():
        data[key] = dh.Process(rdd)
    # data = {filename:(header,RDD), filename:(header,RDD), ...}

    # Repartitions RDDs for better performance
    for key, value in data.items():
        rdd = value[1].repartition(8)
        data[key] = (value[0], rdd)

    # Make all headers have same feature names (each year can be different)
    for key, value in data.items():
        header = value[0]
        for i, col in enumerate(header):
            header[i] = ReplaceFeature(features, col)
        data[key] = (header, value[1])

    # Get list of features common to all years
    features = features[:4] + RemoveFeatures(data, features[4:])

    # Remove unwanted features from the data
    for key, value in data.items():
        header, rdd = value
        header, rdd = dh.FeatureVector(rdd, header, features)
        data[key] = (header, rdd)

    # Filter out entries with too many missing values or weights of 0
    data = FilterData(data, 0, no_mcf=False, no_gf=False)

    # Extracts features for each year, taking into account the difference in
    # column header organization between years
    # data = {year: RDD[year_data], ...}
    data = MakeKeyValue(data, features)

    # Combine different years into one RDD
    data = JoinYears(data)

    # Get top half or i items for each food
    data = AvgTopi(data, i=5)

    # Normalize
    data = data.mapValues(lambda x: x[1:]/x[0])

    # Standardize each column of the features
    # RDD = RDD[(key, [standardized features])]
    data = stats.Standardize(data)

    return data
