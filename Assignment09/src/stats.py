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
from operator import add, truediv, sub

# Note: All arrays should be numpy arrays

# Calculates Z-score, also known as standardization
# Input: x       = RDD element [values]
#        mean    = [means]
#	 std_dev = [std_devs]
# Output: RDD element [(value - mean) / std_dev]
def _ZScore(x, mean, std_dev):
    return (x - mean) / std_dev

# Helper function for calculating the standard deviation for each key in an RDD
# Input: x     = RDD element (key, [values])
#	 denom = [denominators]
#	 exp   = exponent to raise fraction to
# Output: RDD element (key, [(value / denom) ^ exp])
def _StdDevKeyHelper(x, vals, exp, op):
    # Gets key from RDD element
    k = x[0]
    # Gets index of denominator value(s) for key
    i = [y[0] for y in vals].index(k)
    return (k, op(x[1], vals[i][1])**exp)

# Calculates Standard Deviation for data by key
# Input: data = RDD of (key, [values])
#	 mean = boolean value, True returns mean values of data
# Output: RDD of (key, [std_dev values])
# Optional Output: [(key, data_mean)]
def StdDevByKey(data, mean):
    # Calculate sum of each column for each key
    data_sum = data.reduceByKey(add)
    # Gets the count of each key
    data_count = data.mapValues(lambda x: 1).reduceByKey(add).collect()
    # Calculates mean of each column for each key
    data_mean = data_sum.map(lambda x: _StdDevKeyHelper(x, data_count, 1, div)).collect()
    # Calculates numerator of std_dev for each column for each key
    data = data.map(lambda x: _StdDevKeyHelper(x, data_mean, 2, sub))
    # Sums numerators for each key
    data = data.reduceByKey(add)
    # Calculates fractions and square-roots of std_devs for each key
    data = data.map(lambda x: _StdDevKeyHelper(x, data_count, 0.5, div))
    # Return mean of data for each key if necessary
    if mean == True:
        return data, data_mean
    return data

# Calculates the Standard Deviation of data columns
# Input: data   = RDD of (key, [values])
#        by_key = boolean value, True gets std_dev by key
#        mean   = boolean value, True returns mean values of data
# Output: [std_dev values]
# Optional Output: [mean values]
def StdDev(data, by_key=False, mean=False):
    # Calculate std_dev by key if necessary
    if by_key == True:
        return StdDevByKey(data, mean)
    # Calculate sum for each column
    data_sum = data.map(lambda x: x[1]).reduce(add)
    # Gets count of data points
    data_count = data.count()
    # Calculates mean of each column
    data_mean = np.array(data_sum)/data_count
    # Calculates numerator of std_dev for each column
    data = data.map(lambda x: (x[1]-data_mean)**2)
    # Sums numerators
    data = data.reduce(add)
    # Calculates fractions and square roots for std_devs
    data = (np.array(data) / data_count)**0.5
    # Return mean of data if necessary
    if mean == True:
        return data, data_mean
    return data

# Standardizes data columns (Z-score)
# Input: data = RDD of (key, [values])
# Output: RDD of (key, [standardized values])
def Standardize(data, by_key=False):
    # Calculate Standard Deviation
    std_dev, data_mean = StdDev(data, by_key=by_key, mean=True)
    # Standardize the data
    data = data.mapValues(lambda x: _ZScore(x, data_mean, std_dev))
    return data

# Finds average of columns by key
# Input: data = RDD of (key, [values])
# Output: RDD of (key, [average values])
def AverageByKey(data):
    # Add column to get count for each key
    data = data.mapValues(lambda x: np.hstack((np.array([1]), x)))
    # Sum columns by key
    data = data.reduceByKey(add)
    # Find the average
    data = data.mapValues(lambda x: x[1:]/x[0])
    return data
