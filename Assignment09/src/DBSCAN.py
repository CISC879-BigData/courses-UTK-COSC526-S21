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

from .search import SimilaritySearch
from operator import add

def _FindCores(x, minpts):
    if len(x[1]) >= minpts:
        return True
    return False

def _CoreNeighbors(x, cores):
    cores = cores.value
    return (x[0], list(set(x[1]).intersection(cores)))

def _FilterNoise(x):
    if len(x[1]) > 0:
        return True
    return False

def _FindMinNeighbor(x, y):
    x[1] = min(x[1], y[1])
    x[0] += y[0]
    return x

def _TestSame(x):
    if x[1][0] == x[1][1]:
        return 0
    return 1

def _TestConverge(rdd1, rdd2):
    rdd1 = rdd1.join(rdd2)
    rdd1 = rdd1.map(_TestSame).reduce(add)
    if rdd1 == 0:
        return True
    return False

def DBSCAN(sc, data, epsilon=10, minpts=10, metric='euclidean'):
    partitions = data.getNumPartitions()

    neighbors = SimilaritySearch(sc, data, metric=metric, radius=epsilon).persist()

    cores = neighbors.filter(lambda x: _FindCores(x, minpts))
    if cores.count() == 0:
        print('Could not find any clusters with given parameters')
        return cores

    cores = cores.keys().collect()
    cores = sc.broadcast(cores)

    neighbors = neighbors.map(lambda x: _CoreNeighbors(x, cores))
    data = neighbors.filter(_FilterNoise)

    min_neighbor = data.mapValues(lambda x: min(x))

    converge = False
    iteration = 0
    while not converge:
        iteration += 1
        # RDD: (self, [neighbors])
        data = data.map(lambda x: (x[1], x[0]))
        # RDD: ([neighbors], self)
        data = data.flatMap(lambda x: [(y, x[1]) for y in x[0]])
        # RDD: (neighbor, self)
        data = data.leftOuterJoin(min_neighbor, partitions)
        # RDD: (neighbor, (self, min_neighbor))
        data = data.map(lambda x: (x[1][0], [[x[0]], x[1][1]]))
        # RDD: (self, [[neighbor], min_neighbor])
        data = data.reduceByKey(_FindMinNeighbor)
        tmp = data.map(lambda x: (x[0], x[1][1]))
        if _TestConverge(min_neighbor, tmp):
            converge = True
        min_neighbor = tmp
        data = data.map(lambda x: (x[0], x[1][0]))

    print ('DBSCAN completed in {} iterations'.format(iteration))

    return min_neighbor

