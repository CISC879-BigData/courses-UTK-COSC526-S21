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
import sys

def _Euclidean(x, y):
    return np.sum((x - y) ** 2) ** 0.5

def _Cosine(x, y):
    dist = np.dot(x, y)
    dist = dist / ((np.sum(x ** 2) ** 0.5) * (np.sum(y ** 2) ** 0.5))
    dist = 1 - dist
    return dist

# Computer distance between two points
def PPDistance(x, y, metric='euclidean'):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    if metric == 'euclidean':
        return _Euclidean(x, y)
    if metric == 'cosine':
        return _Cosine(x, y)

def _RadiusSearch(x, neighbors, radius, metric):
    neighbors = neighbors.value
    key = x[0]
    val = x[1]
    rn= []
    for neighbor in neighbors:
        if key == neighbor[0]:
            rn.append(neighbor[0])
            continue
        dist = PPDistance(val, neighbor[1], metric=metric)
        if dist <= radius:
            rn.append(neighbor[0])
    return (key, rn)

def SimilaritySearch(sc, data, metric='euclidean', radius=1):
    neighbors = data.collect()
    neighbors = sc.broadcast(neighbors)

    data = data.map(lambda x: _RadiusSearch(x, neighbors, radius, metric))

    return data

