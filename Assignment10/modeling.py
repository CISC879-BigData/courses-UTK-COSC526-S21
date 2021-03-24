#!/usr/bin/env python3

# Code by Travis Johnston, 2017; Danny Rorabaugh, 2018-20.
# Polynomial and Nearest-Neighbor modeling methods: KNN, HYPPO, SBM.
# A stripped-down version of hypppo7.py found in SOMOSPIE.


import argparse, csv, random
import numpy as np
# https://docs.python.org/3.1/library/itertools.html#itertools.combinations_with_replacement
from itertools import combinations_with_replacement as cwr 
from itertools import chain
#from os import cpu_count, sched_getaffinity # This import causes issue on some Unix OS
from scipy.special import comb


# Function for logging to specified file, or printing if no file given.
def log(item, file=""):
    if file:
        with open(file, "a") as log_file:
            log_file.write(f"{item}")
    else:
        print(item)


# Parallelization initialization.
def init_parallel():
    #print(f"There are {cpu_count()} cores, of which {len(sched_getaffinity(0))} are available.")

    import findspark
    findspark.init()
    # https://spark.apache.org/docs/0.9.0/api/pyspark/
    global SC
    from pyspark import SparkContext as SC


# This function expects:
# * a list of coefficients for the polynomial in order: 
# * the degree of the polynomial (integer).
# * a point (list of floats) of where to evaluate the polynomial.
# This function returns the value of the polynomial evaluated at the point provided.
def evaluate_polynomial(coefficients, degree, point):
    if degree == 0:
        return coefficients[0]
    
    monomials = [ np.product(x) for x in cwr(chain([1.0], point), degree) ]
    return sum( [ a[0]*a[1] for a in zip(coefficients, monomials) ] )


# Given a list coefs of coefficients, and polynomial dimension n and degree d, 
# This returns the number of non-zero monomials of each degree up to d.
def degree_counts(coefs, n, d):
    i = 0
    tallies = []
    for deg in range(d + 1):
        #print(len(coefs), n, d, i, deg)
        terms = comb(n, deg, exact=True, repetition=True)
        # Quick check to make sure coefs is still as long as it needs to be for given n and d
        if (i + terms) > len(coefs):
            print(coefs, n, d, tallies, i, terms)
            return False
        tallies.append(np.count_nonzero(coefs[i:(i + terms)]))
        i += terms
    #print(d, coefs, tallies, len(coefs), i)
    return tallies


# independent_variable_points is a list of settings for the independent variables that were observed.
# dependent_variable_values is a list of observed values of the dependent variable.
# It is important that for each i the result of independent_variable_points[i] is stored as dependent_variable_values[i].
# degree is the degree of the polynomial to build.
# This function returns the list of coefficients of the best fit polynomial surface of degree "degree".
def determine_coefficients(independent_variable_points, dependent_variable_values, degree):
    
    # If degree==0, cwr returns [()], then np.product[()] is 1.0.
    #print(degree, list(cwr(chain([1.0], independent_variable_points[0]), degree)))
    A = [ [np.product(x) for x in cwr(chain([1.0], iv), degree)] for iv in independent_variable_points ]
    Z = np.array(dependent_variable_values)
    Zbar = np.mean(Z)
            
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.solve.html
    # We need matrix A to be square to solve AX=Z for vector X with np.linalg.solve
    # So multiply both sides on the left by the transpose of A. 
    At = np.transpose(A)
    AtA = np.dot(At, A)
    
    #if degree:
        #print(f"\nAt.shape: {At.shape}\nnp.linalg.matrix_rank(At): {np.linalg.matrix_rank(At)}")#\nA: {A}")
        #print(f"degree: {degree}\nlen(Z): {len(Z)}\nnp.mean(Z): {np.mean(Z)}\nZ: {Z}")
        #print(f"AtA.shape: {AtA.shape}\nnp.linalg.det(AtA): {np.linalg.det(AtA)}")
    
    # If the polynomial is degree zero, just the the average.
    if not degree:
        coef = [Zbar]
        
    # Otherwise, if the determinant of A.transpose() * A is non-zero, 
    #  we can solve the system with the least-squares method
    #elif abs(np.linalg.det(AtA))>2**(-20):
    elif np.linalg.matrix_rank(At) >= At.shape[0]:
        #print(f"AtA has determinent {np.linalg.det(AtA)}")
        coef = np.linalg.solve(AtA, np.dot(At, Z))
        #print(f"solve'd coef: {coef}")
        
    else:
        # If the system is under-determined, 
        # ... or dangerous and naive use of lstsq's built-in handling of underdetermined systems.
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html
        coef = np.linalg.lstsq(A, Z, rcond=-1)[0]#, rcond=None)[0]

        #print(f"lstsq'd coef: {coef}")
    
    return list( coef )


# data_points is a list of the observed independent variable settings.
# specific_points is one chosen setting of the independent variables.
# k is the number of nearest neighbors to find.
# This function returns a list of indices (in data_points) of the k nearest neighbors.
def indices_of_NNs(data_points, specific_point, k, norm=2):
    if len(data_points) == 0:
        print("Error: data_points empty!")
        return False
    if len(data_points[0]) != len(specific_point):
        print("Error: specific_point not same dim as elements of data_points!")
        return False
    
    # If the number of available datapoints is not greater than k, 
    #  then everything is a "nearest neighbor".
    if (k > len(data_points)):
        print("Warning! You're asking for more nearest neighbors than there are available points.") 
    if (k >= len(data_points)):
        return range(len(data_points))
    
    distances = [ sum( (x - specific_point)**norm ) for x in data_points ]
    indices = np.argsort( distances, kind='mergesort' )[:k]
    return indices


# indep_data_points is a list of the observed independent variables to build models from.
# dep_data_points is a list of the observed dependent variables (in the same order).
# args.k is the number of folds or partitions to divide the data into.
# args.cvIters is the number of times the data is randomly partitioned (for averaging over many runs).
# args.degree is the max degree.
def crossvalidation(indep_data_points, dep_data_points, args):#k, num_random_partitions, D):
    
    # Number of data points.
    n = len(indep_data_points)
    
    # A list of 0's of same length as possible degrees.
    Total_SSE = np.zeros(args.degree + 1)
    
    indices = list(range(n))
    
    for iteration in range(args.cvIters):
        # Randomly partition the data into k sets as equally sized as possible.

        # Get a new random shuffling of the indices.
        # https://docs.python.org/2/library/random.html#random.shuffle
        random.shuffle(indices)
        Folds = [ [indices[i] for i in range(fold, n, args.k)] for fold in range(args.k) ]
        
        # Try every possible degree.
        for d in range(args.degree + 1):
            
            #print(d, Folds)

            # Build k models of degree d (each model reserves one set as testing set).
            for testing_fold in range(args.k):
                testing_indep_data = [ indep_data_points[i] for i in Folds[testing_fold] ]
                testing_dep_data = [ dep_data_points[i] for i in Folds[testing_fold] ]
                
                model_indep_data = []
                model_dep_data = []
                for fold in range(args.k):
                    if fold != testing_fold:
                        model_indep_data.extend([ indep_data_points[i] for i in Folds[fold] ])
                        model_dep_data.extend([ dep_data_points[i] for i in Folds[fold] ])
                
                best_SSE = 9999
                SSE = 0
                # Get the polynomial built from the model data of degree d.

                try:
                    coefficients = determine_coefficients(model_indep_data, model_dep_data, d)

                    # Predict the testing points and add the error to the Total_SSE[d].
                    for x, z in zip(testing_indep_data, testing_dep_data):
                        # The square of the difference between polynomial prediction and observed value (z) at x.
                        SSE += (evaluate_polynomial(coefficients, d, x) - z)**2    
                    #print(f"d: {d}; Total_SSA[d]: {Total_SSE[d]}; \ncoefficients: 
                    if SSE <= best_SSE:
                        best_SSE = SSE
                except:
                    SSE = 9999
                
                Total_SSE[d] += best_SSE

    # Return index of minimum Total_SSE.
    # Note: Total_SSE[i] corresponds to polynomial of degree i.
    winning_degree = np.argmin(Total_SSE)
    
    #print(f"n: {n}; winning_degree: {winning_degree}; \nTotal_SSE: {Total_SSE}\n")
    #print(f"Total_SSE: {Total_SSE}")
    return [winning_degree, list(Total_SSE)]


# Used by model_at_point to determine local model degree.
def determine_model_degree(indep, dep, args):
    if args.model=="KNN":
        # Setting the degree to 0 forces us to just average the nearest neighbors.
        # This is exactly kNN (a degree 0 polynomial).
        degree_with_errors = [0, []]

    elif args.model in ["SBM", "HYPPO"]:#=="SBM":
        degree_with_errors = crossvalidation(indep, dep, args)

    else:
        raise ValueError(f"\"{args.model}\" is not a valid model.")
        
    return degree_with_errors
    

def create_model(selected_indep_data, selected_dep_data, args):
    # Determine the best polynomial degree.
    degree, errors = determine_model_degree(selected_indep_data, selected_dep_data, args)
    
    # Compute the coefficients of the "best" polynomial of degree degree.
    #print(f"selected_indep_data: {selected_indep_data}\nselected_dep_data: {selected_dep_data}\ndegree: {degree}")
    coefficients = determine_coefficients(selected_indep_data, selected_dep_data, degree)
    
    return degree, errors, coefficients

    
# Main function for a single neighborhood.
# This function will be called independently many times.
# This can be run on every element of a Spark RDD.
def model_in_neighborhood(selected_indep_data, selected_dep_data, args):

    degree, errors, coefficients = create_model(selected_indep_data, selected_dep_data, args)

    return [degree, coefficients]


# input1 and input2 are arrays or ndarrays.
# Columns index 0 and 1 of input1 and input2 are the x/y-coordinates.
# input1 should have 1 more column than input2, the column with the dependent variable.
# depIndex is the index of the dependent variable column in input1.
# model is one of ["HYPPO", "KNN", "SBM"].
# Implementations of HYPPO and SBM are not well-suited for high dimensional data.
# k is the number of nearest neighbors for HYPPO or KNN (is overridden for SBM).
def main(input1, input2, args=False):
    
    if not args:
        class arguments():
            model="KNN"
            k=10
            depIndex=2
            degree=0
            variables=0
            scale=None
            norm=2
            parallel=0
            lowerBound=0
            upperBound=1
            logFile=""
        args = arguments()
    
    if args.variables:
        indepCount = args.variables
    else:
        indepCount = input1.shape[1] - 1

    shift = np.mean(input1, axis=0)[:indepCount]
    
    # Extract from the training data the values for the dependent and independent variables. 
    # ToDo: Use pandas for importing train data and spliting into indep and dep.
    Independent_Data = []
    Dependent_Data = []
    for line in input1:
        numbers = list(line)
        Dependent_Data.append(numbers.pop(args.depIndex))
        Independent_Data.append(np.array(numbers[:indepCount]))

    # Unless an array of scaling factors is specified, 
    #  every column will be divided by the standard deviation of that column
    #  to normalize the data.
    if args.scale:
        args.scale = np.array([float(s) for s in args.scale.split(',')])
        if len(args.scale)!=indepCount:
            raise ValueError("Error: scale was specified, but isn't the same length as the sepcified number of independent variables!")
    else:
        args.scale = 1/np.std(Independent_Data, axis=0)
    scale=args.scale
    #log(f"scale: {scale}\n", file=args.logFile)

    #log(f"Dependent_Data is an array of length {len(Dependent_Data)} with first elements:\n{Dependent_Data[:5]}\n", file=args.logFile)
    #log(f"Independent_Data is a length-{len(Independent_Data)} array of arrays with first element:\n{Independent_Data[0]}\n", file=args.logFile)    
    
    # Perform any shifting, scaling on the training data.
    # The same operations will also be performed on the evaluation data later as necessary.
    Independent_Data = [(row - shift)*scale for row in Independent_Data]
    #log(f"Independent_Data post-scaling is an array of arrays with first element:\n{Independent_Data[0]}\n", file=args.logFile)
 
    # If SBM, we will need to shuffle and partition the training data multiple times, and use all data as neighbors.
    if args.model=="SBM":
        args.cvIters = 10
        args.k = len(Dependent_Data) - 1
    else:
        args.cvIters = 1
    log(f"Each local model will be generated with {args.k} nearest neighbors.\n", file=args.logFile)
    
    # This is the operation that takes a point (X) and returns its coordinates (xy) and itself shifted (P).
    def XtP(X):
        xy = list(X[:2])
        P = (np.array(X[:indepCount]) - shift)*scale
        return (xy, P)
        
    # This is the operation that finds the neighborhood of a point (P).
    def NoP(P):
        indices_of_nearest_neighbors = indices_of_NNs(Independent_Data, P, args.k, norm=args.norm)
        return frozenset(indices_of_nearest_neighbors)
    
    # This is the operation that finds the data for the neighbors of a neighborhood (N).
    def DoN(N):
        selected_indep_data = [ Independent_Data[i] for i in N ]
        selected_dep_data = [ Dependent_Data[i] for i in N ]
        return [selected_indep_data, selected_dep_data]
    
    # This is the operation that finds the degree and coefficients of the model 
    #  for a neighborhood (Ndata = [indep_data, dep_data]).
    def MiN(Ndata):
        return model_in_neighborhood(Ndata[0], Ndata[1], args)
               
    # This is the operation that evaluates a local model (M = [deg, coefs]) at every point in a neighborhood (Ps).
    def EoN(M, xyPs):
        degree, coefs = M
        zs = [evaluate_polynomial(coefs, degree, xyP[1]) for xyP in xyPs]
        zs = [min(max(z, args.lowerBound), args.upperBound) for z in zs]
        #return [(xyP[0][0], xyP[0][1], z, degree) for (xyP, z) in zip(xyPs, zs)]
        return [(xyP[0][0], xyP[0][1], z) for (xyP, z) in zip(xyPs, zs)]
    
    # Run the above operations on the eval data, in parallel if directed to; ...
    if args.parallel:   
        init_parallel()
        sc = SC.getOrCreate()
        rdd = sc.parallelize(input2)
        rdd = rdd.map(XtP)
        
        rdd = rdd.map(lambda xyP: (NoP(xyP[1]), [xyP]))
        rdd = rdd.reduceByKey(lambda a, b: a + b)
        
        rdd = rdd.map(lambda NP: (DoN(NP[0]), NP[1]))
        rdd = rdd.map(lambda DP: (MiN(DP[0]), DP[1]))
        
        rdd = rdd.map(lambda MP: EoN(*MP))
        
        output = rdd.reduce(lambda a, b: a + b)
        sc.stop()
            
    # ... otherwise, do it in serial.        
    else:
        # First stage: shift points.
        xyPs = [XtP(X) for X in input2]
        
        # Second stage: sort the points into their neighborhoods.
        stored_nbrs = {}
        for xyP in xyPs:
            nbrs = NoP(xyP[1])
            if nbrs in stored_nbrs:
                stored_nbrs[nbrs].append(xyP)
            else:
                stored_nbrs[nbrs] = [xyP]
        
        # Third stage: compute the coefficients for each neighborhood.
        stored_nbrdata = {N: DoN(N) for N in stored_nbrs}
        #print(stored_nbrdata[sampleN])
        stored_coefs = {N: MiN(stored_nbrdata[N]) for N in stored_nbrs}
        #print(stored_coefs[sampleN])
        
        # Fourth stage: evaluate the model of each neighborhood on every point in that neighborhood.
        stored_evaluations = {N: EoN(stored_coefs[N], stored_nbrs[N]) for N in stored_nbrs}
        #print(stored_evaluations[sampleN])
        
        # Finally, flatten all the predictions into a single array.
        output = [xyzd for N in stored_evaluations for xyzd in stored_evaluations[N]]
    
    return output


def get_args():  
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", #required=True,
                        help="The path to the csv file containing the training data (required).")
    parser.add_argument("-m", "--model", choices=["HYPPO", "KNN", "SBM"], default="KNN", 
                        help="The type of model to build (default: %(default)s).")
    parser.add_argument("-k", "--k", type=int, default=10, 
                        help="The number of nearest neighbors to use for either the KNN or HYPPO model. Number of folds for cross-validation to use with the SBM model (default: %(default)s).")
    parser.add_argument("-e", "--eval", #required=True, 
                        help="Name of file where the evaluation points are stored (required).")
    parser.add_argument("-o", "--out", 
                        help="Name of file where prediction is to be stored.")
    parser.add_argument("-i", "--depIndex", type=int, default=2, 
                        help="Index of column in train file with dependent variable to be tested for building a model (default: %(default)s).")
    parser.add_argument("-r", "--headerRows", type=int, default=1, 
                        help="Number of rows to ignore, being header row(s) (default: %(default)s).")
    parser.add_argument("-d", "--delimiter", default=",", 
                        help="Delimiter of train and eval files (default: %(default)s).")
    parser.add_argument("-D", "--degree", type=int, default=3, 
                        help="Maximum polynomial degree (default: %(default)s). If -m KNN, this will be overrode with 0.")
    parser.add_argument("-v", "--variables", type=int, default=0, 
                        help="Number of independent variables--i.e. number of predictor columns (default: 0, uses all columns).")
    parser.add_argument("-S", "--scale", 
                        help="Specify the scale to multiply your independent variables by; for example -s0 -v2 -S1,2. Uses reciprocals of standard deviations if unspecified.")
    parser.add_argument("-N", "--norm", type=int, default=2, 
                        help="Specify N for l_N norm; default is 2 (Euclidean). This is used for identifying the nearest neighbors.")
    parser.add_argument("-p", "--parallel", type=int, default=0, 
                        help="1 to run in parallel with Spark; 0 otherwise (default).")
    parser.add_argument("-b", "--lowerBound", type=float, default=0, 
                        help="A firm lower bound for model output (default: %(default)s).")
    parser.add_argument("-B", "--upperBound", type=float, default=1, 
                        help="A firm upper bound for model output (default: %(default)s).")
    parser.add_argument("-l", "--logFile", default="", 
                        help="The path for a log file; will print instead of logging if empty string (default).")

    return parser.parse_args()
    

if __name__ == "__main__":  
    
    args = get_args()

    # args.train contains the data from which to build the model.
    # It is expected that the file be comma separated and have a header row.
    # Default format is x, y, z, c1, ..., cm.
    # Where x and y are geographic coordinates, z is the observed dependent variable,  
    #  and c1, ..., cm are additional independent variables.
    # args.eval should be the same format, but lacking the z column.
    
    # Commandline example:
    # ./hypppo6.py -t train.csv -e eval.csv -k9 -Edump.err -Hdump.nbr -Cdump.deg

    # Read in the training data and evaluation data and save to numpy dataframes
    original_values = np.loadtxt(args.train, delimiter=args.delimiter, skiprows=args.headerRows)
    log(f"\n{len(original_values)} lines of original data have been loaded from {args.train}.\n", file=args.logFile)
    values_to_model = np.loadtxt(args.eval, delimiter=args.delimiter, skiprows=args.headerRows)
    log(f"{len(values_to_model)} lines of evaluation data have been loaded from {args.eval}.\n", file=args.logFile)

    output = main(original_values, values_to_model, args)

    # If the output filename isn't specified, 
    #  the name will be generated by some arguments, separated by -.
    if not args.out:
        args.out=f"-t{args.train.split('/')[-1]}-e{args.eval.split('/')[-1]}-i{args.depIndex}-m{args.model}-k{args.k}.csv"

    np.savetxt(args.out, output, delimiter=",", fmt='%.15f')
