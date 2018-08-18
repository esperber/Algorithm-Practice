import numpy as np


A = np.array([[1,2], [3,4], [5,6]])
print(A)



def PCA(A):
    #Get mean of all the features/columns
    M = np.mean(A.T, axis=1)

    #Subtract mean of columns from each column
    #Center columns
    C = A - M


    #Covariance describes how two variables change together
    #Multiplication of the sum of differences of the X values, with its expected values
    #and the sum of difference of the Y values, with its expected values
    # multiplied by the reciprocal of # of samples


    V = np.cov(C.T)

    #Take the co-variance matrix and put it into the Eigenvector
    values, vectors = np.linalg.eig(V)
    #print(vectors)
    final = vectors.T.dot(C.T)

    return final

print(PCA(final))