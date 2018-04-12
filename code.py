import numpy as np
from numpy import linalg as LA
from scipy.stats import norm


#Finds Centalized matrix and reurns it
def center_mat(x):
    ans1 = []
    for x1 in x:
        ans2 = []
        mean = np.mean(x1)
        for x2 in x1:
            ans2.append(x2-mean)
        ans1.append(ans2)
    return ans1


#Prints Top PC and projection of data on that PC
def pca():
    matrix = [
            [-2, 1, 4, 6, 5, 3, 6, 2],
            [9, 3, 2, -1, -4, -2, -4, 5],
            [0, 7, -5, 3, 2, -3, 4, 6]
            ]


    #Centralize Matrix
    matrix = center_mat(matrix)

    #Creates covariance matrix usin C = X*transpose(X)
    covariance_mat = np.matmul(matrix,np.transpose(matrix))

    #Find Eigenvectors of covarience matrix
    w, v = LA.eig(covariance_mat)
    print "*****Top PC*****"
    max_eigen = w.argmax()
    print v[:,max_eigen]

    #Projection B = traspose(U)*Centered matrix
    B = np.matmul(np.transpose(v[:,max_eigen]),matrix)
    print "******Projection on Top PC******"
    print B


#Prints LDA direction and Projection of Data on LDA and optimal classifier
def lda():
    positive_mat = [
        [4, 2, 2, 3, 4, 6, 3, 8],
        [1, 4, 3, 6, 4, 2, 2, 3],
        [0, 1, 1, 0, -1, 0, 1, 0],
    ]
    negative_mat = [
        [9, 6, 9, 8, 10],
        [10, 8, 5, 7, 8],
        [1, 0, 0, 1, -1]
    ]

    #Finding Sb
    #finds u+
    positive_u = []
    for x in positive_mat:
        positive_u.append(np.mean(x))
    positive_u = np.reshape(positive_u,(len(positive_u),1))

    #finds u-
    negative_u = []
    for x in negative_mat:
        negative_u.append(np.mean(x))
    negative_u=np.reshape(negative_u, (len(negative_u), 1))

    #finds u+ - u-
    tmp = np.subtract(positive_u,negative_u)

    #finds sb = (u+ - u-) * tanspose(u+ - u-)
    sb = np.matmul(tmp, np.transpose(tmp))

    #Now finding Sw
    # Centralize Input data Matrix
    positive_mat_c = center_mat(positive_mat)
    negative_mat_c = center_mat(negative_mat)

    # Creates covariance matrix usin C = X*transpose(X)
    p_covariance_mat = np.matmul(positive_mat_c, np.transpose(positive_mat_c))
    n_covariance_mat = np.matmul(negative_mat_c, np.transpose(negative_mat_c))

    nplus = len(positive_mat[0])
    nminus = len(negative_mat[0])
    n = nplus+nminus

    #find Sw
    sw = (float(nplus)/float(n))*np.array(p_covariance_mat) + (float(nminus)/float(n))*np.array(n_covariance_mat)

    w, v = LA.eig(np.matmul(np.linalg.inv(sw), sb))
    max_eigen = w.argmax()

    print "********Projection direction for LDA**********"
    print v[:,max_eigen]

    wlda = v[:,max_eigen]
    wldaT = np.reshape(wlda,(len(wlda),1))
    print "********Projection of Positive data on LDA**********"
    ppd = np.matmul(np.transpose(positive_mat), wldaT)
    print ppd
    print "********Projection of Negative data on LDA**********"
    pnd = np.matmul(np.transpose(negative_mat), wldaT)
    print pnd

    print "*********Optimal classifier***********"
    threshold_lda = (np.mean(ppd)+np.mean(pnd))/2.0
    print threshold_lda


pca()
lda()
