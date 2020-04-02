import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import init_centroids
import sys

from PIL import Image
from scipy.misc import imread, imsave
from init_centroids import init_centroids
from scipy.spatial.distance import euclidean

#printing method given to us
def print_cent(cent):
    if type(cent) == list:
        cent = np.asarray(cent)
    if len(cent.shape) == 1:
        return ' '.join(str(np.floor(100*cent)/100).split()).replace('[ ', '[').replace('\n', ' ').replace(' ]',']').replace(' ', ', ')
    else:
        return ' '.join(str(np.floor(100*cent)/100).split()).replace('[ ', '[').replace('\n', ' ').replace(' ]',']').replace(' ', ', ')[1:-1]

#finding out the centroid place witch is closest to the point
def minDistFromPoint(point, List):
    L = []
    for i in List:
        L.append(euclidean(i, point))
    L = np.asarray(L)
    return L.argmin()

#gets the centroids already and the pixels array
def kMeans(centroids, pixels):
    #running for 10 iterations
    for i in range(0, 11):
        #printing as needed
        print("iter " + str(i) + ": ", end="")
        print(print_cent(centroids))
        #creating cluster list to save each point in its place
        #according to where it was closer to
        clusters = [[]] * len(centroids)
        #main loop we check for closest and putting the points in their place
        for pix in pixels:
            clusters[minDistFromPoint(pix, centroids)] = np.append(clusters[minDistFromPoint(pix, centroids)], pix)
        for n in range(0, len(centroids)):
            clusters[n] = np.reshape(clusters[n], (-1, 3))
        #creating new centroids with the avarage function
        for j in range(0, len(centroids)):
            centroids[j] = np.average(clusters[j], axis=0)
    #return the new picture array to be displayed
    newpic = []
    for pix in pixels:
        newpic.append(centroids[minDistFromPoint(pix, centroids)])
    return np.asarray(newpic)


# data preperation (loading, normalizing, reshaping)
path = 'dog.jpeg'
A = imread(path)
A = A.astype(float) / 255.
img_size = A.shape
X = A.reshape(img_size[0] * img_size[1], img_size[2])
ks = [2, 4, 8, 16]
for k in ks:
    print("k=" + str(k) + ":")
    out = kMeans(init_centroids(X, k), X)
    '''
    #the lines needed to add to show the image
    out = out.reshape(128, 128, 3)
    plt.imshow(out)
    plt.grid(False)
    plt.show()
    '''