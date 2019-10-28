from numpy import zeros, inf
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import tslearn.metrics as ts

def dtw(x, y, d=euclidean_distances, warp=1):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.
    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    :param int warp: how many shifts are computed.
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]  # view
    for i in range(r):
        for j in range(c):
            D1[i, j] = d(x[i], y[j])
    C = D1.copy()#cost matrix
    for i in range(r):
        for j in range(c):
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                i_k = min(i + k, r - 1)
                j_k = min(j + k, c - 1)
                min_list += [D0[i_k, j], D0[i, j_k]]
            D1[i, j] += min(min_list)

    return D1[-1, -1] / sum(D1.shape)
def dtw_SITS(image):
    r, c, b = tuple(image.shape)
    ref=image[0,0,:]
    dtw_image=np.zeros([r, c], dtype=float)
    for i in range(r):
        for j in range(c):
            dtw_image[i, j] = dtw(ref,image[i,j,:])
    dtw_image=(dtw_image/dtw_image.max())*255.0

    return dtw_image
def meanSITS(imarray):
    r, c, d = imarray.shape
    mean_img = np.zeros([r, c], dtype=float)
    for x in range(r):
        for y in range(c):
            mean = np.mean(imarray[x, y, :])
            mean_img[x, y] = mean
    mean_img=np.array(mean_img,dtype=np.uint16)
    return mean_img
def stdSITS(imarray):
    r, c, d = imarray.shape
    std_img = np.zeros([r, c], dtype=float)
    for x in range(r):
        for y in range(c):
            std = np.std(imarray[x, y, :])
            std_img[x, y] = std

    return std_img
def distanceSITS(imarray):
    r, c, d = imarray.shape
    distance_img = np.zeros([r, c], dtype=float)
    for x in range(r):
        for y in range(c):
            distance= imarray[x, y, :].max()-imarray[x, y, :].min()
            distance_img[x, y] = distance

    return distance_img

def lexsort(imarray):
    r, c, d = imarray.shape
    lex_img1 = np.zeros([r, c], dtype=float)
    for x in range(r):
        for y in range(c):
            #for b in range(d):
              #value = imarray[x, y, b]
              #if len(str(value))==1:
              #  imarray[x, y, b]=value*100 #basina 0 eklemek gerekiyor
              #if len(str(value))==2:
              #   imarray[x, y, b]=value * 10
          lex_img1[x, y] = "".join(map(str,imarray[x,y,:]))
    vector=np.reshape(lex_img1,[r*c])
    temp=np.argsort(vector)
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(vector))
    lex_img=np.reshape(ranks,[r,c])
    return lex_img
def lexsortnew(imarray):
    r, c, d = imarray.shape
    imvector=np.reshape(imarray,[r*c,d])
    transpose=np.matrix.transpose(imvector)
    a=np.argsort((transpose))
    imvectorsorted=np.matrix.transpose(a)
    lex_img=np.reshape(imvectorsorted[:,0],[r,c])#siralanmis 6 band icin sadece ilk bandini aliyorum
    return lex_img
def im_normalize(image,bit):
    image=np.array(image,dtype=float)
    max=image.max()

    imagenew=(image/max)*(2**bit-1)
    return imagenew
def euclidean_distance(imarray):
    r, c, d = imarray.shape
    dist_img = np.zeros([r, c], dtype=float)
    reference1=np.zeros([1, d], dtype=float)
    reference2=[65535, 65535, 65535, 65535, 65535, 65535]
    for x in range(r):
        for y in range(c):
            distance1 = np.linalg.norm(imarray[x,y,:]-reference1)
            #distance2 = np.linalg.norm(imarray[x, y, :] - reference2)
            #distance=min(distance1,distance2)
            dist_img[x, y] = distance1
    #dist_img = np.array(dist_img, dtype=np.uint16)
    return dist_img
def dtw_image(imarray):
    r, c, b = tuple(imarray.shape)
    ref1=[0,0,0,0,0,0]
    ref2=[65535,65535,65535,65535,65535,65535]
    #ref2 = imarray[10, 0, :]
    ref3 = imarray[20, 0, :]
    ref4 = imarray[30, 0, :]
    ref5=imarray[40,0,:]
    ref6=imarray[50,0,:]
    ref7=imarray[60,0,:]
    ref8=imarray[40,0,:]
    ref9=imarray[40,0,:]
    dtw_image=np.zeros([r, c], dtype=float)
    for i in range(r):
        for j in range(c):
            distance1= ts.dtw(ref1,imarray[i,j,:])
            #distance2 = ts.dtw(ref2, imarray[i, j, :])
            #distance3 = ts.dtw(ref3, imarray[i, j, :])
            #distance4 = ts.dtw(ref4, imarray[i, j, :])
            #distance5 = ts.dtw(ref5, imarray[i, j, :])
            #distance6 = ts.dtw(ref6, imarray[i, j, :])
            #distance7 = ts.dtw(ref7, imarray[i, j, :])
            #distance8 = ts.dtw(ref8, imarray[i, j, :])
            #distance9 = ts.dtw(ref9, imarray[i, j, :])
            #distance=min(distance1,distance2,distance3,distance4,distance5)
            dtw_image[i,j]=distance1
    #dtw_image=(dtw_image/dtw_image.max())*255.0
    return dtw_image
def lexsort_reverse(imarray,rank,rankfiltered):
    r, c, d = imarray.shape
    result = np.zeros([r, c,d], dtype=float)
    for x in range(r):
        for y in range(c):
            value=rankfiltered[x,y]
            a,b=np.where(rank==value)[0]
            result[x,y,:]=imarray[a,b,:]     
    return result
