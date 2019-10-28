import siamxt
import numpy as np



def neighborsof(img,i,j):
    try:
      neighbors = []
      neighbors=[img[i+1,j],img[i,j+1],img[i-1,j],img[i,j-1]]
      return np.array(neighbors)
    except IndexError:
      print('error')
      return np.array(neighbors)
def neighborsOfNeighbors(img,i,j):
    neighbors = neighborsof(img,i,j)
    nOfn=[neighborsof(neighbors[0]),neighborsof(neighbors[1]),neighborsof(neighbors[2]),neighborsof(neighbors[3])]
    return nOfn
def neighborlocation(index):
    index=index[0]
    if index==0:
        x=1
        y=0
    if index==1:
        x=0
        y=1
    if index==2:
        x=-1
        y=0
    if index==3:
        x=0
        y=-1
    return x,y


Bc = np.zeros((3,3,3), dtype = bool)
Bc[1,1,:] = True
Bc[:,1,1] = True
Bc[1,:,1] = True

img1 = np.array([
    [[0,0,0,0],
    [0,5,0,1],
    [0,0,0,3],
    [2,5,4,0]],

    [[0,1,0,0],
    [1,5,0,1],
    [2,5,0,8],
    [0,3,4,0]]],dtype = np.uint8)
mxt1 = siamxt.MaxTreeAlpha(img1,Bc)

img2 = np.array([
    [[0,0,0,0],
    [0,5,0,1],
    [0,0,0,3],
    [2,5,4,0]],

    [[0,1,0,0],
    [1,5,0,1],
    [2,5,0,8],
    [0,3,4,0]],

     [[1,0,0,0],
     [0,5,5,1],
     [2,0,0,3],
     [0,4,4,1]]],dtype = np.uint8)


mxt2 = siamxt.MaxTreeAlpha(img2,Bc)
#your code here    
print (time.clock() - start)

mxt1.node_index=np.concatenate((mxt1.node_index,np.zeros([1,4,4],np.int)),axis=0)
new_img=img2[2,:,:]
start = time.clock()
for i in range(img1[0,:,:].shape[0]):
    for j in range(img1.shape[1]):
         if (mxt1.node_array[2,mxt1.node_index[1,i,j]]==new_img[i,j]) :
            mxt1.node_index[2,i,j]=mxt1.node_index[1,i,j]

         if (mxt1.node_array[2, mxt1.node_index[0, i, j]] == new_img[i, j]):
            mxt1.node_index[2, i, j] = mxt1.node_index[0, i, j]

         if (mxt1.node_array[2, mxt1.node_index[0, i, j]] != new_img[i, j]) and (mxt1.node_array[2, mxt1.node_index[1, i, j]] > new_img[i, j]):
            mxt1.node_index[2, i, j] = mxt1.node_index[1, i, j]

         if (mxt1.node_array[2, mxt1.node_index[1, i, j]] != new_img[i, j]) and (mxt1.node_array[2, mxt1.node_index[0, i, j]] > new_img[i, j]):
            mxt1.node_index[2, i, j] = mxt1.node_index[0, i, j]

         if (mxt1.node_array[2, mxt1.node_index[1, i, j]] < new_img[i, j]) and (mxt1.node_array[2, mxt1.node_index[0, i, j]] < new_img[i, j]) and new_img[i-1,j] < new_img[i, j] and new_img[i+1,j] < new_img[i, j] and new_img[i,j-1] < new_img[i, j] and new_img[i,j+1] < new_img[i, j]:
            mxt1.node_index[2, i, j] = min(mxt1.node_index[1, i, j],mxt1.node_index[0, i, j],new_img[i-1,j],new_img[i+1,j],new_img[i,j-1],new_img[i,j+1])+1
            mxt1.node_index[1, :, :][mxt1.node_index[1, :, :]==mxt1.node_index[2, i, j]]=mxt1.node_index[2, i, j]+1
            mxt1.node_index[0, :, :][mxt1.node_index[0, :, :] == mxt1.node_index[2, i, j]] = mxt1.node_index[2, i, j] + 1

         if  (mxt1.node_array[2, mxt1.node_index[1, i, j]] != new_img[i, j]) and (mxt1.node_array[2, mxt1.node_index[0, i, j]] != new_img[i, j]) and new_img[i,j] in neighborsof(new_img,i,j):
            x,y=neighborlocation(np.where(neighborsof(new_img,i,j)==new_img[i,j]))
            mxt1.node_index[2, i, j]=mxt1.node_index[2, i+x, j+y]


node_array1=mxt1.node_index

node_array2=mxt2.node_index


