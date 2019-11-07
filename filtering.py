import siamxt
from project_functions import *
from scipy import stats
from scipy.misc import imsave

def im_show(a):
    plt.figure()
    plt.imshow(a, cmap='gray')
    plt.show()

Image1 = geoimread('data/phr1.png')
imarray1=geoImToArray(Image1)
imarray1=np.array(imarray1,dtype=np.uint8)
im1=np.reshape(imarray1,[imarray1.shape[0],imarray1.shape[1]])



Image2 = geoimread('data/phr2.png')
imarray2=geoImToArray(Image2)
imarray2=np.array(imarray2,dtype=np.uint8)
im2=np.reshape(imarray2,[imarray2.shape[0],imarray2.shape[1]])


Image3 = geoimread('data/phr3.png')
imarray3=geoImToArray(Image3)
imarray3=np.array(imarray3,dtype=np.uint8)
im3=np.reshape(imarray3,[imarray3.shape[0],imarray3.shape[1]])

im_show(im1)
im_show(im2)
im_show(im3)

merged=np.concatenate((imarray1,imarray2,imarray3),axis=2)
#max tree 
Bc = np.ones((3,3,3), dtype=bool)
tree1 = siamxt.MaxTreeAlpha(merged, Bc)
t=1000 #threshold
a=attribute_area_filter(tree1,t)


im_show(a[:,:,0])
im_show(a[:,:,1])
im_show(a[:,:,2])

imsave('results/orig1.png',im1)
imsave('results/orig2.png',im2)
imsave('results/orig3.png',im3)

imsave('results/filt1.png',a[:,:,0])
imsave('results/filt2.png',a[:,:,1])
imsave('results/filt3.png',a[:,:,2])


#th strategy
bc = np.ones((3,3), dtype=bool)
tree1 = siamxt.MaxTreeAlpha(im1, bc)
tree2 = siamxt.MaxTreeAlpha(im2, bc)
tree3 = siamxt.MaxTreeAlpha(im3, bc)

b=attribute_area_filter(tree1,t)
c=attribute_area_filter(tree2,t)
d=attribute_area_filter(tree3,t)

imsave('results/filt1th.png',b)
imsave('results/filt2th.png',c)
imsave('results/filt3th.png',d)


#color composition
#merged=np.concatenate((imarray1[1500:1700,2000:2200],imarray2[1500:1700,2000:2200],imarray3[1500:1700,2000:2200]),axis=2)
#merged=np.concatenate((b,c,d),axis=2)
imsave('results/origcmerged.png',merged) #original
imsave('sthmerged.png',a)  #sth composite


mergedth=np.concatenate((np.reshape(b,[b.shape[0],b.shape[1],1]),np.reshape(c,[c.shape[0],c.shape[1],1]),np.reshape(d,[d.shape[0],d.shape[1],1])),axis=2)
imsave('results/thmerged.png',mergedth)




#tophat

imsave('differencesth.png',merged-a)
('differenceth.png',merged-np.reshape(b,[b.shape[0],b.shape[1],1]),np.reshape(c,[c.shape[0],c.shape[1],1]),np.reshape(d,[d.shape[0],d.shape[1],1]))
