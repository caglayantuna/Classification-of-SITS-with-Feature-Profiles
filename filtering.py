import siamxt
from project_functions import *
from scipy import stats
from scipy.misc import imsave

def im_show(a):
    plt.figure()
    plt.imshow(a, cmap='gray')
    plt.show()
a=1500
b=1800
c=2300
d=2600
Image1 = geoimread('20180507clipped.tif')
imarray1=geoImToArray(Image1)
imarray1[imarray1>255]=255
imarray1=np.array(imarray1,dtype=np.uint8)
im1=np.reshape(imarray1,[imarray1.shape[0],imarray1.shape[1]])
im1=im1[a:b,c:d]


Image2 = geoimread('20180623clipped.tif')
imarray2=geoImToArray(Image2)
imarray2[imarray2>255]=255
imarray2=np.array(imarray2,dtype=np.uint8)
im2=np.reshape(imarray2,[imarray2.shape[0],imarray2.shape[1]])
im2=im2[a:b,c:d]


Image3 = geoimread('20180806clipped.tif')
imarray3=geoImToArray(Image3)
imarray3[imarray3>255]=255
imarray3=np.array(imarray3,dtype=np.uint8)
im3=np.reshape(imarray3,[imarray3.shape[0],imarray3.shape[1]])
im3=im3[a:b,c:d]

im_show(im1)
im_show(im2)
im_show(im3)

merged=np.concatenate((imarray1[a:b,c:d],imarray2[a:b,c:d],imarray3[a:b,c:d]),axis=2)
#max tree 
Bc = np.ones((3,3,3), dtype=bool)
tree1 = siamxt.MaxTreeAlpha(merged, Bc)
t=800 #threshold
a=attribute_area_filter(tree1,t)


im_show(a[:,:,0])
im_show(a[:,:,1])
im_show(a[:,:,2])

imsave('orig1.png',im1)
imsave('orig2.png',im2)
imsave('orig3.png',im3)

imsave('filt1.png',a[:,:,0])
imsave('filt2.png',a[:,:,1])
imsave('filt3.png',a[:,:,2])


#th strategy
bc = np.ones((3,3), dtype=bool)
tree1 = siamxt.MaxTreeAlpha(im1, bc)
tree2 = siamxt.MaxTreeAlpha(im2, bc)
tree3 = siamxt.MaxTreeAlpha(im3, bc)

b=attribute_area_filter(tree1,t)
c=attribute_area_filter(tree2,t)
d=attribute_area_filter(tree3,t)

imsave('filt1th.png',b)
imsave('filt2th.png',c)
imsave('filt3th.png',d)


#color composition
#merged=np.concatenate((imarray1[1500:1700,2000:2200],imarray2[1500:1700,2000:2200],imarray3[1500:1700,2000:2200]),axis=2)
#merged=np.concatenate((b,c,d),axis=2)
imsave('origcmerged.png',merged) #original
imsave('sthmerged.png',a)  #sth composite


mergedth=np.concatenate((np.reshape(b,[b.shape[0],b.shape[1],1]),np.reshape(c,[c.shape[0],c.shape[1],1]),np.reshape(d,[d.shape[0],d.shape[1],1])),axis=2)
imsave('thmerged.png',mergedth)
