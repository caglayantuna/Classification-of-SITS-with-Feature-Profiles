# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:26:25 2019

@author: caglayan
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 15:24:35 2019

@author: caglayan
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from functions import *
import siamxt 
from sklearn.metrics import f1_score
import pickle
from sklearn.model_selection import GridSearchCV

def mean_gray_image(imarray):
    Bc = np.ones((3, 3), dtype=bool)
    r, c, b = tuple(imarray.shape)
    mean_img = np.zeros([r, c, b], dtype=float)
    for i in range(b):
        mxt = siamxt.MaxTreeAlpha(imarray[:, :, i], Bc)
        mean=mxt.computeNodeGrayAvg()
        mean_img[:, :, i] = mean[mxt.node_index]
        mean_img=np.array(mean_img,dtype=np.uint16)
    return mean_img
def volume_image(imarray):
    Bc = np.ones((3, 3), dtype=bool)
    r, c, b = tuple(imarray.shape)
    volume_img = np.zeros([r, c, b], dtype=float)
    for i in range(b):
        mxt = siamxt.MaxTreeAlpha(imarray[:, :, i], Bc)
        volume=mxt.computeVolume()
        volume_img[:, :, i] =volume[mxt.node_index]
        volume_img=np.array(volume_img,dtype=np.uint16)
    return volume_img
def height_image(imarray):
    Bc = np.ones((3, 3), dtype=bool)
    r, c, b = tuple(imarray.shape)
    height_img = np.zeros([r, c, b], dtype=float)
    for i in range(b):
        mxt = siamxt.MaxTreeAlpha(imarray[:, :, i], Bc)
        height=mxt.computeHeight()
        height_img[:, :, i] =height[mxt.node_index]
        height_img=np.array(height_img,dtype=np.uint16)
    return height_img
def area_image(imarray):
    Bc = np.ones((3, 3), dtype=bool)
    r, c, b = tuple(imarray.shape)
    area_img = np.zeros([r, c, b], dtype=float)
    for i in range(b):
        mxt = siamxt.MaxTreeAlpha(imarray[:, :, i], Bc)
        area=mxt.node_array[3,:]
        area_img[:, :, i] =area[mxt.node_index]
        #area_img=np.array(area_img,dtype=np.uint16)
    return area_img
def data_prepare(gt,input):
    #class index
    firstclass=1
    secondclass = 2
    thirdclass = 3
    forthtclass = 4
    fifthclass = 5
    #coordinates
    firstindices = np.where(gt == firstclass)
    secondindices = np.where(gt == secondclass)
    thirdindices = np.where(gt == thirdclass)
    forthindices = np.where(gt == forthtclass)
    fifthindices = np.where(gt == fifthclass)

    #data
    testone=input[firstindices[0],firstindices[1],:]
    testtwo = input[secondindices[0], secondindices[1],:]
    testthree = input[thirdindices[0],thirdindices[1],:]
    testfour = input[forthindices[0],forthindices[1],:]
    testfive = input[fifthindices[0],fifthindices[1],:]

    test = np.concatenate((testone,testtwo,testthree,testfour,testfive))

    # test labels
    testlabelone = np.full((testone.shape[0]), 1, dtype=np.uint8)
    testlabeltwo = np.full((testtwo.shape[0]), 2, dtype=np.uint8)
    testlabelthree = np.full((testthree.shape[0]), 3, dtype=np.uint8)
    testlabelfour = np.full((testfour.shape[0]), 4 ,dtype=np.uint8)
    testlabelfive = np.full((testfive.shape[0]), 5, dtype=np.uint8)

    testlabel = np.concatenate((testlabelone,testlabeltwo,testlabelthree,testlabelfour,testlabelfive))

    return test,testlabel
def RFclassification(train,test,trainlabel,testlabel):
    clf = RandomForestClassifier()
    param_grid = {
                 'n_estimators': [10, 20, 30],
             }
    grid_clf = GridSearchCV(clf, param_grid, cv=5)
    grid_clf.fit(train, trainlabel)
    y_pred = grid_clf.predict(test)
    #best_grid = grid_search.best_estimator_
    #y_pred=best_grid.predict(test_features)
    print("Accuracy:", metrics.accuracy_score(testlabel, y_pred))
    print(f1_score(testlabel, y_pred, average=None)) 
    filename = 'finalized_model.sav'
    pickle.dump(grid_clf, open(filename, 'wb'))
if __name__ == "__main__":
    Image = geoimread('data/gtdordogne.tif')


    gt = geoImToArray(Image)
    gt = gt.astype(np.uint8)
    gt=gt[:,:,0]
    gt=gt[500:1500,500:1500]

    Image = geoimread('data/ndvimergeddordogne.tif')    
    imarray = geoImToArray(Image)
    imarray=imarray[500:1500,500:1500,:]

    #train and test
    imarraytrain= imarray[:,0:480,:] 
    imarraytest=imarray[:,520:,:]
    gttrain=gt[:,0:480]  
    gttest=gt[:,520:]
  

    #feature profile train
    #area
    featmax=area_image(imarraytrain)
    featmin=area_image(imarraytrain.max()-imarraytrain)
    #height
    #featmax=height_image(imarraytrain)
    #featmin=height_image(imarraytrain.max()-imarraytrain)
    #volume
    #volume=volume_image(imarraytrain)
    #volumemin=volume_image(imarraytrain.max()-imarraytrain)
    #mean
    #mean=mean_gray_image(imarraytrain)
    #meanmin=mean_gray_image(imarraytrain.max()-imarraytrain)
    
    fptrain= np.concatenate((imarraytrain,featmax,featmin),axis=2)
    train, trainlabel=data_prepare(gttrain, fptrain)
    
    #feature profile test
    #area
    featmax=area_image(imarraytest)
    featmin=area_image(imarraytest.max()-imarraytest)
    #height
    #featmax=height_image(imarraytest)
    #featmin=height_image(imarraytest.max()-imarraytest)
    #volume
    #volume=volume_image(imarraytest)
    #volumemin=volume_image(imarraytest.max()-imarraytest)
    #mean
    #mean=mean_gray_image(imarraytest)
    #meanmin=mean_gray_image(imarraytest.max()-imarraytest)
    fptest= np.concatenate((imarraytest,featmax,featmin),axis=2)
    test, testlabel=data_prepare(gttest, fptest)
    #classification
    
    RFclassification(train, test, trainlabel, testlabel)  
