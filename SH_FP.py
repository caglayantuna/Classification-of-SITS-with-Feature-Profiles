# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 17:33:58 2019

@author: caglayan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 20:22:39 2019

@author: caglayantuna
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 15:24:35 2019

@author: caglayan
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from spatial_extent_func import *
from project_functions import *
import siamxt 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

def area_image_img(imarray):
    Bc = np.ones((3, 3), dtype=bool)

    mxt = siamxt.MaxTreeAlpha(imarray,Bc)
    area=mxt.node_array[3,:]
    area_img=area[mxt.node_index]
    area_img=np.reshape(area_img,[area_img.shape[0],area_img.shape[1],1])
    return area_img
def volume_img(imarray):
    Bc = np.ones((3, 3), dtype=bool)
    mxt = siamxt.MaxTreeAlpha(imarray, Bc)
    volume=mxt.computeVolume()
    volume_img =volume[mxt.node_index]
    volume_img=np.reshape(volume_img,[volume_img.shape[0],volume_img.shape[1],1])
    return volume_img
def mean_gray_img(imarray):
    Bc = np.ones((3, 3), dtype=bool)
    mxt = siamxt.MaxTreeAlpha(imarray, Bc)
    mean=mxt.computeNodeGrayAvg()
    mean_img =mean[mxt.node_index]
    mean_img=np.reshape(mean_img,[mean_img.shape[0],mean_img.shape[1],1])
    return mean_img
def height_gray_img(imarray):
    Bc = np.ones((3, 3), dtype=bool)
    mxt = siamxt.MaxTreeAlpha(imarray, Bc)
    mean=mxt.computeHeight()
    height_img =mean[mxt.node_index]
    height_img=np.reshape(height_img,[height_img.shape[0],height_img.shape[1],1])
    return height_img

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
    best_grid = grid_clf.best_estimator_
    y_pred=best_grid.predict(test)
    print(f1_score(testlabel, y_pred, average=None)) 
    print("Accuracy:", metrics.accuracy_score(testlabel, y_pred))
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
    
    #spatial hierarchy train
    imsingletrain=lexsortnew(imarraytrain)
    imsingletrain=im_normalize(imsingletrain,16)
    #imsingletrain=meanSITS(imarraytrain)
    #imsingletrain=dtw_image(imarraytrain)
    imsingletrain= imsingletrain.astype(np.uint16)
    
    #spatial hierarchy test
    imsingletest=lexsortnew(imarraytest)
    imsingletest=im_normalize(imsingletest,16)
    #imsingletest=meanSITS(imarraytest)
    #imsingletest=dtw_image(imarraytest)
    imsingletest= imsingletest.astype(np.uint16)


    #feature profile
    #area
    featmax=area_image_img(imsingletrain)
    featmin=area_image_img(imsingletrain.max()-imsingletrain)
    #height
    #featmax=height_gray_img(imsingletrain)
    #featmax=height_gray_img(imsingletrain)
    #featmin=height_gray_img(imsingletrain.max()-imsingletrain)
    #volume
    #featmax=volume_img(imsingletrain)
    #featmin=volume_img(imsingletrain.max()-imsingletrain)
    #mean
    #featmax=mean_gray_img(imsingletrain)
    #featmin=mean_gray_img(imsingletrain.max()-imsingletrain)
    
    imsingletrainup=np.reshape(imsingletrain,[imsingletrain.shape[0],imsingletrain.shape[1],1])
    fptrain= np.concatenate((imsingletrainup,featmax,featmin),axis=2)
    train, trainlabel=data_prepare(gttrain, fptrain)
    
    #feature profile
    #area
    featmax=area_image_img(imsingletest)
    featmin=area_image_img(imsingletest.max()-imsingletest)
    #height
    #featmax=height_gray_img(imsingletest)
    #featmin=height_gray_img(imsingletest.max()-imsingletest)  
    #volume
    #featmax=volume_img(imsingletest)
    #featmin=volume_img(imsingletest.max()-imsingletest)
    #mean
    #featmax=mean_gray_img(imsingletest)
    #featmin=mean_gray_img(imsingletest.max()-imsingletest)
    
    imsingletestup=np.reshape(imsingletest,[imsingletest.shape[0],imsingletest.shape[1],1])
    fptest= np.concatenate((imsingletestup,featmax,featmin),axis=2)
    test, testlabel=data_prepare(gttest, fptest)
    #classification
    
    RFclassification(train, test, trainlabel, testlabel)  
