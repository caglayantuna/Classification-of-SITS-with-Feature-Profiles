from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from spatial_extent_func import *
from project_functions import *
import siamxt 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

def area_image_cube(imarray):
    Bc = np.ones((3, 3), dtype=bool)

    mxt = siamxt.MaxTreeAlpha(imarray,Bc)
    area=mxt.node_array[3,:]
    area_img=area[mxt.node_index]
    area_img=np.reshape(area_img,[area_img.shape[0],area_img.shape[1],1])
    return area_img
def volume_cube(imarray):
    Bc = np.ones((3, 3), dtype=bool)
    mxt = siamxt.MaxTreeAlpha(imarray, Bc)
    volume=mxt.computeVolume()
    volume_img =volume[mxt.node_index]
    volume_img=np.reshape(volume_img,[volume_img.shape[0],volume_img.shape[1],1])
    return volume_img
def mean_gray_cube(imarray):
    Bc = np.ones((3, 3), dtype=bool)
    mxt = siamxt.MaxTreeAlpha(imarray, Bc)
    mean=mxt.computeNodeGrayAvg()
    mean_img =mean[mxt.node_index]
    mean_img=np.reshape(mean_img,[mean_img.shape[0],mean_img.shape[1],1])
    return mean_img
def height_gray_cube(imarray):
    Bc = np.ones((3, 3), dtype=bool)
    mxt = siamxt.MaxTreeAlpha(imarray, Bc)
    mean=mxt.computeHeight()
    height_img =mean[mxt.node_index]
    height_img=np.reshape(height_img,[height_img.shape[0],height_img.shape[1],1])
    return height_img

def data_prepare(gt,input):
   # class colors values for toulouse
    #firstclass=7
    #secondclass = 13
    #thirdclass = 16
    #forthtclass = 8
    #fifthclass = 6
    
    #class colors values for morbihan
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
    #y_pred = grid_clf.predict(test)
    best_grid = grid_clf.best_estimator_
    y_pred=best_grid.predict(test)
    print(f1_score(testlabel, y_pred, average=None)) 
    print("Accuracy:", metrics.accuracy_score(testlabel, y_pred))
if __name__ == "__main__":
    Image = geoimread('data/gtdordogne.tif')


    gt = geoImToArray(Image)
    gt = gt.astype(np.uint8)
    gt=gt[500:1500,500:1500,0]
    Image = geoimread('data/ndvimergeddordogne.tif')    
    imarray = geoImToArray(Image)
    imarray=imarray[500:1500,500:1500,:]
    #train and test
    imarraytrain= imarray[:,0:480,:] 
    imarraytest=imarray[:,520:,:]
    gttrain=gt[:,0:480]  
    gttest=gt[:,520:]



    train, trainlabel=data_prepare(gttrain, imarraytrain)
    
    test, testlabel=data_prepare(gttest, imarraytest)
    #classification
    
    RFclassification(train, test, trainlabel, testlabel)  
