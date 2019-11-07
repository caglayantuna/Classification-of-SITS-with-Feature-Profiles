from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from functions import *
import siamxt 
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV


def area_image_cube(imarray):
    Bc = np.zeros((3,3,3), dtype = bool)
    Bc[1,1,:] = True
    Bc[:,1,1] = True 
    Bc[1,:,1] = True

    mxt = siamxt.MaxTreeAlpha(imarray,Bc)
    area=mxt.node_array[3,:]
    area_img=area[mxt.node_index]
    return area_img
def volume_cube(imarray):
    Bc = np.ones((3,3,3), dtype=bool)
    mxt = siamxt.MaxTreeAlpha(imarray, Bc)
    volume=mxt.computeVolume()
    volume_img =volume[mxt.node_index]
    return volume_img
def mean_gray_cube(imarray):
    Bc = np.ones((3,3,3), dtype=bool)
    mxt = siamxt.MaxTreeAlpha(imarray, Bc)
    mean=mxt.computeNodeGrayAvg()
    mean_img =mean[mxt.node_index]
    return mean_img
def height_gray_cube(imarray):
    Bc = np.ones((3,3,3), dtype=bool)
    mxt = siamxt.MaxTreeAlpha(imarray, Bc)
    mean=mxt.computeHeight()
    height_img =mean[mxt.node_index]
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
    #sixthindices = np.where(gt == sixthclass)
    #seventhindices = np.where(gt == seventhclass)

    #data
    testone=input[firstindices[0],firstindices[1],:]
    testtwo = input[secondindices[0], secondindices[1],:]
    testthree = input[thirdindices[0],thirdindices[1],:]
    testfour = input[forthindices[0],forthindices[1],:]
    testfive = input[fifthindices[0],fifthindices[1],:]
    #testsix = input[fifthindices[0],fifthindices[1],:]
    #testseven = input[fifthindices[0],fifthindices[1],:]

    test = np.concatenate((testone,testtwo,testthree,testfour,testfive))

    # test labels
    testlabelone = np.full((testone.shape[0]), 1, dtype=np.uint8)
    testlabeltwo = np.full((testtwo.shape[0]), 2, dtype=np.uint8)
    testlabelthree = np.full((testthree.shape[0]), 3, dtype=np.uint8)
    testlabelfour = np.full((testfour.shape[0]), 4 ,dtype=np.uint8)
    testlabelfive = np.full((testfive.shape[0]), 5, dtype=np.uint8)
    #testlabelsix = np.full((testsix.shape[0]), 6, dtype=np.uint8)
    #testlabelseven = np.full((testseven.shape[0]), 7, dtype=np.uint8)

    testlabel = np.concatenate((testlabelone,testlabeltwo,testlabelthree,testlabelfour,testlabelfive))

    return test,testlabel
def RFclassification(train,test,trainlabel,testlabel):
    clf = RandomForestClassifier()
    param_grid = {
                 'n_estimators': [100, 200, 300],
                     'max_depth': [80, 90, 100],
             }
    grid_clf = GridSearchCV(clf, param_grid, cv=5)
    grid_clf.fit(train, trainlabel)
    best_grid = grid_clf.best_estimator_
    y_pred=best_grid.predict(test)
    print("Accuracy:", metrics.accuracy_score(testlabel, y_pred))
    print(f1_score(testlabel, y_pred, average=None)) 
    
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
  

    #feature profile
    #area
    featmax=area_image_cube(imarraytrain)
    featmin=area_image_cube(imarraytrain.max()-imarraytrain)
    #height
    #featmax=height_gray_cube(imarraytrain)
    #featmin=height_gray_cube(imarraytrain.max()-imarraytrain)
    #volume
    #featmax=volume_cube(imarraytrain)
    #featmin=volume_cube(imarraytrain.max()-imarraytrain)
    #mean
    #featmax=mean_gray_cube(imarraytrain)
    #featmin=mean_gray_cube(imarraytrain.max()-imarraytrain)
    
    fptrain= np.concatenate((imarraytrain,featmax,featmin),axis=2)
    train, trainlabel=data_prepare(gttrain, fptrain)
    
    #feature profile
    #area
    featmax=area_image_cube(imarraytest)
    featmin=area_image_cube(imarraytest.max()-imarraytest)
    #height
    #featmax=height_gray_cube(imarraytest)
    #featmin=height_gray_cube(imarraytest.max()-imarraytest)
    #volume
    #featmax=volume_cube(imarraytest)
    #featmin=volume_cube(imarraytest.max()-imarraytest)
    #mean
    #featmax=mean_gray_cube(imarraytest)
    #featmin=mean_gray_cube(imarraytest.max()-imarraytest)
    fptest= np.concatenate((imarraytest,featmax,featmin),axis=2)
    test, testlabel=data_prepare(gttest,fptest)
    #classification
    
    RFclassification(train, test, trainlabel, testlabel)  
