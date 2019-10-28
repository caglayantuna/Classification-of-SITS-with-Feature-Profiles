import gdal
import numpy as np
from osgeo import ogr
import matplotlib.pyplot as plt
from scipy.misc import imread as imread



def geoimread(filename):
    Image = gdal.Open(filename)
    return Image

def geoImToArray(Image):
    row=Image.RasterYSize
    col = Image.RasterXSize
    band = Image.RasterCount

    imarray = np.zeros([row, col, band], dtype=np.uint16)
    for band in range(band):
        imarray[:, :, band] = np.array(Image.GetRasterBand(band + 1).ReadAsArray())
    return imarray

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
    dist_img = np.zeros([r, c], dtype=float)
    for x in range(r):
        for y in range(c):
            dist=np.max(imarray[x, y, :])-np.min(imarray[x, y, :])
            dist_img[x, y] = dist
    return dist_img
def quartilecoeff(imarray):
    r, c, d = imarray.shape
    quartilecoefficient = np.zeros([r, c], dtype=float)
    for x in range(r):
        for y in range(c):
            Q1=np.percentile(imarray[x, y, :],25)
            Q3 = np.percentile(imarray[x, y, :], 75)
            quartilecoefficient[x, y]=(Q3-Q1)/(Q3+Q1)
    return quartilecoefficient
def entropySITS(imarray):
    r, c, d = imarray.shape
    entropy_image = np.zeros([r, c], dtype=float)
    for x in range(r):
        for y in range(c):
            value, counts = np.unique(imarray[x, y, :],return_counts=True)
            norm_counts = counts / counts.sum()
            entropy = -(norm_counts * np.log(norm_counts))
            entropy_image[x,y]=entropy
    return entropy_image
def interqrange(imarray):
    r, c, d = imarray.shape
    dist_img = np.zeros([r, c], dtype=float)
    for x in range(r):
        for y in range(c):
            q75, q25 = np.percentile(imarray[x,y], [75, 25])
            iqr = q75 - q25
            dist_img[x, y] = iqr
    return dist_img
def shape_feature_read(filename,feature):
    file = ogr.Open(filename)
    shape = file.GetLayer(0)
    feature = shape.GetFeature(feature)
    first = feature.ExportToJson()
    print (first)
    return first
def writenodeindex(mxt,data,filename):
    result = mxt.node_index
    result = np.array(result, dtype=np.float32)

    [cols, rows] = result.shape
    trans = data.GetGeoTransform()
    proj = data.GetProjection()
    outfile = filename
    outdriver = gdal.GetDriverByName("GTiff")
    outdata = outdriver.Create(str(outfile), rows, cols, 1, gdal.GDT_Float32)
    # Write the array to the file, which is the original array in this example
    outdata.SetProjection(proj)
    outdata.SetGeoTransform(trans)
    outdata.GetRasterBand(1).WriteArray(result)
    outdata.FlushCache()

def max_tree_example(node,mxt):
    A = mxt.node_array[3, :]

    anc = mxt.getAncestors(int(node))[::-1]

    area = A[anc]

    gradient = area[0:-1] - area[1:]
    indexes = np.argsort(gradient)

    max1 = indexes[-1]
    anc_max1 = [anc[max1], anc[max1 + 1]]

    result = mxt.recConnectedComponent(anc_max1[0])

    result = result + 1
    result = result - 1
    result = result * 255
    return result
def attribute_area_filter(mxt,area):
    mxt.areaOpen(area)
    ao = mxt.getImage()


    return ao
def array_to_raster(array,data,filename):
    rows = []
    cols = []
    band = []
    if   array.ndim==2:
          [cols, rows]= array.shape
          band=1
    elif array.ndim==3:
         [cols, rows, band] = array.shape

    trans = data.GetGeoTransform()
    proj = data.GetProjection()
    outfile = filename
    outdriver = gdal.GetDriverByName("GTiff")
    outdata = outdriver.Create(str(outfile), rows, cols, band, gdal.GDT_Int16)
    # Write the array to the file, which is the original array in this example
    outdata.SetProjection(proj)
    outdata.SetGeoTransform(trans)
    if array.ndim == 2:
        outdata.GetRasterBand(1).WriteArray(array)
    elif array.ndim == 3:
      for b in range(band):
        outdata.GetRasterBand(b + 1).WriteArray(array[:, :,b])
    outdata.FlushCache()
def max_tree_signature(filename,node):

    a = imread(filename)
    Bc = np.ones((3, 3), dtype=bool)
    mxt = siamxt.MaxTreeAlpha(a, Bc)
    result = max_tree_example(node, mxt)
    return result
