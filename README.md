# Classification-of-SITS-with-Feature-Images

This project inlcues codes for classification of SITS with stacked original NDVI time series and feature images produced by spatial attributes. These codes are used to obtain the results in the paper "Component trees for image sequences and streams" which is under review on Journal Pattern Recognition Letters journal. Data is not available.

Requirements: Siamxt

filtering.py: Ihis file is used to create filtered images with time series.

STH_FP.py: This file is used for classification for time series images. We are adding new features by using space-time tree and spatial attributes.

TH_FP.py: This file is used for classification for time series images. We are building tree for each date and extract spatial features from them.

SH_FP.py: This file is used for classification for time series images.We are building only one tree from time series images and extracting spatial feature from it.

baseline.py: As a baseline, we are using original NDVI pixel values for classification.


