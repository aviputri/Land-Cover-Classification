a_ = gdal.Open('/Volumes/ga87rif/Study Project/satelite images/2017/Sentinel 2/clip_b8.tif', gdal.GA_ReadOnly)
a = a_.GetRasterBand(1).ReadAsArray()

b_ = gdal.Open('/Volumes/ga87rif/Study Project/satelite images/2017/Sentinel 2/clip_b11.tif', gdal.GA_ReadOnly)
b = b_.GetRasterBand(1).ReadAsArray()