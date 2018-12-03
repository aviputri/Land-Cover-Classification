from osgeo import gdal
from osgeo import gdal_array
import numpy as np
import scipy

#-------------------------------------------------------------------------------------------------
# Open an image
#-------------------------------------------------------------------------------------------------

# Open a GDAL dataset
dataset = gdal.Open('/Users/aviputripertiwi/Documents/TU Munchen/Study Project/Images/LS7_09_stack.tif', gdal.GA_ReadOnly)

print(dataset)


#-------------------------------------------------------------------------------------------------
# Image attributes
#-------------------------------------------------------------------------------------------------
# How many bands does this image have?
num_bands = dataset.RasterCount
print('Number of bands in image: {n}\n'.format(n=num_bands))

# How many rows and columns?
rows = dataset.RasterYSize
cols = dataset.RasterXSize
print('Image size is: {r} rows x {c} columns\n'.format(r=rows, c=cols))

# Does the raster have a description or metadata?
desc = dataset.GetDescription()
metadata = dataset.GetMetadata()

print('Raster description: {desc}'.format(desc=desc))
print('Raster metadata:')
print(metadata)
print('\n')

# What driver was used to open the raster?
driver = dataset.GetDriver()
print('Raster driver: {d}\n'.format(d=driver.ShortName))

# What is the raster's projection?
proj = dataset.GetProjection()
print('Image projection:')
print(proj + '\n')

# What is the raster's "geo-transform"
gt = dataset.GetGeoTransform()
print('Image geo-transform: {gt}\n'.format(gt=gt))

#-------------------------------------------------------------------------------------------------
# Image raster bands
#-------------------------------------------------------------------------------------------------

######### BAND 1 - BLUE
# Open the blue band in our image
blue = dataset.GetRasterBand(1)

print(blue)

# What is the band's datatype?
datatype = blue.DataType
print('Band datatype: {dt}'.format(dt=blue.DataType))

# If you recall from our discussion of enumerated types, this "3" we printed has a more useful definition for us to use
datatype_name = gdal.GetDataTypeName(blue.DataType)
print('Band datatype: {dt}'.format(dt=datatype_name))

# We can also ask how much space does this datatype take up
bytes = gdal.GetDataTypeSize(blue.DataType)
print('Band datatype size: {b} bytes\n'.format(b=bytes))

# How about some band statistics?
band_max, band_min, band_mean, band_stddev = blue.GetStatistics(0, 1)
print('Band range: {minimum} - {maximum}'.format(maximum=band_max,
                                                 minimum=band_min))
print('Band mean, stddev: {m}, {s}\n'.format(m=band_mean, s=band_stddev))


######### BAND 2 - GREEN
green = dataset.GetRasterBand(2)

print(green)

# What is the band's datatype?
datatype = green.DataType
print('Band datatype: {dt}'.format(dt=green.DataType))

# If you recall from our discussion of enumerated types, this "3" we printed has a more useful definition for us to use
datatype_name = gdal.GetDataTypeName(green.DataType)
print('Band datatype: {dt}'.format(dt=datatype_name))

# We can also ask how much space does this datatype take up
bytes = gdal.GetDataTypeSize(green.DataType)
print('Band datatype size: {b} bytes\n'.format(b=bytes))

# How about some band statistics?
band_max, band_min, band_mean, band_stddev = green.GetStatistics(0, 1)
print('Band range: {minimum} - {maximum}'.format(maximum=band_max,
                                                 minimum=band_min))
print('Band mean, stddev: {m}, {s}\n'.format(m=band_mean, s=band_stddev))


######### BAND 3 - RED
red = dataset.GetRasterBand(3)

print(red)

# What is the band's datatype?
datatype = red.DataType
print('Band datatype: {dt}'.format(dt=red.DataType))

# If you recall from our discussion of enumerated types, this "3" we printed has a more useful definition for us to use
datatype_name = gdal.GetDataTypeName(red.DataType)
print('Band datatype: {dt}'.format(dt=datatype_name))

# We can also ask how much space does this datatype take up
bytes = gdal.GetDataTypeSize(red.DataType)
print('Band datatype size: {b} bytes\n'.format(b=bytes))

# How about some band statistics?
band_max, band_min, band_mean, band_stddev = red.GetStatistics(0, 1)
print('Band range: {minimum} - {maximum}'.format(maximum=band_max,
                                                 minimum=band_min))
print('Band mean, stddev: {m}, {s}\n'.format(m=band_mean, s=band_stddev))


######### BAND 4 - NIR
nir = dataset.GetRasterBand(4)

print(nir)

# What is the band's datatype?
datatype = nir.DataType
print('Band datatype: {dt}'.format(dt=nir.DataType))

# If you recall from our discussion of enumerated types, this "3" we printed has a more useful definition for us to use
datatype_name = gdal.GetDataTypeName(nir.DataType)
print('Band datatype: {dt}'.format(dt=datatype_name))

# We can also ask how much space does this datatype take up
bytes = gdal.GetDataTypeSize(nir.DataType)
print('Band datatype size: {b} bytes\n'.format(b=bytes))

# How about some band statistics?
band_max, band_min, band_mean, band_stddev = nir.GetStatistics(0, 1)
print('Band range: {minimum} - {maximum}'.format(maximum=band_max,
                                                 minimum=band_min))
print('Band mean, stddev: {m}, {s}\n'.format(m=band_mean, s=band_stddev))


######### BAND 5 - SWIR1
swir1 = dataset.GetRasterBand(5)

print(swir1)

# What is the band's datatype?
datatype = swir1.DataType
print('Band datatype: {dt}'.format(dt=swir1.DataType))

# If you recall from our discussion of enumerated types, this "3" we printed has a more useful definition for us to use
datatype_name = gdal.GetDataTypeName(swir1.DataType)
print('Band datatype: {dt}'.format(dt=datatype_name))

# We can also ask how much space does this datatype take up
bytes = gdal.GetDataTypeSize(swir1.DataType)
print('Band datatype size: {b} bytes\n'.format(b=bytes))

# How about some band statistics?
band_max, band_min, band_mean, band_stddev = swir1.GetStatistics(0, 1)
print('Band range: {minimum} - {maximum}'.format(maximum=band_max,
                                                 minimum=band_min))
print('Band mean, stddev: {m}, {s}\n'.format(m=band_mean, s=band_stddev))


######### BAND 7 - SWIR2
swir2 = dataset.GetRasterBand(6)

print(swir2)

# What is the band's datatype?
datatype = swir2.DataType
print('Band datatype: {dt}'.format(dt=swir2.DataType))

# If you recall from our discussion of enumerated types, this "3" we printed has a more useful definition for us to use
datatype_name = gdal.GetDataTypeName(swir2.DataType)
print('Band datatype: {dt}'.format(dt=datatype_name))

# We can also ask how much space does this datatype take up
bytes = gdal.GetDataTypeSize(swir2.DataType)
print('Band datatype size: {b} bytes\n'.format(b=bytes))

# How about some band statistics?
band_max, band_min, band_mean, band_stddev = swir2.GetStatistics(0, 1)
print('Band range: {minimum} - {maximum}'.format(maximum=band_max,
                                                 minimum=band_min))
print('Band mean, stddev: {m}, {s}\n'.format(m=band_mean, s=band_stddev))

######### NDVI
ndvi = dataset.GetRasterBand(7)

print(ndvi)

# What is the band's datatype?
datatype = ndvi.DataType
print('Band datatype: {dt}'.format(dt=ndvi.DataType))

# If you recall from our discussion of enumerated types, this "3" we printed has a more useful definition for us to use
datatype_name = gdal.GetDataTypeName(ndvi.DataType)
print('Band datatype: {dt}'.format(dt=datatype_name))

# We can also ask how much space does this datatype take up
bytes = gdal.GetDataTypeSize(ndvi.DataType)
print('Band datatype size: {b} bytes\n'.format(b=bytes))

# How about some band statistics?
band_max, band_min, band_mean, band_stddev = ndvi.GetStatistics(0, 1)
print('Band range: {minimum} - {maximum}'.format(maximum=band_max,
                                                 minimum=band_min))
print('Band mean, stddev: {m}, {s}\n'.format(m=band_mean, s=band_stddev))