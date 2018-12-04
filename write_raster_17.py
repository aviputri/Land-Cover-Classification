import numpy as np

#load result
a = "/Volumes/ga87rif/Study Project/Result/predict_17_50trees_from09full.npy"
predict2017 = np.load(a)

#reshape into 2D
#b = np.reshape(predict2017, (-1,X pixels))
b = np.reshape(predict2017, (-1,9194))
c = np.flipud(b) #flip vertically
#because rasterio is lame it writes from bottom to top

import rasterio
from rasterio.transform import from_origin

#transform = from_origin(110....<--dari origin, -7....<--bukan dari origin, 0.000271658, -0.000271658)
transform = from_origin(110.0346413234388621, -7.8712348941239227, 0.0000905403, -0.0000905403)

new_dataset = rasterio.open('/Volumes/ga87rif/Study Project/Result/image_17_50trees_from09full.tif', 'w', driver='GTiff',
                            height = c.shape[0], width = c.shape[1],
                            count=1, dtype=c.dtype,
                            crs='+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs',
                            transform=transform)

new_dataset.write(c, 1)
new_dataset.close()