### 
# Mask out water. The code processes landmasking in a area-based projection of Europe (EPSG:3035). 
# INPUT: 
# (1) a retiled landmask layer- land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_01_02.tif
# (2) a retiled landsat_based_spectral_indices layer- e.g. ndvi_glad.landsat.ard2.seasconv.m.yearly_p25_30m_s_20030101_20031231_eu_epsg.3035_v20231127.vrt
# OUTPUT:
# (1) a retiled landmased landsat_based_spectral_indices layer
###

### Bash commands
# 1. tiled rasters to a AI4Soil Health DataCube spatial extent: gdal_cmd_sparse="gdalwarp -overwrite -t_srs EPSG:3035 -tr 30 30 -te 900000 899000 7401000 5501000 --config GDAL_CACHEMAX 9216  -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024 -co BIGTIFF=YES -co COMPRESS=DEFLATE -co NUM_THREADS=8 -co SPARSE_OK=TRUE"
# 2. retile map into a new tiled raster in main Europe map projection EPSG:3035: gdal_cmd_retiling="gdal_retile.py -v -ps 17336 38350 -co TILED=YES -co COMPRESS=DEFLATE -co ZLEVEL=9 -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024 -co NUM_THREADS=90 -targetDir"
# 3. make vrt for each landsat_based_spectral_indices layer after landmasking: gdal_cmd_vrt_template="gdalbuildvrt input folder/*.tif"
# 4. produce COG in AI4SoilHealthDataCube spatial extent: GDAL_CMD="gdalwarp -overwrite -t_srs 'EPSG:3035' -tr 30 30 -te 900000 899000 7401000 5501000 --config GDAL_CACHEMAX 9216 -co BLOCKSIZE=1024 -co BIGTIFF=YES -co COMPRESS=DEFLATE -co NUM_THREADS=8 -co LEVEL=9 -of COG"

import rasterio
import os
import numpy as np
import sys
from eumap import parallel
host = sys.argv[1]
folder = sys.argv[2]
output_dir=f'/mnt/{host}/landmask_cog_ai4sh'

# load in landmask
landmask_files = ['http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_01_01.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_01_02.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_01_03.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_01_04.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_01_05.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_01_06.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_01_07.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_01_08.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_01_09.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_01_10.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_01_11.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_01_12.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_01_13.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_02_01.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_02_02.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_02_03.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_02_04.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_02_05.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_02_06.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_02_07.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_02_08.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_02_09.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_02_10.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_02_11.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_02_12.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_02_13.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_03_01.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_03_02.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_03_03.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_03_04.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_03_05.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_03_06.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_03_07.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_03_08.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_03_09.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_03_10.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_03_11.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_03_12.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_03_13.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_04_01.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_04_02.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_04_03.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_04_04.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_04_05.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_04_06.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_04_07.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_04_08.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_04_09.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_04_10.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_04_11.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_04_12.tif',
'http://192.168.49.30:8333/ai4sh/eu_landmask/land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v20230719_04_13.tif'
]
raster_files = sorted([i.path for i in os.scandir(folder)])
args = []
for i in range(len(raster_files)):
    args.append((raster_files[i],landmask_files[i]))
def worker(raster_file, landmask_file):
    try:
        with rasterio.open(landmask_file) as src_inf:
            landmask = src_inf.read(1)
        with rasterio.open(raster_file) as src_inf:
            layer = src_inf.read(1)
        # landmasking water and border
        layer[landmask>1]=255
        out_file = raster_file.replace('_sparse_tmp','_sparse_landmasked')
        os.makedirs(out_file.rsplit('/',1)[0],exist_ok=True)
        # Open the source raster for reading
        with rasterio.open(raster_file) as src:

            # Extract relevant information from the source raster
            profile = src.profile
            # Update the profile to include compression
            profile.update(
                compress='lzw',  # You can use other compression options like 'deflate', 'gzip', etc.
                predictor=2,  # This is for LZW compression, you may adjust for other compressions
                blockxsize=1024,  # Set the desired block size in the x-direction
                blockysize=1024  # Set the desired block size in the y-direction
            )

            # Create a new raster file for writing using the extracted information
            with rasterio.open(out_file, 'w', **profile) as dst:

                # Write the data to the new raster;s
                dst.write(layer, 1)  # You may need to adjust the band index (1 in this case)
    except:
        print(f'ERROR in {raster_file}')
for result in parallel.job(worker, args, n_jobs=30):
    print(result)
