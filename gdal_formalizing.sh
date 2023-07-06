# VRT to 10m Sparse TIF
## landmask
nohup gdal_translate -co TILED=YES -co BIGTIFF=YES -co COMPRESS=DEFLATE -co ZLEVEL=9 -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024 -co NUM_THREADS=8 -co SPARSE_OK=TRUE eu_landmask.vrt landmask.tif >&landmask.out&
## iso-3166 code
nohup gdal_translate -co TILED=YES -co BIGTIFF=YES -co COMPRESS=DEFLATE -co ZLEVEL=9 -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024 -co NUM_THREADS=8 -co SPARSE_OK=TRUE iso_landmask.vrt iso_landmask.tif >&iso_landmask.out&
## nut-3 code
nohup gdal_translate -co TILED=YES -co BIGTIFF=YES -co COMPRESS=DEFLATE -co ZLEVEL=9 -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024 -co NUM_THREADS=8 -co SPARSE_OK=TRUE nuts_landmask.vrt iso_landmask.tif >&nut_landmask.out&

### parallel - 1
# 10m Sparse TIF to resampled reprojected 10m TIF
## landmask
nohup gdalwarp -overwrite -t_srs EPSG:3035 --config GDAL_CACHEMAX 2048 -tr 10 10 -r mode -co TILED=YES -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024 -co BIGTIFF=YES -co COMPRESS=DEFLATE -co NUM_THREADS=25 -co ZLEVEL=9 landmask3.tif  landmask3_10m.tif >&landmask3_10m.out&
## iso-3166 code
nohup gdalwarp -overwrite -t_srs EPSG:3035  --config GDAL_CACHEMAX 2048 -tr 10 10 -r mode -co TILED=YES -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024 -co BIGTIFF=YES -co COMPRESS=DEFLATE -co NUM_THREADS=25 -co ZLEVEL=9  iso_landmask.tif  iso_landmask_10m.tif >&iso_landmask_10m.out&
## nut-3 code
nohup gdalwarp -overwrite -t_srs EPSG:3035  --config GDAL_CACHEMAX 2048 -tr 10 10 -r mode -co TILED=YES -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024 -co BIGTIFF=YES -co COMPRESS=DEFLATE -co NUM_THREADS=25 -co ZLEVEL=9  nuts_landmask.tif  nuts_landmask_10m.tif >&nut_landmask_10m.out&

# resampled reprojected 10m TIF to soilhealthdatacube_eu & COG & right dtype
## landmask
nohup gdalwarp -overwrite -te 900000 899000 7401000 5501000 -ot Byte --config GDAL_CACHEMAX 2048 -co BLOCKSIZE=1024 -co BIGTIFF=YES -co COMPRESS=DEFLATE -co NUM_THREADS=8 -co LEVEL=9 -of COG landmask3_10m.tif land.mask_ecodatacube.eu_c_10m_s_20210101_20211231_eu_epsg.3035_v1.0.tif
## iso-3166 code
nohup gdalwarp -overwrite -te 900000 899000 7401000 5501000 -ot Int16 --config GDAL_CACHEMAX 2048 -co BLOCKSIZE=1024 -co BIGTIFF=YES -co COMPRESS=DEFLATE -co NUM_THREADS=8 -co LEVEL=9 -of COG  iso_landmask_10m.tif country.code_iso.3166_c_10m_s_20210101_20211231_eu_epsg.3035_v1.0.tif>&iso_landmask_10m.out&
## nut-3 code
nohup gdalwarp -overwrite -te 900000 899000 7401000 5501000 -ot Int16  --config GDAL_CACHEMAX 2048 -co BLOCKSIZE=1024 -co BIGTIFF=YES -co COMPRESS=DEFLATE -co NUM_THREADS=8 -co LEVEL=9 -of COG  nuts_landmask_10m.tif country.code_nuts3_c_100m_s_20210101_20211231_eu_epsg.3035_v1.0.tif>&nut_landmask_10m.out&

### parallel - 2
# Sparse TIF to resampled reprojected 30m TIF
## landmask 
nohup gdalwarp -overwrite -t_srs EPSG:3035 -tr 30 30 -r mode --config GDAL_CACHEMAX 2048 -co TILED=YES -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024 -co BIGTIFF=YES -co COMPRESS=DEFLATE -co NUM_THREADS=25 -co ZLEVEL=9 landmask.tif  landmask_30m.tif >&landmask30m.out&  
## iso-3166 code
nohup gdalwarp -overwrite -t_srs EPSG:3035 -tr 30 30 -r mode --config GDAL_CACHEMAX 2048 -co TILED=YES -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024 -co BIGTIFF=YES -co COMPRESS=DEFLATE -co NUM_THREADS=25 -co ZLEVEL=9  iso_landmask.tif  iso_landmask_30m.tif >&iso_landmask30m.out&  
## nut-3 code
nohup gdalwarp -overwrite -t_srs EPSG:3035 -tr 30 30 -r mode --config GDAL_CACHEMAX 2048 -co TILED=YES -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024 -co BIGTIFF=YES -co COMPRESS=DEFLATE -co NUM_THREADS=25 -co ZLEVEL=9  nuts_landmask.tif  nuts_landmask_30m.tif >&nuts_landmask30m.out& 
  
# resampled 30m TIF to soilhealthdatacube_eu & COG & right dtype
## landmask 
nohup gdalwarp -overwrite -te 900000 899000 7401000 5501000 -ot Byte -tr 30 30 --config GDAL_CACHEMAX 2048 -co BLOCKSIZE=1024 -co BIGTIFF=YES -co COMPRESS=DEFLATE -co NUM_THREADS=8 -co LEVEL=9 -of COG landmask_30m.tif land.mask_ecodatacube.eu_c_30m_s_20210101_20211231_eu_epsg.3035_v1.0.tif >&landmask_30m_cog.out&
## iso-3166 code
nohup gdalwarp -overwrite -te 900000 899000 7401000 5501000 -ot Int16 -tr 30 30 --config GDAL_CACHEMAX 2048 -co BLOCKSIZE=1024 -co BIGTIFF=YES -co COMPRESS=DEFLATE -co NUM_THREADS=8 -co LEVEL=9 -of COG iso_landmask_30m.tif country.code_iso.3166_c_30m_s_20210101_20211231_eu_epsg.3035_v1.0.tif >&iso_landmask30m_cog.out&
## nut-3 code 
nohup gdalwarp -overwrite -te 900000 899000 7401000 5501000 -ot Int16 -tr 30 30 --config GDAL_CACHEMAX 2048 -co BLOCKSIZE=1024 -co BIGTIFF=YES -co COMPRESS=DEFLATE -co NUM_THREADS=8 -co LEVEL=9 -of COG nuts_landmask_30m.tif country.code_nuts3_c_30m_s_20210101_20211231_eu_epsg.3035_v1.0.tif >&nuts_landmask30m_cog.out&


### parallel - 3
# Sparse TIF to resampled reprojected 100m TIF
## landmask 
nohup gdalwarp -overwrite -t_srs EPSG:3035 -tr 100 100 -r mode --config GDAL_CACHEMAX 2048 -co TILED=YES -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024 -co BIGTIFF=YES -co COMPRESS=DEFLATE -co NUM_THREADS=25 -co ZLEVEL=9 landmask.tif  landmask3_100m.tif >&landmask100m.out&  
## iso-3166 code
nohup gdalwarp -overwrite -t_srs EPSG:3035 -tr 100 100 -r mode --config GDAL_CACHEMAX 2048 -co TILED=YES -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024 -co BIGTIFF=YES -co COMPRESS=DEFLATE -co NUM_THREADS=25 -co ZLEVEL=9  iso_landmask.tif  iso_landmask_100m.tif >&iso_landmask100m.out&  
## nut-3 code 
nohup gdalwarp -overwrite -t_srs EPSG:3035 -tr 100 100 -r mode --config GDAL_CACHEMAX 2048 -co TILED=YES -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024 -co BIGTIFF=YES -co COMPRESS=DEFLATE -co NUM_THREADS=25 -co ZLEVEL=9  nuts_landmask.tif  nuts_landmask_100m.tif >&nuts_landmask100m.out&  

# resampled 100m TIF to soilhealthdatacube_eu & COG & right dtype
## landmask 
nohup gdalwarp -overwrite -te 900000 899000 7401000 5501000 -ot Byte -tr 100 100 --config GDAL_CACHEMAX 2048 -co BLOCKSIZE=1024 -co BIGTIFF=YES -co COMPRESS=DEFLATE -co NUM_THREADS=8 -co LEVEL=9 -of COG landmask_100.tif land.mask_ecodatacube.eu_c_100m_s_20210101_20211231_eu_epsg.3035_v1.0.tif >&landmask_100m_cog.out&
## iso-3166 code
nohup gdalwarp -overwrite -te 900000 899000 7401000 5501000 -ot Int16 -tr 100 100 --config GDAL_CACHEMAX 2048 -co BLOCKSIZE=1024 -co BIGTIFF=YES -co COMPRESS=DEFLATE -co NUM_THREADS=8 -co LEVEL=9 -of COG iso_landmask_100m.tif country.code_iso.3166_c_100m_s_20210101_20211231_eu_epsg.3035_v1.0.tif >&iso_landmask_100m_cog.out&
## nut-3 code 
nohup gdalwarp -overwrite -te 900000 899000 7401000 5501000 -ot Int16 -tr 100 100 --config GDAL_CACHEMAX 2048 -co BLOCKSIZE=1024 -co BIGTIFF=YES -co COMPRESS=DEFLATE -co NUM_THREADS=8 -co LEVEL=9 -of COG nuts_landmask_100m.tif country.code_nuts3_c_100m_s_20210101_20211231_eu_epsg.3035_v1.0.tif >&nuts_landmask_100m_cog.out&
