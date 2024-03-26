from skmap.io import RasterData
from skmap.io import process
import numpy as np
from skmap.misc import ttprint
import pandas as pd
import traceback
import sys

gdal_opts = {
     'GDAL_HTTP_MULTIRANGE': 'SINGLE_GET',
     'GDAL_HTTP_MERGE_CONSECUTIVE_RANGES': 'NO',
     'GDAL_HTTP_VERSION': '1.0',
     'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
     'VSI_CACHE': 'FALSE',
     'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif',
     'GDAL_HTTP_CONNECTTIMEOUT': '320',
     'CPL_VSIL_CURL_USE_HEAD': 'NO',
     'GDAL_HTTP_TIMEOUT': '320',
     'CPL_CURL_GZIP': 'NO'
}

hosts = [ f'192.168.49.{i}:8333' for i in range(30,43) ]
df = pd.read_csv('/mnt/slurm/jobs/eumap_ai4sh/tiles.csv')
bands = ['red','nir','swir1','swir2']

start_tile=int(sys.argv[1])
end_tile=int(sys.argv[2])

for i in range(start_tile, end_tile):

  try:
    tile = df.iloc[i]['TILE']

    ttprint(f"Processing {tile}")
    raster_files = {}

    for b in bands:
        raster_files[b] = f'http://192.168.49.30:8333/prod-landsat-ard2/{tile}/seasconv/{b}_glad.SeasConv.ard2_m_30m_s_' + '{dt}_go_epsg.4326_v20230908.tif'

    rdata = RasterData(raster_files, max_rasters=2000, verbose=True
        ).timespan('20000101', '20221231', date_unit='months', date_step=2, ignore_29feb=True)

    rdata.info['input_path'] = rdata.info.apply(lambda x: x['input_path'].replace('192.168.49.30', f'192.168.49.{30+x.name%13}'), axis=1)

    rdata = rdata.read(gdal_opts=gdal_opts, n_jobs=96,
            ).run(
                process.Calc( 
                    expressions = {
                        'ndwi': '( ( (nir * 0.004) - (swir1 * 0.004) ) / ( (nir * 0.004) + (swir1 * 0.004) ) ) * 125 + 125',
                        'savi': '( ( (nir * 0.004) - (red * 0.004) )*1.5 / ( (nir * 0.004) + (red * 0.004)  + 0.5) ) * 125 + 125',
                    }
                ), outname= '{gr}_glad.landsat.ard2.seasconv.bemonthly.m_30m_s_{dt}_eu_epsg.3035_v20231218'
            ).run(process.TimeAggregate(time=[process.TimeEnum.YEARLY], operations = ['p25', 'p50', 'p75'], n_jobs = 96
                 ), group=['ndwi',]
            ).drop(group=bands
            ).to_s3(
              host = hosts,
              access_key='iwum9G1fEQ920lYV4ol9',
              secret_key='GMBME3Wsm8S7mBXw3U4CNWurkzWMqGZ0n2rXHggS0',
              path=f"tmp-eumap-ai4sh/{tile}",
              secure=False,
              n_jobs=96,
              verbose_cp = False
            )


    ttprint("#End")
  except:
    ttprint(f"Error in tile {tile}")
    traceback.print_exc()
