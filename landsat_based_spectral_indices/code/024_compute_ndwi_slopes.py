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

start_tile=int(sys.argv[1])
end_tile=int(sys.argv[2])

for i in range(start_tile, end_tile):

  try:
    tile = df.iloc[i]['TILE']

    ttprint(f"Processing {tile}")
    raster_files = {}

    raster_files = {}
    raster_files['ndwi'] = f'http://192.168.49.30:8333/tmp-eumap-ai4sh/{tile}/skmap_aggregate.ndwi-gao.yearly_p50_' + '{dt}.tif'

    rdata = RasterData(raster_files, max_rasters=2000, verbose=True
        ).timespan('20000101', '20221231', date_unit='years', date_step=1, ignore_29feb=True)

    rdata.info['input_path'] = rdata.info.apply(lambda x: x['input_path'].replace('192.168.49.30', f'192.168.49.{30+x.name%13}'), axis=1)

    rdata = rdata.read(gdal_opts=gdal_opts, n_jobs=96,
      ).run(process.SlopeAnalysis(scale_expr = '(data - 125) / 125', scaling = 10000), group='ndwi', outname= '{gr}_glad.landsat.ard2.seasconv.yearly.m.{nm}_{pr}_30m_s_{dt}_eu_epsg.3035_v20231218'
      ).drop(group=['ndwi']
      ).to_s3(
          host = hosts,
          access_key='iwum9G1fEQ920lYV4ol9',
          secret_key='GMBME3Wsm8S7mBXw3U4CNWurkzWMqGZ0n2rXHggS0',
          path=f"tmp-eumap-ai4sh/{tile}",
          secure=False,
          dtype='int16',
          n_jobs=96,
          verbose_cp = False
        )



    ttprint("#End")
  except:
    ttprint(f"Error in tile {tile}")
    traceback.print_exc()
