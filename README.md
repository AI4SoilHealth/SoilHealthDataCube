# SoilHealthDataCube
Soil Health Data Cube for Europe

All the data is licensed under CC-BY-SA, and the code is licensed under the MIT License.
## Pan-EU Landmask
Three Pan-EU land masks designed for different specific applications in the production of soil health data cube:
   - **Land mask**: with values differentiating land, ocean, and inland water
   - **NUT-3 code map**: with values differentiating administrative area at nut-3 level
   - **ISO-3166 country code map**: with values differentiating countries according to ISO-3166 standard
The jupyter notebooks and bash files that are used to produce masks, merge tiles, reproject crs, resample to another resolution.

All the landmasks are aligned with the standard spatial/temporal resolution and sizes indicated/recomended by AI4SoilHealth project, Work Package - 5. The coverage of these maps closely match the data coverage of https://land.copernicus.eu/pan-european i.e. the official selection of countries listed here: https://lanEEA39d.copernicus.eu/portal_vocabularies/geotags/eea39. 

These masks are created by [Xuemeng](xuemeng.tian@opengeohub.org), [Yu-Feng](yu-feng.ho@opengeohub.org), and [Martijn](martijn.witjes@opengeohub.org) from [OpenGeoHub](https://opengeohub.org/). If you spot any problems in the land masks, or see any possible improvements in them, or have any questions, or etc..., just raise an issue [here](https://github.com/AI4SoilHealth/SoilHealthDataCube/issues) or send us emails! We appreciate any feedbacks that could refine these masks.

## Landsat-based Spectral Indices Data Cube
This data cube offers a time-series of Landsat-based spectral indices maps across continental Europe—including Ukraine, the UK, and Turkey—from 2000 to 2022. At a resolution of 30 meters, it includes bi-monthly, annual, and long-term analyses, focusing on key aspects of soil health such as vegetation cover, soil exposure, tillage practices, and crop intensity. Apart from direct monitoring, analysis, and verification of specific aspects of soil health, this data cube also provides important input for modeling and mapping soil properties. All the maps are aligned with the standard spatial/temporal resolution and sizes indicated/recomended by AI4SoilHealth project, Work Package - 5.

Please cite as:

- Tian, X., Consoli, D., Hengl, T., Schneider, F., Parente, L., Şahin, M., Minařík, R., Ho, Y., (2024?) "Time-series of Landsat-based spectral indices for continental Europe for 2000–2022 to support soil health monitoring", submitted to [PeerJ], preprint available at: [to be filled].

### Summary
The corresponding folder provides 
1. Essential code & data used to generate/analyze/visualize/upload the landsat-based spectral indices data cube,
2. Visualization for selected indices.

The indices include:
- **Vegetation index**: Normalized Difference Vegetation Index (NDVI), Soil Adjusted Vegetation Index (SAVI), and Fraction of Absorbed Photosynthetically Active Radiation (FAPAR).
- **Soil exposure**: Bare Soil Fraction (BSF).
- **Tillage and soil sealing**: Normalized Difference Tillage Index (NDTI) and minimum Normalized Difference Tillage Index (minNDTI).
- **Crop patterns**: Number of Seasons (NOS) and Crop Duration Ratio (CDR).
- **Water dynamics**: Normalized Difference Snow Index (NDSI) and Normalized Difference Water Index (NDWI)
  
General steps of maps production are:

![00_general_workflow drawio](https://github.com/AI4SoilHealth/SoilHealthDataCube/assets/96083275/b8ce7d5e-4e2a-4695-83be-f809eb95d80b)

A preview of the BSF (%) time series for Europe from 2000 to 2022:

![global_view4](https://github.com/AI4SoilHealth/SoilHealthDataCube/assets/96083275/1b14d38b-30d9-42c8-9b03-d257576cdb43)



### Access to the data cube
**Yearly Landsat ARD Red band**
  - URL: https://stac.ecodatacube.eu/red_glad.landsat.ard2.seasconv.m.yearly/collection.json
  - Description: Red band aggregated yearly from 30-m bi-monthly gapfilled GLAD Landsat ARD red band from 2000 to 2022.
  - Theme: Surface reflectance
  - DOI: https://doi.org/10.5281/zenodo.10851081


  **Yearly Landsat ARD Green band**
  - URL: https://stac.ecodatacube.eu/green_glad.landsat.ard2.seasconv.m.yearly/collection.json
  - Description: Green band aggregated yearly from 30-m bi-monthly gapfilled GLAD Landsat ARD green band from 2000 to 2022.
  - Theme: Surface reflectance
  - DOI: https://doi.org/10.5281/zenodo.10851081


  **Yearly Landsat ARD Blue band**
  - URL: https://stac.ecodatacube.eu/blue_glad.landsat.ard2.seasconv.m.yearly/collection.json
  - Description: Blue band aggregated yearly from 30-m bi-monthly gapfilled GLAD Landsat ARD blue band from 2000 to 2022.
  - Theme: Surface reflectance
  - DOI: https://doi.org/10.5281/zenodo.10851081


  **Yearly Landsat ARD Near-Infrared band (NIR)**
  - URL: https://stac.ecodatacube.eu/nir_glad.landsat.ard2.seasconv.m.yearly/collection.json
  - Description: NIR band aggregated yearly from 30-m bi-monthly gapfilled GLAD Landsat ARD NIR band from 2000 to 2022.
  - Theme: Surface reflectance
  - DOI: https://doi.org/10.5281/zenodo.10851081


  **Yearly Landsat ARD Shortwave Near-Infrared band (SWIR1)**
  - URL: https://stac.ecodatacube.eu/swir1_glad.landsat.ard2.seasconv.m.yearly/collection.json
  - Description: SWIR1 band aggregated yearly from 30-m bi-monthly gapfilled GLAD Landsat ARD SWIR1 band from 2000 to 2022.
  - Theme: Surface reflectance
  - DOI: https://doi.org/10.5281/zenodo.10851081


  **Yearly Landsat ARD Shortwave Near-Infrared 2 band (SWIR2)**
  - URL: https://stac.ecodatacube.eu/swir2_glad.landsat.ard2.seasconv.m.yearly/collection.json
  - Description: SWIR2 band aggregated yearly from 30-m bi-monthly gapfilled GLAD Landsat ARD SWIR2 band from 2000 to 2022.
  - Theme: Surface reflectance
  - DOI: https://doi.org/10.5281/zenodo.10851081


  **Yearly Landsat ARD Thermal band**
  - URL: https://stac.ecodatacube.eu/thermal_glad.landsat.ard2.seasconv.m.yearly/collection.json
  - Description: Thermal band aggregated yearly from 30-m bi-monthly gapfilled GLAD Landsat ARD thermal band from 2000 to 2022.
  - Theme: Surface reflectance
  - DOI: https://doi.org/10.5281/zenodo.10851081

  **Bi-monthly Normalized Difference Vegetation Index (NDVI)**
  - URL: https://stac.ecodatacube.eu/ndvi_glad.landsat.ard2.seasconv/collection.json
  - Description: [NDVI](Tucker, 1979) quantifies vegetation greenness, computed from 30-m bi-monthly aggregated and gapfilled GLAD Landsat ARD from 2000 to 2022.
  - Theme: Vegetation
  - DOI: https://doi.org/10.5281/zenodo.10884235


  **Bi-monthly Soil Adjusted Vegetation Index(SAVI)**
  - URL: https://stac.ecodatacube.eu/savi_glad.landsat.ard2.seasconv.bimonthly.m/collection.json
  - Description: [SAVI](Huete, 1988) is a vegetation index that attempts to minimize soil brightness influences using a soil-brightness correction factor, computed from 30-m bi-monthly aggregated and gapfilled GLAD Landsat ARD from 2000 to 2022.
  - Theme: Vegetation
  - DOI: https://doi.org/10.5281/zenodo.10884235


  **Bi-monthly Fraction of Absorbed Photosynthetically Active Radiation (FAPAR)**
  - URL: https://stac.ecodatacube.eu/fapar_glad.landsat.ard2.seasconv/collection.json
  - Description: [FAPAR](Robinson et al., 2018) quantifies the fraction of the solar radiation absorbed by live leaves for the photosynthesis activity, computed from 30-m bi-montlhy NDVI from 2000 to 2022.
  - Theme: Vegetation
  - DOI: https://doi.org/10.5281/zenodo.10884235


  **Bi-monthly Normalized Difference Snow Index (NDSI)**
  - URL: https://stac.ecodatacube.eu/ndsi_glad.landsat.ard2.seasconv/collection.json
  - Description: NDSI computed for 30-m bi-monthly aggregated and gapfilled GLAD Landsat ARD from 2000 to 2022.
  - Theme: Vegetation
  - DOI: https://doi.org/10.5281/zenodo.10884235


  **Bi-monthly Normalized Difference Water Index (NDWI, Gao)**
  - URL: https://stac.ecodatacube.eu/ndwi.gao_glad.landsat.ard2.seasconv.bimonthly.m/collection.json
  - Description: [NDWI](Gao, 1996) indicates vegetation liquid water content, computed from 30-m bi-monthly aggregated and gapfilled GLAD Landsat ARD from 2000 to 2022.
  - Theme: Water
  - DOI: https://doi.org/10.5281/zenodo.10884235


  **Bi-monthly Landsat Normalized Difference Tillage Intensity (NDTI)**
  - URL: https://stac.ecodatacube.eu/ndti_glad.landsat.ard2.seasconv/collection.json
  - Description: [NDTI](Van Deventer et al., 1997) differentiates crop residues from soil, computed from 30-m bi-monthly aggregated and gapfilled GLAD Landsat ARD from 2000 to 2022.
  - Theme: Tillage
  - DOI: https://doi.org/10.5281/zenodo.10884235


**Yearly Normalized Difference Vegetation Index (NDVI)**
  - URL: https://stac.ecodatacube.eu/ndvi_glad.landsat.ard2.seasconv.m.yearly/collection.json
  - Description: NDVI aggregated yearly from bi-monthly NDVI time series from 2000 to 2022.
  - Theme: Vegetation
  - DOI: https://doi.org/10.5281/zenodo.10851081


  **Yearly Normalized Difference Water Index (NDWI, Gao)**
  - URL: https://stac.ecodatacube.eu/ndwi.gao_glad.landsat.ard2.seasconv.m.yearly/collection.json
  - Description: NDWI (Gao) aggregated yearly from bi-monthly NDWI (Gao)
  - Theme: Water
  - DOI: https://doi.org/10.5281/zenodo.10851081


  **Yearly minimum Normalized Difference Tillage Intensity (minNDTI)**
  - URL: https://stac.ecodatacube.eu/ndti.min_glad.landsat.ard2.seasconv.bimonthly.min/collection.json
  - Description: Yearly minimum NDTI selected from bi-monthly NDTI at 30-m from 2000 to 2022.
  - Theme: Tillage
  - DOI: https://doi.org/10.5281/zenodo.10777869


  **Yearly Bare Soil Fraction (BSF)**
  - URL: https://stac.ecodatacube.eu/bsf_glad.landsat.ard2.seasconv.m.yearly/collection.json
  - Description: BSF (bare soil fraction) computed for 30-m bi-monthly aggregated and gapfilled GLAD Landsat ARD from 2000 to 2022, to indicate the yearly duration a location stays bare.
  - Theme: Soil exposure
  - DOI: https://doi.org/10.5281/zenodo.10777869


  **Yearly Number of Seasons (NOS)**
  - URL: https://stac.ecodatacube.eu/nos_glad.landsat.ard2.seasconv/collection.json
  - Description: Number of Seasons (NOS) derived from bimonthly NDVI time series at 30-m from 2000 to 2022, indicating the annual crop cycle numbers.
  - Theme: Crop intensity
  - DOI: https://doi.org/10.5281/zenodo.10777869


  **Yearly Crop Duration Ratio (CDR)**
  - URL: https://stac.ecodatacube.eu/cdr_glad.landsat.seasconv/collection.json
  - Description: Crop Duration Ratio (CDR) measures the active cropping period's proportion of the year, calculated from bimonthly NDVI time series at 30-m from 2000 to 2022.
  - Theme: Crop intensity
  - DOI: https://doi.org/10.5281/zenodo.10777869


  **Long term trend of NDVI-P50 between 2000 and 2022**
  - URL: https://stac.ecodatacube.eu/ndvi_glad.landsat.ard2.seasconv.yearly.m.theilslopes/collection.json
  - Description: NDVI slopes fitted on the annual NDVI P50 time series from 2000 to 2022.
  - Theme: Vegetation
  - DOI: https://zenodo.org/records/10776892


  **Long term trend of NDWI-P50 (Gao) between 2000 and 2022**
  - URL: https://stac.ecodatacube.eu/ndwi_glad.landsat.ard2.seasconv.yearly.m.theilslopes/collection.json
  - Description: Slope fitted with Theil-Sen estimator on annual NDWI (Gao) time series between 2000 and 2022
  - Theme: Water
  - DOI: https://zenodo.org/records/10776892


 **Long term trend of BSF between 2000 and 2022**
  - URL: https://stac.ecodatacube.eu/bsf_glad.landsat.ard2.seasconv.yearly.m.theilslopes/collection.json
  - Description: Slope fiited with Theil-Sen estimator on annual BSF time series between 2000 and 2022.
  - Theme: Soil exposure
  - DOI: https://zenodo.org/records/10776892


  **Long term trend of minNDTI between 2000 and 2022**
  - URL: https://stac.ecodatacube.eu/ndti.min.slopes_glad.landsat.ard2.seasconv.yearly.min.theilslopes/collection.json
  - Description: Slope fiited with Theil-Sen estimator on annual minNDTI time series between 2000 and 2022.
  - Theme: Tillage
  - DOI: https://zenodo.org/records/10776892

### Contacts
These maps are created by [Xuemeng](xuemeng.tian@opengeohub.org), [Davide](davide.consoli@opengeohub.org), [Leandro](leandro.parente@opengeohub.org), and [Yu-Feng](yu-feng.ho@opengeohub.org) from [OpenGeoHub](https://opengeohub.org/). If you spot any problems in the maps, or see any possible improvements in them, or see any potential collaborations, or etc..., just raise an issue [here](https://github.com/AI4SoilHealth/SoilHealthDataCube/issues) or send us emails! We appreciate any feedbacks/helps that could refine them.

## Disclaimer
The production of these data layers are parts of [AI4SoilHealth](https://cordis.europa.eu/project/id/101086179) project. The AI4SoilHealth project project has received funding from the European Union's Horizon Europe research an innovation programme under grant agreement No. 101086179. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or European Commision. Neither the European Union nor the granting authority can be held responsible for them. The data is provided “as is”. AI4SoilHealth project consortium and its suppliers and licensors hereby disclaim all warranties of any kind, express or implied, including, without limitation, the warranties of merchantability, fitness for a particular purpose and non-infringement. Neither AI4SoilHealth Consortium nor its suppliers and licensors, makes any warranty that the Website will be error free or that access thereto will be continuous or uninterrupted. You understand that you download from, or otherwise obtain content or services through, the Website at your own discretion and risk. 




 
