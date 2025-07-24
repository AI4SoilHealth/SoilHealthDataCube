# Soil Health Data Cube

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14334987.svg)](https://doi.org/10.5281/zenodo.14334987)

The **Soil Health Data Cube for Europe** provides technical documentation and computational notebooks to support soil health monitoring across Europe.

- **Data License:** CC-BY (unless stated otherwise)  
- **Code License:** MIT License  

For detailed technical information, see the **[Soil Health Data Cube for Europe Technical Manual](https://shdc.ai4soilhealth.eu/)**.

---

## Repository Contents

### 1. [paneu_landmask]

This folder contains files used to produce three Pan-EU land masks:

1. **Jupyter Notebook (Tile Products):**  
   - **Land Mask**: Differentiates land, ocean, and inland water  
   - **NUT-3 Code Map**: Administrative areas at the NUT-3 level  
   - **ISO-3166 Country Code Map**: Countries coded according to ISO-3166 standard  

2. **Bash Scripts:**  
   - Merge tiles, reproject CRS, and resample to different resolutions  

All land masks follow **AI4SoilHealth Work Package 5** standards and align with data coverage from [Copernicus Pan-European Land Service](https://land.copernicus.eu/pan-european), closely matching the official [EEA39 countries](https://lanEEA39d.copernicus.eu/portal_vocabularies/geotags/eea39).

This landmask serves as a **reference for landmask, spatial content, and resolution** for all data products in this repository.

**Contacts**  
- [Xuemeng Tian](mailto:xuemeng.tian@opengeohub.org)  
- [Yu-Feng Ho](mailto:yu-feng.ho@opengeohub.org)  
- [Martijn Witjes](mailto:martijn.witjes@opengeohub.org)  

---

### 2. [landsat_based_spectral_indices]

A time-series of Landsat-based spectral indices (2000–2022) for continental Europe (including Ukraine, the UK, and Turkey).

- **Resolution:** 30 meters  
- **Temporal Coverage:** Bi-monthly, annual, and long-term analyses  
- **Applications:**  
  - Vegetation cover monitoring  
  - Soil exposure assessment  
  - Tillage and crop intensity analysis  
  - Input for soil property modeling  

**Publication / Citation**  
Tian, X., Consoli, D., Witjes, M., Schneider, F., Parente, L., Şahin, M., Ho, Y.-F., Minařík, R., and Hengl, T. (2025):  
*Time series of Landsat-based bimonthly and annual spectral indices for continental Europe for 2000–2022*.  
**Earth Syst. Sci. Data, 17, 741–772.** [https://doi.org/10.5194/essd-17-741-2025](https://doi.org/10.5194/essd-17-741-2025)

**Indices Provided**  
- **Vegetation:** NDVI, SAVI, FAPAR  
- **Soil Exposure:** Bare Soil Fraction (BSF)  
- **Tillage & Soil Sealing:** NDTI, minNDTI  
- **Crop Patterns:** Number of Seasons (NOS), Crop Duration Ratio (CDR)  
- **Water Dynamics:** NDSI, NDWI  

**Production Workflow**  
![General Workflow](https://github.com/AI4SoilHealth/SoilHealthDataCube/assets/96083275/b8ce7d5e-4e2a-4695-83be-f809eb95d80b)

**Example**  
Bare Soil Fraction (%) time series for Europe (2000–2022):  
![BSF Time Series](https://github.com/AI4SoilHealth/SoilHealthDataCube/assets/96083275/1b14d38b-30d9-42c8-9b03-d257576cdb43)

**Complete Access Catalog**  
[Google Spreadsheet Catalog](https://docs.google.com/spreadsheets/d/1QTA6OkkYlZljfHst_inCrkC7DJcMAyHnM9k0iHulwpg/edit?gid=436017183#gid=436017183)

---

### 3. [SOCD_map]

Contains notebooks and scripts for predictive modeling of **soil organic carbon density (SOCD):**

- **Notebooks (001–009):** Testing various steps in the predictive modeling workflow  
- **Benchmark Pipeline Script:** `benchmark_pipeline.py` automates model building  
- **Property-Specific Modeling (010–011):** Loops pipeline across soil properties  
- **Prediction Interval Models (012–014):** Adds uncertainty quantification  

**Publication / Citation**  
Tian, X., de Bruin, S., Simoes, R., Isik, M.S., Minarik, R., Ho, Y., Şahin, M., Herold, M., Consoli, D., and Hengl, T. (2025):  
*Spatiotemporal prediction of soil organic carbon density in Europe (2000–2022) using earth observation and machine learning*.  
**PeerJ, 13:e19605.** [https://doi.org/10.7717/peerj.19605](https://doi.org/10.7717/peerj.19605)

---

### 4. [soil_property_model_pipeline]

Implements the tested pipeline from **SOCD_map** to predict **10 key soil properties**, with the resulting maps available at [https://ecodatacube.eu](https://ecodatacube.eu).

---

### 5. WRB_map

Scripts to test, train, and evaluate predictive models for mapping soil types based on the **IUSS World Reference Base (WRB)** classification.

---

## Acknowledgments & Funding

This work is part of the **[AI4SoilHealth](https://AI4SoilHealth.eu)** project, funded by the **European Union's Horizon Europe Research and Innovation Programme** under **Grant Agreement [No. 101086179](https://cordis.europa.eu/project/id/101086179)**.

*Funded by the European Union. The views expressed are those of the authors and do not necessarily reflect those of the European Union or the European Research Executive Agency.*

---
