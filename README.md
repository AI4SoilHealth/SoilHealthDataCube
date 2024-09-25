# SoilHealthDataCube
Soil Health Data Cube for Europe. All the data is licensed under CC-BY, and the code is licensed under the MIT License.

For technical information please refer to the **Soil Health Data Cube for Europe technical manual**: <https://shdc.ai4soilhealth.eu/>

List of notebooks available in this repository:

- **[landsat_based_spectral_indices](/landsat_based_spectral_indices)**: Essential code & data used to generate/analyze/visualize/upload the landsat-based spectral indices data cube;

- **[soil_property_model_pipeline](/soil_property_model_pipeline)**: contains scripts used to build predictive models for 10 key soil properties;

- **WRB_map**: contains scripts used to test, train, evaluate predictive models to map soil types based on the IUSS World Reference Base classification system.

- **[SOCD_map](/SOCD_map)**: contains number of notebooks listed below:

- **Notebooks (001 -- 009)**
  - Designed to test various steps in the predictive model building process
  - Explore and validate different methodologies and approaches for model construction

- **Benchmark Pipeline Script**
  - `benchmark_pipeline.py` automates the entire model-building pipeline
  - Streamlines the process based on the initial 10 notebooks

- **Property-Specific Modeling (010 -- 011)**
  - Notebooks with indices `010 -- 011` loop the pipeline through different soil properties
  - Identify and optimize the best model for each property

- **Prediction Interval Models (012 -- 014)**
  - Notebooks with indices `012 -- 014` build models that estimate prediction intervals
  - Add a layer of uncertainty quantification to the predictions

## Acknowledgments

**[AI4SoilHealth.eu](https://AI4SoilHealth.eu)** project has received funding 
from the European Union's Horizon Europe research an innovation programme under 
grant agreement **[No. 101086179](https://cordis.europa.eu/project/id/101086179)**.

Funded by the European Union. Views and opinions expressed are however those of 
the author(s) only and do not necessarily reflect those of the European Union or 
European Research Executive Agency. Neither the European Union nor the granting 
authority can be held responsible for them.




 
