# Sampling network of the Basque country corresponding to the EU Soil Law

Written by: [Ichsani Wheeler](mailto:ichsani.wheeler@opengeohub.org), [Tomislav Hengl (OpenGeoHub Foundation)](mailto:tom.hengl@opengeohub.org) and [Robert Minarik](mailto:robert.minarik@opengeohub.org) (OpenGeoHub)

Recipient: Lur Epelde (NEIKER)

## Important notes:

It was agreed in the contract that the shared methods, scripts and know-how will be not shared with any third party. The know-how is the intellectual property of OpenGeoHub Foundation and the authors. The supporting files are only for internal use of NEIKER to modify the sampling locations if needed. 

## Preparation of the candidate locations for sampling

### Preparing the covariates

Raw covariates were provided by the experts from Neiker representing the main soil forming factors (the climate, the soil and the terrain) in the Basque country. We agreed on the target spatial resolution of 50 m. to All raster layers with finer resolution were resampled to 50 m using the nearest neighborhood algorithm and they were reprojected to ETRS89/UTM30N projection (EPSG: 25830). The extent of all layers was harmonized using the grid of the resampled DTM. The parent material map was rasterized using the grid of the DTM layer. All raster layers having the spatial resolution coarser than 50 m stayed untouched, because the downsampling would not increase the information provided by the layers. 

OGH calculated new agreed terrain derivatives (Slope, Topographic position index and Topographic Wetness Index) from the resampled DTM using common tools in SAGA GIS. TPI index was excluded for stratification of the covariates, because the algorithms did not accept negative values. The rescaling of the covariates was not an option, because the applied [‘Sampling Strata’](https://barcaroli.github.io/SamplingStrata/articles/SamplingStrata.html) algorithm rescaled the covariates internally. The final list of covariates used for stratification is presented below: 

* Temperature (mean, minimum and maximum of 1971-2016):

  * /mnt/landmark/robert\_neiker/used\_covariates/clim\_average\_temperature\_1971\_2016\_ETRS89.tif

  * /mnt/landmark/robert\_neiker/used\_covariates/clim\_maximum\_temperature\_1971\_2016\_ETRS89.tif

  * /mnt/landmark/robert\_neiker/used\_covariates/clim\_minimum\_temperature\_1971\_2016\_ETRS89.tif

* Total Precipitation in mm per year of 1971-2016:

  * /mnt/landmark/robert\_neiker/used\_covariates/clim\_total\_precipitation\_mm\_per\_year\_1971\_2016\_ETRS89.tif

* Sum of average daily temperature exceeding 5ºC, adding the degree days of each year (complete), and averaging the accumulated annual temperature in all years of the period (1971-2016)

  * /mnt/landmark/robert\_neiker/used\_covariates/clim\_gdd\_1km\_CAPV\_annual\_1971\_2016\_ETRS89.tif

* Parent material (regolith thickness, surface formations, geomorphology, geotechnical); vector map 1 : 25 000; converted to the raster and resampled to 50 m using nearest neighborhood

  * /mnt/landmark/robert\_neiker/used\_covariates/soil\_litho\_50.tif

* Clay map, 1 m resolution; resampled to 50 m using nearest neighborhood

  * /mnt/landmark/robert\_neiker/used\_covariates/soil\_clay\_50.tiff

* Lidar DTM, 25 m resolution; resampled to 50 m using nearest neighborhood

  * /mnt/landmark/robert\_neiker/used\_covariates/dtm\_dtm\_50.tif

* LULC mask for sampling, vector map; 

  * /mnt/landmark/robert\_neiker/used\_covariates/mask/mask\_aggre.gpkg (the polygons were aggregated by using /mnt/landmark/robert\_neiker/used\_covariates/mask/SIGPAC\_05\_01\_2022\_mask\_v2\_ROBERT.shp

* Terrain derivates:

  * /mnt/landmark/robert\_neiker/used\_covariates/dtm\_slope\_50.tif

  * /mnt/landmark/robert\_neiker/used\_covariates/dtm\_twi\_50.tif

The modification of the covariates resulted in the raster data cube. The pixels in 50 m resolution represented the candidate sampling locations in R\_script\_support\_files folder (\~ 2.5 millions). For the need of the sampling algorithm, the pixels were turned to centroids with coordinates. It resulted in the table where centroids were rows and the values of the covariates were stored in the columns. This table was the input for the sampling procedure. **Therefore It is possible to move the sampling point inside the pixel, because the sampling location in the center represents the whole pixel (the area of 50 X 50m).** 

The candidate locations are provided as the Geopackage file including all covariate values as attributes. The ID column is “cell”.  It is possible to work with the file in GIS or R to add new covariates as columns, filter the points according to LULC etc. The file can be used for running new sampling designs of course. 

## Sampling algorithm similar to the Bethel algorithm

The basic idea/assumption is that carefully selected environmental covariates reflect the variability of the target variables. Therefore, the precision constraints set on the precision of the estimates of the covariates should reflect the precision constraint of the survey target variables.

The proposed algorithm distributes the optimal sampling locations over geographical and feature space (doubly balanced) of the covariates with respect to the precision constraint. In this case, the precision constraint is a maximum 5 % coefficient of variation on the input data in the feature space. 

The proposed algorithm is valid both for mapping the target variables (e.g. Soil Organic Carbon density using predictive modelling) and for estimating the subpopulation parameters such as mean SOC stock in each stratum or in the whole country ([Brus, 2022](https://dickbrus.github.io/SpatialSamplingwithR/)). 

### Stratification of the area

The optimization procedure from R package [‘Sampling Strata’](https://barcaroli.github.io/SamplingStrata/articles/SamplingStrata.html) (continuous method) was applied to optimize the number and the pattern of strata in the specified area in order to get the maximum advantage by the available auxiliary information. The optimization algorithm also allocated the number of samples to cover the variability in the strata respecting the precision constraint. The same stratification algorithm was used when designing sampling networks of LUCAS 2018 and LUCAS 2022 [soil modules](https://ec.europa.eu/eurostat/documents/3888793/14812757/KS-TC-22-005-EN-N.pdf/5277ce01-20b9-41a4-8d70-2a8b82a21fd0?t=1656410533651). The algorithm accepted only continuous covariates, because of the continuous method. It was not possible to include lithology into the stratification, but the lithology layer was used to allocate the sampling locations which is described later. 

The optimization procedure stratified the Basque country into 9 landscape types and the algorithm recommended to distribute 26 sampling locations in total (test\_one\_domain) in order to meet the precision constraint. The Neiker asked for increasing the number of points to 400 but preserving the stratification of the country into 9 landscape types. We increased the number of points in each stratum proportionally. The resulting CVs were below 0.01 for all used covariates. 

The optimization procedure script is neiker1\_samplingstrata\_domain1.R in R\_script\_support\_files. However the optimizing stratification algorithm did not select the sampling locations. The optimizing algorithm returned the number of points needed to be sampled in each stratum to cover the variability of covariates in each landscape type. The algorithm also returned the landscape type of each pixel where it belonged to. We used the information to calculate inclusion probabilities of every landscape type which was needed in the next step. 

### Selecting sampling locations

The sampling locations are selected by a probabilistic double balanced sampling algorithm from R package ‘Balanced Sampling’ ([Grafström and Tillé, 2013](https://doi.org/10.1002/env.2194)). The algorithm selects the random sample that is balanced on the covariates and is well spread in the geographical space. 

This is used to select the optimal sampling locations over geographical and feature space with respect to the previous stratification represented by the inclusion probabilities. The equal inclusion probability of each stratum was calculated as the number of allocated samples divided to the total number of pixels in each stratum. Therefore all pixels of one stratum had the equal inclusion probability, but the inclusion probability of each stratum differed according to the size of the sample to allocate and the size of the stratum to keep the proportions given by the optimizing algorithm. The equal inclusion probabilities were necessary for preserving the randomness of sampling in the strata. 

The algorithm is run in the neiker1\_lcube\_domain1.R script in [R\_script\_support\_files](https://drive.google.com/drive/folders/1ACpCJp1IQ0T_69iU61QHGR904ReKZFbe). Among the previously used covariates, the lithology classes and the memberships to the landscape classes (the strata) were used in the sampling algorithm. The categorical variables were converted to the dummy features. The algorithm returned the sampling locations of 400 well spread points as GeoPackage (sampling400\_balapoints\_lithostrato.gpkg). It resulted in sampling points distributed equally over the country and matching the precision constraint seated by the EU law. 

### Subsampling to the 6 years

The same double balanced algorithm was used to split 400 points into 6 subsamplings. We took the 400 points as the input. The algorithm selected approximately 67 points (1/6) that were representative of 400\. These points were excluded from the poll of candidates for the next round. In the next round, 67 points out  of the remaining candidates were sampled. We ran 5 rounds. The 6th round was the rest. The 

The algorithm is run in the neiker1\_lcube\_subsampling.R script in R\_script\_support\_files. 

