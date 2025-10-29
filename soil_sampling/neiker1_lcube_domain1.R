library(BalancedSampling)
library(fastDummies)
library(SamplingStrata)
library(sf)
library(dplyr)
##doubly balanced including strata and litho
#to_sample_sf = st_read('to_sample_sf6_croplands.gpkg')
#to_sample_sf = st_read('to_sample_sf6_grasslands.gpkg')
#to_sample_sf = st_read('to_sample_sf6_forest.gpkg')
solution1 = readRDS('solution1.rds')
to_sample_sf = st_read('to_sample_sf6.gpkg')
to_sample_sf = to_sample_sf[,!(names(to_sample_sf) %in% c('clusters_h20', 'provinces'))]

set.seed(1234)
#adjustedStrata <- adjustSize(size=65,strata=solution1crop$aggr_strata,cens=NULL)
#adjustedStrata <- adjustSize(size=120,strata=solution1grass$aggr_strata,cens=NULL)
#adjustedStrata <- adjustSize(size=250,strata=solution1forest$aggr_strata,cens=NULL)
#adjustedStrata <- adjustSize(size=200,strata=solution1$aggr_strata,cens=NULL)
adjustedStrata <- adjustSize(size=400,strata=solution1$aggr_strata,cens=NULL)
expected_CV(adjustedStrata)
#smrstrata_cov <- summaryStrata(solution1crop$framenew, adjustedStrata, progress = FALSE)
#smrstrata_cov <- summaryStrata(solution1grass$framenew, adjustedStrata, progress = FALSE)
#smrstrata_cov <- summaryStrata(solution1forest$framenew, adjustedStrata, progress = FALSE)
smrstrata_cov <- summaryStrata(solution1$framenew, adjustedStrata, progress = FALSE)
#N_h <-  table(solution1crop$framenew$STRATO)
#N_h <-  table(solution1grass$framenew$STRATO)
#N_h <-  table(solution1forest$framenew$STRATO)
N_h <-  table(solution1$framenew$STRATO)
n_h <- smrstrata_cov$Allocation
picov <- n_h / N_h
dataframe = to_sample_sf
dataframe = st_drop_geometry(to_sample_sf)
dataframe = dataframe[,!(names(dataframe) %in% c("cell"))]
#dataframe$strato = factor(solution1crop$framenew$STRATO)
#dataframe$strato = factor(solution1grass$framenew$STRATO)
#ataframe$strato = factor(solution1forest$framenew$STRATO)
dataframe$strato = factor(solution1$framenew$STRATO)
lut = data.frame(strato = levels(dataframe$strato), picov = as.numeric(picov))
dataframe = merge(x = dataframe, y = lut)
dataframe = fastDummies::dummy_cols(dataframe, select_columns = c("soil_litho_50", 'strato'))
Xbal = dataframe[,!(names(dataframe) %in% c("strato", "soil_litho_50", 'picov', "dtm_tpi_50"))]
#Xbal = dataframe[,!(names(dataframe) %in% c("strato", "soil_litho_50", 'picov', "dtm_tpi_50", 'clusters_h20', 'provinces'))]
Xbal = as.matrix(sapply(Xbal, as.numeric))
Xspread = st_coordinates(to_sample_sf)
#to_sample_sf$strata_solution1crop = solution1crop$framenew$STRATO
#to_sample_sf$strata_solution1grass = solution1grass$framenew$STRATO
#to_sample_sf$strata_solution1forest = solution1forest$framenew$STRATO
to_sample_sf$strata_solution1 = solution1$framenew$STRATO
set.seed(1234)
balanced_points <- BalancedSampling::lcube(Xbal = Xbal, Xspread = Xspread, prob = dataframe$picov)
balanced_points  = to_sample_sf[balanced_points,]
smrstrata_cov$Allocation
#table(balanced_points$strata_solution1crop)
#table(balanced_points$strata_solution1grass)
#table(balanced_points$strata_solution1forest)
table(balanced_points$strata_solution1)
plot(balanced_points)
#st_write(balanced_points, 'sampling65_crops_lithostrato.gpkg')
#st_write(balanced_points, 'sampling117_grass_lithostrato.gpkg')
#st_write(balanced_points, 'sampling243_forest_lithostrato.gpkg')
#st_write(balanced_points, 'sampling200_balapoints_lithostrato.gpkg')
st_write(balanced_points, 'sampling400_balapoints_lithostrato.gpkg')
### creating raster 
clay50 = terra::rast('./used_covariates/soil_clay_50.tiff')
centroids_vect <- terra::vect(to_sample_sf)
#rasterized <- terra::rasterize(centroids_vect, clay50, field = 'strata_solution1crop', background = NA)
#rasterized <- terra::rasterize(centroids_vect, clay50, field = 'strata_solution1grass', background = NA)
rasterized <- terra::rasterize(centroids_vect, clay50, field = 'strata_solution1forest', background = NA)
#terra::writeRaster(rasterized, "sampling65_croplands.tif", overwrite = TRUE)
terra::writeRaster(rasterized, "sampling117_grasslands.tif", overwrite = TRUE)
terra::writeRaster(rasterized, "sampling243_forest.tif", overwrite = TRUE)