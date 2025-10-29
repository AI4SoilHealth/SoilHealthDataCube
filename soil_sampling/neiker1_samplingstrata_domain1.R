library(terra)
library(sf)
library(SamplingStrata)
#to_sample_sf = st_read('to_sample_sf6.gpkg')
colnames(to_sample_sf)
features = colnames(to_sample_sf)[colnames(to_sample_sf) %in% c("soil_clay_50", 
                                                                "clim_minimum_temperature_1971_2016_ETRS89", "clim_average_temperature_1971_2016_ETRS89","clim_maximum_temperature_1971_2016_ETRS89",
                                                                "clim_gdd_1km_CAPV_annual_1971_2016_ETRS89", "clim_total_precipitation_mm_per_year_1971_2016_ETRS89",
                                                                "dtm_dtm_50", "dtm_tpi_50", "dtm_slope_50", "dtm_twi_50"
)]

# features = colnames(to_sample_sf)[!(colnames(to_sample_sf) %in% c('cell', 'geom'
# ))]
dataframe = to_sample_sf
dataframe = st_drop_geometry(to_sample_sf)
dataframe$dom1 = 1
frame_conti_cov <- buildFrameDF(df = dataframe,
                                id ='cell',
                                X = features,
                                Y = features,
                                domainvalue = "dom1")

# cv <- as.data.frame(list(DOM = "DOM1", 
#                          CV1 = 0.05, CV2 = 0.05, CV3 = 0.5, CV4 = 0.05, CV5 = 0.05, 
#                          CV6 = 0.05, CV7 = 0.5, CV8 = 0.05, CV9 = 0.05, CV10 = 0.05,
#                          CV11 = 0.05, CV12 = 0.05, CV13 = 0.5, CV14 = 0.05, CV15 = 0.05, 
#                          CV16 = 0.05, CV17 = 0.05, CV18 = 0.5, CV19 = 0.05, CV20 = 0.05, 
#                          CV21 = 0.05, CV22 = 0.05, CV23 = 0.5,
#                          domainvalue = 1))

cv <- as.data.frame(list(DOM = "DOM1",
                         CV1 = 0.05, CV2 = 0.05, CV3 = 0.5, CV4 = 0.05, CV5 = 0.05,
                         CV6 = 0.05, CV7 = 0.5, CV8 = 0.05, CV9 = 0.05, CV10 = 0.05,
                         domainvalue = 1))
set.seed(1234)
init_sol3 <- KmeansSolution2(frame=frame_conti_cov,
                             errors=cv,
                             maxclusters = 50) 

nstrata3 <- tapply(init_sol3$suggestions,
                   init_sol3$domainvalue,
                   FUN=function(x) length(unique(x)))
nstrata3
initial_solution3 <- prepareSuggestion(init_sol3,frame_conti_cov,nstrata3)
set.seed(1234)
solution3 <- optimStrata(method = "continuous",
                         errors = cv, 
                         framesamp = frame_conti_cov,
                         iter = 50,
                         pops = 10,
                         nStrata = nstrata3,
                         suggestions = initial_solution3)
saveRDS(solution3, 'solution1.rds')