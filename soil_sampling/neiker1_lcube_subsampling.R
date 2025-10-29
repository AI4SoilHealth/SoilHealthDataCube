library(sf)
library(BalancedSampling)
library(caret)
balanced_points = st_read('sampling400_balapoints_lithostrato.gpkg')
n_samples = as.numeric(round(table(balanced_points$strata_solution1)/6))

# Calculate how many points to sample from each strata (e.g., 41 or other fixed number)
points_per_strata <- data.frame(
  strata_solution1 = sort(unique(balanced_points$strata_solution1)),
  n_samples = n_samples  # Specify fixed sample sizes for each strata
)
# Create an empty dataframe to store selected points
selected_points <- balanced_points[0, ]

N_h <-  table(balanced_points$strata_solution1)
n_h <- n_samples
picov <- n_h / N_h
dataframe = balanced_points
dataframe = st_drop_geometry(balanced_points)
dataframe = dataframe[,!(names(dataframe) %in% c("cell"))]
dataframe$strata_solution1 = factor(dataframe$strata_solution1)
lut = data.frame(strata_solution1 = levels(dataframe$strata_solution), picov = as.numeric(picov))
dataframe = merge(x = dataframe, y = lut)
dataframe = fastDummies::dummy_cols(dataframe, select_columns = c("soil_litho_50", 'strata_solution1'))
Xbal = dataframe[,!(names(dataframe) %in% c("strata_solution1", "soil_litho_50", 'picov', "dtm_tpi_50"))]
Xbal = as.matrix(sapply(Xbal, as.numeric))
Xspread = st_coordinates(balanced_points)
set.seed(1234)
first_iteration <- BalancedSampling::lcube(Xbal = Xbal, Xspread = Xspread, prob = dataframe$picov)
first_iteration  = balanced_points[first_iteration,]

table(first_iteration$strata_solution1)
st_write(first_iteration, 'sampling400_first.gpkg')
first_index = as.numeric(row.names(first_iteration))
sampling_pool = balanced_points[-first_index,]

N_h <-  table(sampling_pool$strata_solution1)
n_h <- n_samples
picov <- n_h / N_h
dataframe = sampling_pool
dataframe = st_drop_geometry(sampling_pool)
dataframe = dataframe[,!(names(dataframe) %in% c("cell"))]
dataframe$strata_solution1 = factor(dataframe$strata_solution1)
lut = data.frame(strata_solution1 = levels(dataframe$strata_solution), picov = as.numeric(picov))
dataframe = merge(x = dataframe, y = lut)
dataframe = fastDummies::dummy_cols(dataframe, select_columns = c("soil_litho_50", 'strata_solution1'))
Xbal = dataframe[,!(names(dataframe) %in% c("strata_solution1", "soil_litho_50", 'picov', "dtm_tpi_50"))]
Xbal = as.matrix(sapply(Xbal, as.numeric))
Xspread = st_coordinates(sampling_pool)
set.seed(1234)
second_index <- BalancedSampling::lcube(Xbal = Xbal, Xspread = Xspread, prob = dataframe$picov)
second_iteration  = sampling_pool[second_index,]

table(second_iteration$strata_solution1)
st_write(second_iteration, 'sampling400_second.gpkg', append = F)
sampling_pool = sampling_pool[-second_index,]

N_h <-  table(sampling_pool$strata_solution1)
n_h <- n_samples
picov <- n_h / N_h
dataframe = sampling_pool
dataframe = st_drop_geometry(sampling_pool)
dataframe = dataframe[,!(names(dataframe) %in% c("cell"))]
dataframe$strata_solution1 = factor(dataframe$strata_solution1)
lut = data.frame(strata_solution1 = levels(dataframe$strata_solution), picov = as.numeric(picov))
dataframe = merge(x = dataframe, y = lut)
dataframe = fastDummies::dummy_cols(dataframe, select_columns = c("soil_litho_50", 'strata_solution1'))
Xbal = dataframe[,!(names(dataframe) %in% c("strata_solution1", "soil_litho_50", 'picov', "dtm_tpi_50"))]
Xbal = as.matrix(sapply(Xbal, as.numeric))
Xspread = st_coordinates(sampling_pool)
set.seed(1234)
third_index <- BalancedSampling::lcube(Xbal = Xbal, Xspread = Xspread, prob = dataframe$picov)
third_iteration  = sampling_pool[third_index,]

table(third_iteration$strata_solution1)
st_write(third_iteration, 'sampling400_third.gpkg', append = F)
sampling_pool = sampling_pool[-third_index,]

N_h <-  table(sampling_pool$strata_solution1)
n_h <- n_samples
picov <- n_h / N_h
dataframe = sampling_pool
dataframe = st_drop_geometry(sampling_pool)
dataframe = dataframe[,!(names(dataframe) %in% c("cell"))]
dataframe$strata_solution1 = factor(dataframe$strata_solution1)
lut = data.frame(strata_solution1 = levels(dataframe$strata_solution), picov = as.numeric(picov))
dataframe = merge(x = dataframe, y = lut)
dataframe = fastDummies::dummy_cols(dataframe, select_columns = c("soil_litho_50", 'strata_solution1'))
Xbal = dataframe[,!(names(dataframe) %in% c("strata_solution1", "soil_litho_50", 'picov', "dtm_tpi_50"))]
Xbal = as.matrix(sapply(Xbal, as.numeric))
Xspread = st_coordinates(sampling_pool)
set.seed(1234)
fourth_index <- BalancedSampling::lcube(Xbal = Xbal, Xspread = Xspread, prob = dataframe$picov)
fourth_iteration  = sampling_pool[fourth_index,]

table(fourth_iteration$strata_solution1)
st_write(fourth_iteration, 'sampling400_fourth.gpkg', append = F)
sampling_pool = sampling_pool[-fourth_index,]

N_h <-  table(sampling_pool$strata_solution1)
n_h <- n_samples
picov <- n_h / N_h
dataframe = sampling_pool
dataframe = st_drop_geometry(sampling_pool)
dataframe = dataframe[,!(names(dataframe) %in% c("cell"))]
dataframe$strata_solution1 = factor(dataframe$strata_solution1)
lut = data.frame(strata_solution1 = levels(dataframe$strata_solution), picov = as.numeric(picov))
dataframe = merge(x = dataframe, y = lut)
dataframe = fastDummies::dummy_cols(dataframe, select_columns = c("soil_litho_50", 'strata_solution1'))
Xbal = dataframe[,!(names(dataframe) %in% c("strata_solution1", "soil_litho_50", 'picov', "dtm_tpi_50"))]
Xbal = as.matrix(sapply(Xbal, as.numeric))
Xspread = st_coordinates(sampling_pool)
set.seed(1234)
fifth_index <- BalancedSampling::lcube(Xbal = Xbal, Xspread = Xspread, prob = dataframe$picov)
fifth_iteration  = sampling_pool[fifth_index,]

table(fifth_iteration$strata_solution1)
st_write(fifth_iteration, 'sampling400_fifth.gpkg', append = F)
sampling_pool = sampling_pool[-fifth_index,]
st_write(sampling_pool, 'sampling400_sixth.gpkg', append = F)