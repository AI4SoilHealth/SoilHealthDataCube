import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import KDTree
from scipy.stats import norm
from scipy.spatial.distance import cdist


def bw_cvl(points, srange=None, ns=16, sigma=None):
    """
    Bandwidth selection using the Cronie and Van Lieshout (CvL) criterion.
    Parameters:
        points (GeoDataFrame): GeoDataFrame containing point geometries.
        srange (tuple): Range of bandwidths (min, max).
        ns (int): Number of bandwidths to evaluate.
        sigma (list): Specific bandwidths to evaluate (overrides srange and ns).
    Returns:
        dict: Optimal bandwidth and corresponding CvL score.
    """

    # Extract coordinates
    coords = np.array(list(points.geometry.apply(lambda g: (g.x, g.y))))

    # Calculate study area
    area_w = points.geometry.unary_union.convex_hull.area

    # Define bandwidths
    if sigma is not None:
        bandwidths = np.array(sigma)
    else:
        if srange is None:
            nnd = np.sort(cdist(coords, coords, 'euclidean'), axis=1)[:, 1]
            srange = (np.min(nnd[nnd > 0]), np.max(nnd))
        bandwidths = np.geomspace(srange[0], srange[1], num=ns)

    # Initialize KDTree
    tree = KDTree(coords)

    # Calculate CvL scores for each bandwidth
    cvl_scores = []
    for bw in bandwidths:
        # Kernel density estimation at points
        densities = []
        for coord in coords:
            distances, _ = tree.query(coord, k=len(coords))
            kernel_values = norm.pdf(distances, scale=bw)
            density = np.sum(kernel_values)
            densities.append(density)
        
        densities = np.array(densities)

        # Avoid division by zero
        densities = np.where(densities == 0, np.finfo(float).eps, densities)

        # CvL score
        cvl = (np.sum(1 / densities) - area_w) ** 2
        cvl_scores.append(cvl)

    # Find optimal bandwidth
    optimal_idx = np.argmin(cvl_scores)
    optimal_bw = bandwidths[optimal_idx]
    optimal_cvl = cvl_scores[optimal_idx]

    return {
        "optimal_bandwidth": optimal_bw,
        "optimal_cvl_score": optimal_cvl,
        "bandwidths": bandwidths,
        "cvl_scores": cvl_scores,
    }

import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np

def inverse_density_weight(gdf, grid_size=2700):
    """
    Divides the points into a square grid of a specified size, counts the number
    of points in each grid cell, and assigns weights (1/n) to each point.

    Parameters:
        gdf (GeoDataFrame): GeoDataFrame with point geometries.
        grid_size (float): Length of the grid squares in the CRS units (default 2700m).

    Returns:
        GeoDataFrame: GeoDataFrame with an additional column 'weights'.
    """
    # Ensure the GeoDataFrame has a projected CRS
    if gdf.crs is None or not gdf.crs.is_projected:
        raise ValueError("GeoDataFrame must have a projected CRS.")

    # Calculate the bounds of the entire GeoDataFrame
    minx, miny, maxx, maxy = gdf.total_bounds

    # Create the grid cells
    x_coords = np.arange(minx, maxx, grid_size)
    y_coords = np.arange(miny, maxy, grid_size)

    grid_cells = []
    for x in x_coords:
        for y in y_coords:
            grid_cells.append(
                Polygon([(x, y), (x + grid_size, y), (x + grid_size, y + grid_size), (x, y + grid_size)])
            )

    # Create a GeoDataFrame for the grid
    grid = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=gdf.crs)

    # Perform spatial join to count points in each grid cell
    joined = gpd.sjoin(gdf, grid, how="left", predicate="within")
    point_counts = joined.groupby('index_right').size()

    # Map counts back to the grid GeoDataFrame
    grid['point_count'] = grid.index.map(point_counts).fillna(0).astype(int)

    # Merge the counts back to the original points
    gdf = gdf.join(joined[['index_right']])
    gdf['point_count'] = gdf['index_right'].map(point_counts).fillna(0).astype(int)

    # Assign weights as 1 / n (handle divide by zero gracefully)
    gdf['weights'] = gdf['point_count'].apply(lambda n: 1 / n if n > 0 else 0)

    # Drop unnecessary columns
    gdf = gdf.drop(columns=['index_right'], errors='ignore')

    return gdf


# def density_ppp(points, bandwidth, weights=None, kernel="gaussian"):
#     """
#     Compute density estimates for points using kernel density estimation.
    
#     Parameters:
#         points (GeoDataFrame): GeoDataFrame with point geometries.
#         bandwidth (float): Bandwidth (kernel size) for the Gaussian kernel.
#         weights (array-like, optional): Weights for each point. Defaults to None.
#         kernel (str): Kernel type, currently supports "gaussian". Defaults to "gaussian".
    
#     Returns:
#         np.array: Density estimates for the input points.
#     """
    
#     # Ensure the input GeoDataFrame has a geometry column
#     if "geometry" not in points.columns:
#         raise ValueError("Input GeoDataFrame must contain a geometry column with Point geometries.")

#     # Extract coordinates from GeoDataFrame
#     coords = np.array(list(points.geometry.apply(lambda geom: (geom.x, geom.y))))

#     # Check for zero-area case (all points lie in the same location)
#     if np.ptp(coords, axis=0).max() == 0:
#         raise ValueError("All points are located at the same location, resulting in zero area.")

#     # Initialize weights
#     if weights is None:
#         weights = np.ones(len(points))
#     weights = np.asarray(weights)

#     # Build KDTree for efficient nearest-neighbor searches
#     tree = KDTree(coords)

#     # Compute densities for each point
#     densities = []
#     for coord in coords:
#         # Query all distances for the given point
#         distances, indices = tree.query(coord, k=len(coords))
        
#         # Apply kernel function
#         if kernel == "gaussian":
#             kernel_values = norm.pdf(distances, scale=bandwidth)
#         else:
#             raise ValueError("Currently only the Gaussian kernel is supported.")

#         # Compute weighted density
#         density = np.sum(weights[indices] * kernel_values) / bandwidth
#         densities.append(density)

#     return np.array(densities)
