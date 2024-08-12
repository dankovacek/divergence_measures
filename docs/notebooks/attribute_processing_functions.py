import os
import time
import multiprocessing as mp
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
import xarray as xr
xr.set_options(keep_attrs=True)

import rasterio as rio
import rioxarray as rxr

from numba import jit
from scipy.stats.mstats import gmean

def retrieve_raster(filename):
    """
    Take in a file name and return the raster data, 
    the coordinate reference system, and the affine transform.
    """
    raster = rxr.open_rasterio(filename, mask_and_scale=True)
    crs = raster.rio.crs
    affine = raster.rio.transform(recalc=False)
    return raster, crs.to_epsg(), affine

@jit(nopython=True)
def process_slope_and_aspect(E, el_px, resolution, shape):
    # resolution = E.rio.resolution()
    # shape = E.rio.shape
    # note, distances are not meaningful in EPSG 4326
    # note, we can either do a costly reprojection of the dem
    # or just use the approximate resolution of 90x90m
    # dx, dy = 90, 90# resolution
    dx, dy = resolution
    # print(resolution)
    # print(asdfd)
    # dx, dy = 90, 90
    S, A = np.empty_like(E), np.empty_like(E)
    S[:] = np.nan # track slope (in degrees)
    A[:] = np.nan # track aspect (in degrees)
    # tot_p, tot_q = 0, 0
    for i, j in el_px:
        if (i == 0) | (j == 0) | (i == shape[0]) | (j == shape[1]):
            continue
            
        E_w = E[i-1:i+2, j-1:j+2]

        if E_w.shape != (3,3):
            continue

        a = E_w[0,0]
        b = E_w[1,0]
        c = E_w[2,0]
        d = E_w[0,1]
        f = E_w[2,1]
        g = E_w[0,2]
        h = E_w[1,2]
        # skip i and j because they're already used
        k = E_w[2,2]  

        all_vals = np.array([a, b, c, d, f, g, h, k])

        val_check = np.isfinite(all_vals)

        if np.all(val_check):
            p = ((c + 2*f + k) - (a + 2*d + g)) / (8 * abs(dx))
            q = ((c + 2*b + a) - (k + 2*h + g)) / (8 * abs(dy))
            cell_slope = np.sqrt(p*p + q*q)
            S[i, j] = (180 / np.pi) * np.arctan(cell_slope)
            A[i, j] = (180.0 / np.pi) * np.arctan2(q, p)

    return S, A


def calculate_circular_mean_aspect(a):
    """
    From RavenPy:
    https://github.com/CSHS-CWRA/RavenPy/blob/1b167749cdf5984545f8f79ef7d31246418a3b54/ravenpy/utilities/analysis.py#L118
    """
    angles = a[~np.isnan(a)]
    n = len(angles)
    sine_mean = np.divide(np.sum(np.sin(np.radians(angles))), n)
    cosine_mean = np.divide(np.sum(np.cos(np.radians(angles))), n)
    vector_mean = np.arctan2(sine_mean, cosine_mean)
    degrees = np.degrees(vector_mean)
    if degrees < 0:
        return degrees + 360
    else:
        return degrees


def calculate_slope_and_aspect(raster):  
    """Calculate mean basin slope and aspect 
    according to Hill (1981).

    Args:
        clipped_raster (array): dem raster

    Returns:
        slope, aspect: scalar mean values
    """

    resolution = raster.rio.resolution()
    raster_shape = raster[0].shape

    el_px = np.argwhere(np.isfinite(raster.data[0]))

    S, A = process_slope_and_aspect(raster.data[0], el_px, resolution, raster_shape)

    mean_slope_deg = np.nanmean(S)
    # should be within a hundredth of a degree or so.
    # print(f'my slope: {mean_slope_deg:.4f}, rdem: {np.nanmean(slope):.4f}')
    mean_aspect_deg = calculate_circular_mean_aspect(A)

    return mean_slope_deg, mean_aspect_deg

def check_lulc_sum(data):
    """
    Check if the sum of pct. land cover sums to 1.
    Return value is 1 - sum to correspond with 
    a more intuitive boolean flag, 
    i.e. data quality flags are 1 if the flag is raised,
    0 of no flag.
    """
    checksum = sum(list(data.values())) 
    lulc_check = 1-checksum
    if abs(lulc_check) >= 0.05:
        print(f'   ...checksum failed: {checksum:.3f}')   
    return lulc_check


def recategorize_lulc(data, year):    
    forest = (f'Land_Use_Forest_frac_{year}', [1, 2, 3, 4, 5, 6])
    shrub = (f'Land_Use_Shrubs_frac_{year}', [7, 8, 11])
    grass = (f'Land_Use_Grass_frac_{year}', [9, 10, 12, 13, 16])
    wetland = (f'Land_Use_Wetland_frac_{year}', [14])
    crop = (f'Land_Use_Crops_frac_{year}', [15])
    urban = (f'Land_Use_Urban_frac_{year}', [17])
    water = (f'Land_Use_Water_frac_{year}', [18])
    snow_ice = (f'Land_Use_Snow_Ice_frac_{year}', [19])
    lulc_dict = {}
    for label, p in [forest, shrub, grass, wetland, crop, urban, water, snow_ice]:
        prop_vals = round(sum([data[e] if e in data.keys() else 0.0 for e in p]), 2)
        lulc_dict[label] = prop_vals
    return lulc_dict
    

def get_value_proportions(data, year):
    # create a dictionary of land cover values by coverage proportion
    # assuming raster pixels are equally sized, we can keep the
    # raster in geographic coordinates and just count pixel ratios
    all_vals = data.data.flatten()
    vals = all_vals[~np.isnan(all_vals)]
    n_pts = len(vals)
    unique, counts = np.unique(vals, return_counts=True)    
    prop_dict = {k: 1.0*v/n_pts for k, v in zip(unique, counts)}
    prop_dict = recategorize_lulc(prop_dict, year)
    return prop_dict


def process_lulc(i, basin_geom, nalcms_raster_clipped, year):

    assert basin_geom.crs == nalcms_raster_clipped.rio.crs
    # checksum verifies proportions sum to 1
    prop_dict = get_value_proportions(nalcms_raster_clipped, year)
    lulc_check = check_lulc_sum(prop_dict)
    prop_dict[f'lulc_check_{year}'] = lulc_check
    return pd.DataFrame(prop_dict, index=[i])

def check_and_repair_geometries(in_feature):

    # avoid changing original geodf
    in_feature = in_feature.copy(deep=True)    
        
    # drop any missing geometries
    in_feature = in_feature[~(in_feature.is_empty)]
    
    # Repair broken geometries
    for index, row in in_feature.iterrows(): # Looping over all polygons
        if row['geometry'].is_valid:
            next
        else:
            fix = make_valid(row['geometry'])
            try:
                in_feature.loc[[index],'geometry'] =  fix # issue with Poly > Multipolygon
            except ValueError:
                in_feature.loc[[index],'geometry'] =  in_feature.loc[[index], 'geometry'].buffer(0)
    return in_feature

def process_basin_elevation(clipped_raster):
    # evaluate masked raster data
    values = clipped_raster.data.flatten()
    mean_val = np.nanmean(values)
    median_val = np.nanmedian(values)
    min_val = np.nanmin(values)
    max_val = np.nanmax(values)
    return mean_val, median_val, min_val, max_val


def get_soil_properties(merged, col):
    # dissolve polygons by unique parameter values
    geometries = check_and_repair_geometries(merged)

    df = geometries[[col, 'geometry']].copy().dissolve(by=col, aggfunc='first')
    df[col] = df.index.values
    # re-sum all shape areas
    df['Shape_Area'] = df.geometry.area
    # calculuate area fractions of each unique parameter value
    df['area_frac'] = df['Shape_Area'] / df['Shape_Area'].sum()
    # check that the total area fraction = 1
    total = round(df['area_frac'].sum(), 1)
    sum_check = total == 1.0
    if not sum_check:
        print(f'    Area proportions do not sum to 1: {total:.2f}')
        if np.isnan(total):
            return np.nan
        elif total < 0.9:
            return np.nan
    ###
    # 
    # NOTE: since the permeability values are already log transformed, 
    #       the weighted mean in log space is the geometric mean.
    #
    ####    
    return (df['area_frac'] * df[col]).sum()


def find_nearest_raster_value(raster, basin_polygon):
    # Convert the point to the raster's CRS
    centroid = basin_polygon.geometry.centroid.iloc[0]
    x, y = centroid.x, centroid.y
    # Extract the value at the nearest pixel
    # Use the nearest neighbor to select the value at the closest grid point
    nearest_value = raster.sel(
        x=x,
        y=y,
        method="nearest"
    ).item()
        
    return nearest_value


def clip_raster_to_basin(clipping_polygon, raster):
    bounds = tuple(clipping_polygon.bounds.values[0])
    try:
        # clip first to bounding box, then to polygon for better performance
        subset_raster = raster.rio.clip_box(*bounds).copy()
        clipped_raster = subset_raster.rio.clip(
            clipping_polygon.geometry, 
            clipping_polygon.crs.to_epsg(),
            all_touched=True,
            )
        return True, clipped_raster
    except Exception as e:
        print(e)
        return False, None
    

def process_glhymps(basin_geom, fpath):
    # clipped soil layer is in 3005 (it's in the filename)
    # basin_geom = basin_geom.to_crs(4326)
    # returns INTERSECTION
    gdf = gpd.read_file(fpath, mask=basin_geom)
    # now clip precisely to the basin polygon bounds
    merged = gpd.clip(gdf, mask=basin_geom)
    # now reproject to minimize spatial distortion
    # merged = merged.to_crs(3005)
    return merged
    

def get_nalcms_data(year, fpath):
    nalcms, nalcms_crs, nalcms_affine = retrieve_raster(fpath)
    if not nalcms_crs:
        nalcms_crs = nalcms.rio.crs.to_wkt()
    return nalcms, nalcms_crs, nalcms_affine


def clip_and_reproject_NALCMS(data_folder, output_folder, input_nalcms_fname, mask_path, reproj_nalcms_file, year):
    input_nalcms_fpath = os.path.join(data_folder, input_nalcms_fname)
    reproj_nalcms_fpath = os.path.join(output_folder, reproj_nalcms_file)
    reproj_mask_nalcms_crs = os.path.join(data_folder, 'convex_hull_nalcms_crs.shp')
    if not os.path.exists(reproj_nalcms_fpath):
    
        if not os.path.exists(input_nalcms_fpath):
            raise Exception('Download and unzip the NALCMS data, see the README for details.')
        
        nalcms_data = rxr.open_rasterio(input_nalcms_fpath)
        nalcms_wkt = nalcms_data.rio.crs.wkt
        
        # get the mask geometry and reproject it using the original NALCMS projection
        if not os.path.exists(reproj_mask_nalcms_crs):
            mask = gpd.read_file(mask_path).to_crs(nalcms_wkt)
            mask = mask.convex_hull
            mask.to_file(reproj_mask_nalcms_crs)
        
        # first clip the raster, 
        print('Clipping NALCMS raster to region bounds.')
        clipped_nalcms_path = os.path.join(data_folder, f'NA_NALCMS_landcover_{year}_clipped.tif')
        clip_command = f"gdalwarp -s_srs '{nalcms_wkt}' -cutline {reproj_mask_nalcms_crs} -crop_to_cutline -multi -of gtiff {input_nalcms_fpath} {clipped_nalcms_path} -wo NUM_THREADS=ALL_CPUS"
        if not os.path.exists(clipped_nalcms_path):
            os.system(clip_command)
        
        print('\nReprojecting clipped NALCMS raster.')
        # insert  "-co COMPRESS=LZW" in the command below to use compression (slower but much smaller file size)
        warp_command = f"gdalwarp -q -s_srs '{nalcms_wkt}' -t_srs EPSG:3005 -of gtiff {clipped_nalcms_path} {reproj_nalcms_fpath} -r bilinear -wo NUM_THREADS=ALL_CPUS"
        print(warp_command)
        if not os.path.exists(reproj_nalcms_fpath):
            os.system(warp_command) 
        
        # remove the intermediate step
        if os.path.exists(clipped_nalcms_path):
            os.remove(clipped_nalcms_path)
