import osmnx as ox
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

def extract_building_features(b_in_cell, cell_geometry):
    """
    Extracts geometric and morphometric features from a collection of buildings
    inside a specific grid cell.
    """
    features = {
        'building_count': 0,
        'building_density': 0.0,
        'avg_building_area': 0.0,
        'avg_building_perimeter': 0.0,
        'shape_complexity_ratio': 0.0,
        'courtyard_index': 0.0,
        'std_building_area': 0.0,
    }
    
    building_count = len(b_in_cell)
    if building_count == 0:
        return features
        
    cell_area = cell_geometry.area
    building_areas = b_in_cell.geometry.area
    building_perimeters = b_in_cell.geometry.length
    
    # 1. Basic counts and densities
    features['building_count'] = building_count
    features['building_density'] = building_areas.sum() / cell_area
    features['avg_building_area'] = building_areas.mean()
    features['avg_building_perimeter'] = building_perimeters.mean()
    features['avg_building_area'] = building_areas.mean()
    # 2. Shape Complexity (Perimeter / Area)
    
    features['shape_complexity_ratio'] = (building_perimeters / (building_areas + 1e-5)).mean()

    try:
        convex_hulls = b_in_cell.geometry.convex_hull
        features['courtyard_index'] = (building_areas / (convex_hulls.area + 1e-5)).mean()
    except Exception:
        features['courtyard_index'] = 1.0 # Default fallback if geometry error occurs

    return features

