import numpy as np
import pandas as pd

def extract_building_features(buildings_df, cell_geometry):
    if buildings_df.empty:
        return {
            'building_count': 0,
            'building_density': 0.0,
            'avg_building_area': 0.0,
            'std_building_area': 0.0,
            'shape_complexity_ratio': 1.0,
            'courtyard_index': 0.0,
            'avg_building_elongation': 1.0
        }
    
    b_count = len(buildings_df)
    cell_area = cell_geometry.area
    
    buildings_clipped = buildings_df.clip(cell_geometry)
    total_b_area = buildings_clipped.geometry.area.sum()
    b_density = total_b_area / cell_area if cell_area > 0 else 0.0
    
    raw_areas = buildings_df.geometry.area
    avg_area = raw_areas.mean()
    std_area = raw_areas.std() if b_count > 1 else 0.0
    
    perimeters = buildings_df.geometry.length
    complexity = perimeters / (4 * np.sqrt(raw_areas) + 1e-5)
    avg_complexity = complexity.mean()
    
    convex_areas = buildings_df.geometry.convex_hull.area
    courtyard_idx = (convex_areas - raw_areas) / (raw_areas + 1e-5)
    avg_courtyard = courtyard_idx.mean()
    
    elongations = []
    for geom in buildings_df.geometry:
        try:
            rot_rect = geom.minimum_rotated_rectangle
            x, y = rot_rect.exterior.coords.xy
            edge_lengths = [
                np.sqrt((x[i] - x[i+1])**2 + (y[i] - y[i+1])**2)
                for i in range(3)
            ]
            side1, side2 = edge_lengths[0], edge_lengths[1]
            
            if min(side1, side2) > 0:
                elongations.append(max(side1, side2) / min(side1, side2))
            else:
                elongations.append(1.0)
        except:
            elongations.append(1.0)
            
    avg_elongation = np.mean(elongations) if elongations else 1.0
    
    return {
        'building_count': b_count,
        'building_density': b_density,
        'avg_building_area': avg_area,
        'std_building_area': std_area,
        'shape_complexity_ratio': avg_complexity,
        'courtyard_index': avg_courtyard,
        'avg_building_elongation': avg_elongation
    }