def extract_road_features(r_in_cell):
    features = {
        'road_segment_count': 0,
        'total_road_length': 0.0,
        'avg_road_segment_length': 0.0,
        'road_density_ratio': 0.0
    }
    road_count = len(r_in_cell)
    if road_count == 0:
        return features
        
    road_lengths = r_in_cell.geometry.length
    
    features['road_segment_count'] = road_count
    features['total_road_length'] = road_lengths.sum()
    features['avg_road_segment_length'] = road_lengths.mean()
    features['road_density_ratio'] = road_count / (road_lengths.sum() + 1e-5)
    
    return features