"""
Data processing utilities for WM-811K wafer dataset
Handles loading, preprocessing, and feature extraction
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WaferDataProcessor:
    """Main class for processing WM-811K wafer map dataset"""
    
    def __init__(self, data_path: str = "data/raw/"):
        """
        Initialize the data processor
        
        Args:
            data_path: Path to the raw data directory
        """
        self.data_path = data_path
        self.defect_classes = [
            "Center", "Donut", "Edge-Loc", "Edge-Ring", 
            "Loc", "Random", "Scratch", "Near-full"
        ]
        
    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Load the WM-811K dataset
        
        Returns:
            wafer_maps: Array of wafer map images
            labels: Array of defect class labels
            metadata: DataFrame with additional information
        """
        try:
            # Check if preprocessed data exists
            processed_path = "data/processed/"
            if os.path.exists(f"{processed_path}wafer_data.pkl"):
                return self._load_preprocessed_data()
            
            # Load raw data (placeholder - actual implementation depends on data format)
            logger.info("Loading raw WM-811K dataset...")
            
            # This would be the actual data loading logic
            # For now, create synthetic data for demonstration
            wafer_maps, labels, metadata = self._create_synthetic_data()
            
            # Save preprocessed data
            self._save_preprocessed_data(wafer_maps, labels, metadata)
            
            logger.info("Loaded %d wafer maps", len(wafer_maps))
            return wafer_maps, labels, metadata
            
        except Exception as e:
            logger.error("Error loading dataset: %s", e)
            raise
    
    def _create_synthetic_data(self) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Create synthetic data for demonstration purposes"""
        
        # Create synthetic wafer maps (52x52 is typical size)
        n_samples = 10000  # Subset for demo
        wafer_size = (52, 52)
        
        wafer_maps = np.random.rand(n_samples, *wafer_size)
        
        # Create synthetic labels
        labels = np.random.choice(len(self.defect_classes), n_samples)
        
        # Create metadata
        metadata = pd.DataFrame({
            'wafer_id': [f"W_{i:06d}" for i in range(n_samples)],
            'lot_id': [f"LOT_{i//100:03d}" for i in range(n_samples)],
            'defect_class': [self.defect_classes[label] for label in labels],
            'yield_rate': np.random.normal(0.85, 0.1, n_samples),
            'process_temp': np.random.normal(1000, 50, n_samples),
            'process_pressure': np.random.normal(1.5, 0.2, n_samples),
            'processing_time': np.random.normal(120, 10, n_samples),
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='h')
        })
        
        return wafer_maps, labels, metadata
    
    def _load_preprocessed_data(self) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Load preprocessed data from disk"""
        with open("data/processed/wafer_data.pkl", "rb") as f:
            data = pickle.load(f)
        return data['wafer_maps'], data['labels'], data['metadata']
    
    def _save_preprocessed_data(self, wafer_maps: np.ndarray, labels: np.ndarray, metadata: pd.DataFrame):
        """Save preprocessed data to disk"""
        os.makedirs("data/processed", exist_ok=True)
        
        data = {
            'wafer_maps': wafer_maps,
            'labels': labels,
            'metadata': metadata
        }
        
        with open("data/processed/wafer_data.pkl", "wb") as f:
            pickle.dump(data, f)
    
    def extract_wafer_features(self, wafer_maps: np.ndarray) -> pd.DataFrame:
        """
        Extract engineering features from wafer maps
        
        Args:
            wafer_maps: Array of wafer map images
            
        Returns:
            DataFrame with extracted features
        """
        features = []
        
        for wafer in wafer_maps:
            feature_dict = {
                'total_defects': np.sum(wafer > 0.5),  # Assuming >0.5 indicates defect
                'defect_density': np.mean(wafer > 0.5),
                'center_defects': np.sum(wafer[20:32, 20:32] > 0.5),  # Center region
                'edge_defects': self._count_edge_defects(wafer),
                'defect_clusters': self._count_clusters(wafer),
                'symmetry_score': self._calculate_symmetry(wafer),
                'radial_distribution': self._calculate_radial_distribution(wafer)
            }
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def _count_edge_defects(self, wafer: np.ndarray) -> int:
        """Count defects near the wafer edge"""
        edge_mask = np.zeros_like(wafer)
        edge_mask[:5, :] = 1  # Top edge
        edge_mask[-5:, :] = 1  # Bottom edge
        edge_mask[:, :5] = 1  # Left edge
        edge_mask[:, -5:] = 1  # Right edge
        
        return np.sum((wafer > 0.5) & (edge_mask == 1))
    
    def _count_clusters(self, wafer: np.ndarray) -> int:
        """Count number of defect clusters using simple connected components"""
        # Simplified cluster counting - in reality would use proper image processing
        defect_mask = wafer > 0.5
        return int(np.sum(defect_mask))  # Placeholder
    
    def _calculate_symmetry(self, wafer: np.ndarray) -> float:
        """Calculate symmetry score of defect pattern"""
        # Compare left-right symmetry
        left_half = wafer[:, :wafer.shape[1]//2]
        right_half = np.fliplr(wafer[:, wafer.shape[1]//2:])
        
        if left_half.shape != right_half.shape:
            right_half = right_half[:, :left_half.shape[1]]
        
        symmetry = 1 - np.mean(np.abs(left_half - right_half))
        return max(0.0, float(symmetry))
    
    def _calculate_radial_distribution(self, wafer: np.ndarray) -> float:
        """Calculate how defects are distributed radially from center"""
        center = (wafer.shape[0] // 2, wafer.shape[1] // 2)
        y, x = np.ogrid[:wafer.shape[0], :wafer.shape[1]]
        distances = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        # Calculate weighted average distance of defects
        defects = wafer > 0.5
        if np.sum(defects) == 0:
            return 0.0
        
        avg_distance = np.average(distances[defects])
        max_distance = np.sqrt(center[0]**2 + center[1]**2)
        
        return avg_distance / max_distance

def calculate_yield_metrics(metadata: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate various yield metrics from metadata
    
    Args:
        metadata: DataFrame with wafer information
        
    Returns:
        Dictionary with yield metrics
    """
    metrics = {
        'overall_yield': metadata['yield_rate'].mean(),
        'yield_std': metadata['yield_rate'].std(),
        'yield_min': metadata['yield_rate'].min(),
        'yield_max': metadata['yield_rate'].max(),
        'defect_rate': (metadata['yield_rate'] < 0.8).mean(),
        'high_yield_rate': (metadata['yield_rate'] > 0.9).mean()
    }
    
    # Yield by defect class
    for defect_class in metadata['defect_class'].unique():
        class_yield = metadata[metadata['defect_class'] == defect_class]['yield_rate'].mean()
        metrics[f'yield_{defect_class.lower()}'] = class_yield
    
    return metrics

def get_lot_statistics(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate statistics by manufacturing lot
    
    Args:
        metadata: DataFrame with wafer information
        
    Returns:
        DataFrame with lot-level statistics
    """
    lot_stats = metadata.groupby('lot_id').agg({
        'yield_rate': ['mean', 'std', 'count'],
        'process_temp': 'mean',
        'process_pressure': 'mean',
        'processing_time': 'mean',
        'defect_class': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
    }).round(4)
    
    # Flatten column names
    lot_stats.columns = ['_'.join(col).strip() for col in lot_stats.columns.values]
    lot_stats = lot_stats.reset_index()
    
    return lot_stats
