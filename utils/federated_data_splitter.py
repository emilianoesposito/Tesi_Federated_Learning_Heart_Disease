# utils/federated_data_splitter.py
# -*- coding: utf-8 -*-
"""
Federated data splitter for Veneto employment centers.
Splits training dataset by geographic regions based on candidate residence areas.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VenetoFederatedSplitter:
    """
    Geographic data splitter for Veneto region employment centers.
    
    Divides training dataset into regional subsets based on candidate 
    residence areas, simulating real-world federated learning scenario
    where each Employment Center (CPI) has access only to local data.
    """
    
    def __init__(self):
        self.regions = {
            'CPI_Verona': ['Verona', 'Villafranca', 'Legnago', 'San Bonifacio', 'Isola della Scala'],
            'CPI_Vicenza': ['Vicenza', 'Bassano', 'Thiene', 'Arzignano', 'Schio'],
            'CPI_Padova': ['Padova', 'Cittadella', 'Piove di Sacco', 'Camposampiero', 'Este'],
            'CPI_Treviso': ['Treviso', 'Castelfranco', 'Conegliano', 'Montebelluna', 'Oderzo'],
            'CPI_Venezia': ['Venezia', 'Mestre', 'Portogruaro', 'San DonÃ  di Piave', 'Chioggia']
        }
        
        # Create reverse mapping for quick lookup
        self.city_to_region = {}
        for region, cities in self.regions.items():
            for city in cities:
                self.city_to_region[city.lower()] = region
        
        # Define realistic population weights for simulation
        self.city_population_weights = {
            # Verona province
            'Verona': 0.15, 'Villafranca': 0.04, 'Legnago': 0.025,
            'San Bonifacio': 0.02, 'Isola della Scala': 0.015,
            
            # Vicenza province  
            'Vicenza': 0.12, 'Bassano': 0.05, 'Thiene': 0.035,
            'Arzignano': 0.03, 'Schio': 0.025,
            
            # Padova province
            'Padova': 0.18, 'Cittadella': 0.035, 'Piove di Sacco': 0.025,
            'Camposampiero': 0.02, 'Este': 0.015,
            
            # Treviso province
            'Treviso': 0.10, 'Castelfranco': 0.035, 'Conegliano': 0.03,
            'Montebelluna': 0.025, 'Oderzo': 0.015,
            
            # Venezia province
            'Venezia': 0.06, 'Mestre': 0.09, 'Portogruaro': 0.02,
            'San DonÃ  di Piave': 0.025, 'Chioggia': 0.015
        }
                
        logger.info(f"Initialized splitter with {len(self.regions)} regions and {len(self.city_to_region)} cities")

    def _extract_city_from_address(self, address: str) -> str:
        """
        Extract city name from address string.
        
        Args:
            address: Full address string
            
        Returns:
            Extracted city name or 'Unknown' if not found
        """
        if pd.isna(address) or not isinstance(address, str):
            return 'Unknown'
            
        address_clean = address.lower().strip()
        
        # Try to match known cities
        for city in self.city_to_region.keys():
            if city in address_clean:
                return city.title()
                
        return 'Unknown'

    def _assign_region(self, city: str) -> str:
        """
        Assign region based on city name.
        
        Args:
            city: City name
            
        Returns:
            Region name or 'CPI_Unknown' if city not found
        """
        city_lower = city.lower()
        return self.city_to_region.get(city_lower, 'CPI_Unknown')

    def split_by_geography(self, df_train: pd.DataFrame, 
                          address_column: str = 'candidate_residence',
                          min_samples_per_region: int = 1000) -> Dict[str, pd.DataFrame]:
        """
        Split training dataset by geographic regions.
        
        Args:
            df_train: Training dataset with candidate-company pairs
            address_column: Column name containing residence addresses
            min_samples_per_region: Minimum samples required per region
            
        Returns:
            Dictionary mapping region names to regional datasets
        """
        logger.info(f"Starting geographic split of {len(df_train)} training samples")
        
        # Create copy to avoid modifying original data
        df_work = df_train.copy()
        
        # Extract city information (simulate from existing data)
        if address_column not in df_work.columns:
            logger.warning(f"Address column '{address_column}' not found. Simulating geographic distribution.")
            df_work = self._simulate_geographic_distribution(df_work)
        
        # Extract cities and assign regions
        df_work['city'] = df_work[address_column].apply(self._extract_city_from_address)
        df_work['region'] = df_work['city'].apply(self._assign_region)
        
        # Split into regional datasets
        regional_datasets = {}
        region_stats = {}
        
        for region in self.regions.keys():
            region_data = df_work[df_work['region'] == region].copy()
            
            if len(region_data) >= min_samples_per_region:
                # Remove helper columns before saving
                region_data = region_data.drop(columns=['city', 'region', 'candidate_residence'], errors='ignore')
                regional_datasets[region] = region_data
                region_stats[region] = len(region_data)
                logger.info(f"{region}: {len(region_data)} samples")
            else:
                logger.warning(f"{region}: {len(region_data)} samples (below minimum {min_samples_per_region})")
        
        # Handle unknown region
        unknown_data = df_work[df_work['region'] == 'CPI_Unknown']
        if len(unknown_data) > 0:
            unknown_clean = unknown_data.drop(columns=['city', 'region', 'candidate_residence'], errors='ignore')
            regional_datasets['CPI_Unknown'] = unknown_clean
            region_stats['CPI_Unknown'] = len(unknown_data)
            logger.info(f"CPI_Unknown: {len(unknown_data)} samples")
        
        # Print summary statistics
        total_assigned = sum(region_stats.values())
        logger.info(f"Split summary: {total_assigned}/{len(df_train)} samples assigned to {len(regional_datasets)} regions")
        
        return regional_datasets

    def _simulate_geographic_distribution(self, df_train: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate geographic distribution for existing training data.
        
        Args:
            df_train: Training dataset
            
        Returns:
            Dataset with simulated residence addresses
        """
        logger.info("Simulating geographic distribution based on Veneto demographics")
        
        df_sim = df_train.copy()
        n_samples = len(df_sim)
        
        # Use predefined population weights
        cities = list(self.city_population_weights.keys())
        weights = list(self.city_population_weights.values())
        
        # Normalize weights to ensure they sum to exactly 1.0
        weights = np.array(weights, dtype=float)
        weights = weights / weights.sum()
        
        # Verify weights sum to 1
        weights_sum = weights.sum()
        logger.info(f"Population weights sum: {weights_sum:.8f}")
        
        if abs(weights_sum - 1.0) > 1e-10:
            logger.warning(f"Weights sum deviation: {abs(weights_sum - 1.0):.2e}")
            # Force exact normalization
            weights[-1] = 1.0 - weights[:-1].sum()
            logger.info(f"After correction, weights sum: {weights.sum():.8f}")
        
        # Sample cities according to weights
        np.random.seed(42)  # For reproducibility
        sampled_cities = np.random.choice(cities, size=n_samples, p=weights)
        
        # Create realistic addresses
        df_sim['candidate_residence'] = [f"{city}, Veneto, Italy" for city in sampled_cities]
        
        # Log distribution
        city_counts = pd.Series(sampled_cities).value_counts()
        logger.info(f"Generated geographic distribution for {n_samples} samples")
        logger.info("Top 5 cities by sample count:")
        for city, count in city_counts.head().items():
            percentage = (count / n_samples) * 100
            logger.info(f"  {city}: {count} samples ({percentage:.1f}%)")
        
        return df_sim

    def get_region_statistics(self, regional_datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate statistics about regional data distribution.
        
        Args:
            regional_datasets: Dictionary of regional datasets
            
        Returns:
            DataFrame with regional statistics
        """
        stats_data = []
        
        for region, dataset in regional_datasets.items():
            if len(dataset) > 0:
                stats = {
                    'Region': region,
                    'Total_Samples': len(dataset),
                    'Positive_Matches': dataset['outcome'].sum() if 'outcome' in dataset.columns else 0,
                    'Match_Rate': dataset['outcome'].mean() if 'outcome' in dataset.columns else 0,
                    'Avg_Distance_KM': dataset['distance_km'].mean() if 'distance_km' in dataset.columns else 0,
                    'Avg_Attitude_Score': dataset['attitude_score'].mean() if 'attitude_score' in dataset.columns else 0
                }
                stats_data.append(stats)
        
        stats_df = pd.DataFrame(stats_data)
        return stats_df.round(3)

    def save_regional_datasets(self, regional_datasets: Dict[str, pd.DataFrame], 
                             output_dir: str = 'data/federated') -> None:
        """
        Save regional datasets to separate files.
        
        Args:
            regional_datasets: Dictionary of regional datasets
            output_dir: Output directory for regional datasets
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for region, dataset in regional_datasets.items():
            filepath = os.path.join(output_dir, f"{region}_training_data.csv")
            dataset.to_csv(filepath, index=False)
            logger.info(f"Saved {region}: {len(dataset)} samples to {filepath}")
        
        # Save region statistics
        stats_df = self.get_region_statistics(regional_datasets)
        stats_path = os.path.join(output_dir, 'regional_statistics.csv')
        stats_df.to_csv(stats_path, index=False)
        logger.info(f"Saved regional statistics to {stats_path}")

    def load_regional_datasets(self, input_dir: str = 'data/federated') -> Dict[str, pd.DataFrame]:
        """
        Load previously saved regional datasets.
        
        Args:
            input_dir: Directory containing regional datasets
            
        Returns:
            Dictionary of regional datasets
        """
        regional_datasets = {}
        
        if not os.path.exists(input_dir):
            logger.error(f"Input directory {input_dir} does not exist")
            return regional_datasets
        
        for region in self.regions.keys():
            filepath = os.path.join(input_dir, f"{region}_training_data.csv")
            if os.path.exists(filepath):
                dataset = pd.read_csv(filepath)
                regional_datasets[region] = dataset
                logger.info(f"Loaded {region}: {len(dataset)} samples from {filepath}")
            else:
                logger.warning(f"File not found: {filepath}")
        
        return regional_datasets


def main():
    """
    Test function for the federated splitter.
    """
    # This would be called from the main federated training script
    logger.info("Testing VenetoFederatedSplitter...")
    
    splitter = VenetoFederatedSplitter()
    
    # Load training data (example)
    training_file = 'data/processed/Enhanced_Training_Dataset.csv'
    if os.path.exists(training_file):
        df_train = pd.read_csv(training_file)
        logger.info(f"Loaded training data: {len(df_train)} samples")
        
        # Split by geography
        regional_datasets = splitter.split_by_geography(df_train)
        
        # Save regional datasets
        splitter.save_regional_datasets(regional_datasets)
        
        # Show statistics
        stats_df = splitter.get_region_statistics(regional_datasets)
        print("\nRegional Statistics:")
        print(stats_df.to_string(index=False))
        
    else:
        logger.error(f"Training file not found: {training_file}")


if __name__ == "__main__":
    main()