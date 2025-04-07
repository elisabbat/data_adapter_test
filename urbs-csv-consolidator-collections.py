import os
import pandas as pd
import argparse
from pathlib import Path
import re
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class UrbsDataConverter:
    """
    Class to read CSV files from directories inside the 'collections' directory
    and convert them to urbs Excel format.
    """
    # Define urbs required sheet names and their structures
    URBS_SHEETS = {
        'global': ['param', 'value', 'unit', 'description'],
        'site': ['site', 'lat', 'lon', 'area'],
        'commodity': ['site', 'commodity', 'type', 'price', 'max', 'maxperhour', 'co2'],
        'process': ['site', 'process', 'commodity-in', 'commodity-out', 'cap-min', 'cap-max', 'max-grad', 
                    'min-load', 'inv-cost', 'fix-cost', 'var-cost', 'wacc', 'depreciation', 'area-per-cap'],
        'storage': ['site', 'storage', 'commodity', 'cap-min', 'cap-max', 'eff-in', 'eff-out', 
                    'self-discharge', 'inv-cost-p', 'inv-cost-e', 'fix-cost-p', 'fix-cost-e', 
                    'var-cost-p', 'var-cost-e', 'wacc', 'depreciation', 'initial'],
        'transmission': ['site-in', 'site-out', 'transmission', 'commodity', 'cap-min', 'cap-max', 
                         'eff', 'inv-cost', 'fix-cost', 'var-cost', 'wacc', 'depreciation', 'distance']
    }
    
    def __init__(self, collections_directory="collections", output_file="urbs_input.xlsx", time_series_pattern=None):
        """
        Initialize the converter with collections directory path and output file.
        
        Args:
            collections_directory: Path to the 'collections' directory containing subdirectories with CSV files
            output_file: Path to save the output Excel file
            time_series_pattern: Regex pattern to identify time series CSV files
        """
        self.collections_directory = Path(collections_directory)
        self.output_file = Path(output_file)
        self.time_series_pattern = time_series_pattern or r'(demand|supim|solar|wind|hydro)_.*\.csv$'
        self.csv_files = []
        self.dataframes = {}
        self.time_series = {}
        
    def find_csv_files(self):
        """Recursively find all CSV files in the collections directory and its subdirectories."""
        logger.info(f"Searching for CSV files in {self.collections_directory}")
        
        # Check if collections directory exists
        if not self.collections_directory.exists():
            logger.error(f"Collections directory '{self.collections_directory}' does not exist.")
            return False
            
        for root, _, files in os.walk(self.collections_directory):
            for file in files:
                if file.lower().endswith('.csv'):
                    file_path = Path(root) / file
                    self.csv_files.append(file_path)
        
        logger.info(f"Found {len(self.csv_files)} CSV files in collections directory")
        return len(self.csv_files) > 0
    
    def categorize_and_read_files(self):
        """
        Categorize CSV files as either standard urbs sheets or time series data
        and read them into dataframes.
        """
        if not self.csv_files:
            logger.warning("No CSV files found. Please run find_csv_files() first.")
            return False
        
        for file_path in self.csv_files:
            file_name = file_path.stem.lower()
            
            try:
                # Check if this is a time series file
                if re.search(self.time_series_pattern, file_path.name.lower()):
                    df = pd.read_csv(file_path, sep=None, engine='python')
                    self.time_series[file_name] = df
                    logger.info(f"Read time series file: {file_path.name}")
                else:
                    # Try to match with urbs sheet names
                    matched = False
                    for sheet_name in self.URBS_SHEETS:
                        if sheet_name in file_name:
                            df = pd.read_csv(file_path, sep=None, engine='python')
                            if sheet_name not in self.dataframes:
                                self.dataframes[sheet_name] = df
                            else:
                                # Append to existing dataframe if the structure matches
                                if set(df.columns).issubset(set(self.dataframes[sheet_name].columns)) or \
                                   set(self.dataframes[sheet_name].columns).issubset(set(df.columns)):
                                    common_cols = list(set(df.columns) & set(self.dataframes[sheet_name].columns))
                                    self.dataframes[sheet_name] = pd.concat(
                                        [self.dataframes[sheet_name][common_cols], df[common_cols]], 
                                        ignore_index=True
                                    )
                            matched = True
                            logger.info(f"Read standard sheet file: {file_path.name} as {sheet_name}")
                            break
                    
                    if not matched:
                        # If no match, try to infer the category from content
                        df = pd.read_csv(file_path, sep=None, engine='python')
                        sheet_name = self.infer_sheet_category(df)
                        if sheet_name:
                            if sheet_name not in self.dataframes:
                                self.dataframes[sheet_name] = df
                            else:
                                # Append to existing dataframe
                                common_cols = list(set(df.columns) & set(self.dataframes[sheet_name].columns))
                                if common_cols:
                                    self.dataframes[sheet_name] = pd.concat(
                                        [self.dataframes[sheet_name][common_cols], df[common_cols]], 
                                        ignore_index=True
                                    )
                            logger.info(f"Inferred sheet category for {file_path.name} as {sheet_name}")
                        else:
                            # If still no match, use the filename as sheet name
                            self.dataframes[file_name] = df
                            logger.info(f"Used filename as sheet name for {file_path.name}")
                
            except Exception as e:
                logger.error(f"Error reading {file_path.name}: {e}")
        
        return len(self.dataframes) > 0 or len(self.time_series) > 0
    
    def infer_sheet_category(self, df):
        """
        Infer the urbs sheet category from the dataframe content.
        
        Args:
            df: Pandas DataFrame to categorize
            
        Returns:
            String with the inferred category or None if no match
        """
        # Check if columns match any of the urbs sheet structures
        columns_set = set(df.columns.str.lower())
        
        for sheet_name, expected_columns in self.URBS_SHEETS.items():
            expected_set = set([col.lower() for col in expected_columns])
            if columns_set.intersection(expected_set):
                # If at least some columns match, return the sheet name
                match_ratio = len(columns_set.intersection(expected_set)) / len(expected_set)
                if match_ratio > 0.3:  # At least 30% of columns match
                    return sheet_name
        
        # Check for specific keywords in column names
        column_str = ' '.join(df.columns.str.lower())
        if 'site' in column_str and 'commodity' in column_str:
            return 'commodity'
        elif 'site' in column_str and 'process' in column_str:
            return 'process'
        elif 'storage' in column_str:
            return 'storage'
        elif 'transmission' in column_str or ('site-in' in column_str and 'site-out' in column_str):
            return 'transmission'
        
        return None
    
    def process_time_series(self):
        """
        Process and organize time series data.
        
        Returns:
            Dictionary of time series dataframes organized by type
        """
        organized_ts = {
            'demand': pd.DataFrame(),
            'supim': pd.DataFrame(),
            'solar': pd.DataFrame(),
            'wind': pd.DataFrame(),
            'hydro': pd.DataFrame()
        }
        
        for name, df in self.time_series.items():
            # Determine which type of time series this is
            for ts_type in organized_ts.keys():
                if ts_type in name.lower():
                    # Check if we need to transpose the data
                    if df.shape[0] > df.shape[1] and df.shape[0] > 24:  # Likely time series in rows
                        # First column might be timestamps
                        if df.iloc[:, 0].dtype == 'object' and pd.api.types.is_datetime64_any_dtype(pd.to_datetime(df.iloc[:, 0], errors='coerce')):
                            df = df.set_index(df.columns[0])
                    
                    # If the dataframe is already in the right format (times in rows, sites in columns)
                    if organized_ts[ts_type].empty:
                        organized_ts[ts_type] = df
                    else:
                        # Try to append columns
                        organized_ts[ts_type] = pd.concat([organized_ts[ts_type], df], axis=1)
                    break
        
        # Remove empty dataframes
        return {k: v for k, v in organized_ts.items() if not v.empty}
    
    def standardize_dataframes(self):
        """
        Standardize dataframes to match urbs expected format.
        """
        for sheet_name in list(self.dataframes.keys()):
            df = self.dataframes[sheet_name]
            
            # If this is a known urbs sheet, ensure it has the required columns
            if sheet_name in self.URBS_SHEETS:
                required_cols = self.URBS_SHEETS[sheet_name]
                
                # Convert column names to lowercase for case-insensitive matching
                df.columns = [col.lower() for col in df.columns]
                
                # Add missing columns with None values
                for col in required_cols:
                    if col.lower() not in df.columns:
                        df[col.lower()] = None
                
                # Remove duplicate columns
                df = df.loc[:, ~df.columns.duplicated()]
                
                # Only keep required columns
                df = df[[col.lower() for col in required_cols if col.lower() in df.columns]]
                
                self.dataframes[sheet_name] = df
    
    def create_master_excel(self):
        """
        Create a master Excel file with all the dataframes.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Standardize dataframes before writing
            self.standardize_dataframes()
            
            # Process time series data
            organized_ts = self.process_time_series()
            
            # Create Excel writer
            logger.info(f"Creating Excel file: {self.output_file}")
            with pd.ExcelWriter(self.output_file, engine='openpyxl') as writer:
                # Write standard sheets
                for sheet_name, df in self.dataframes.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    logger.info(f"Wrote sheet: {sheet_name} with {len(df)} rows")
                
                # Write time series sheets
                for ts_type, df in organized_ts.items():
                    df.to_excel(writer, sheet_name=ts_type, index=True)
                    logger.info(f"Wrote time series sheet: {ts_type} with {df.shape[0]} timestamps and {df.shape[1]} columns")
            
            logger.info(f"Successfully created Excel file: {self.output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating Excel file: {e}")
            return False
    
    def run(self):
        """Run the complete conversion process."""
        if self.find_csv_files():
            if self.categorize_and_read_files():
                return self.create_master_excel()
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert CSV files from collections directory to urbs Excel format')
    parser.add_argument('--collections-dir', '-c', type=str, default='collections', 
                        help='Path to the collections directory (default: collections)')
    parser.add_argument('--output-file', '-o', type=str, default='urbs_input.xlsx', 
                        help='Output Excel file path (default: urbs_input.xlsx)')
    parser.add_argument('--time-series-pattern', '-t', type=str, 
                        default=r'(demand|supim|solar|wind|hydro)_.*\.csv$',
                        help='Regex pattern to identify time series CSV files')
    parser.add_argument('--verbose', '-v', action='store_true', 
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Run the converter
    converter = UrbsDataConverter(
        collections_directory=args.collections_dir,
        output_file=args.output_file,
        time_series_pattern=args.time_series_pattern
    )
    
    if converter.run():
        logger.info("Conversion completed successfully")
    else:
        logger.error("Conversion failed")

if __name__ == "__main__":
    main()
