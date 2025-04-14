import os
import sys
import shutil
import pandas as pd
import argparse
from pathlib import Path
import re
import logging
import dataclasses
from data_adapter import databus, collection, main
from data_adapter.preprocessing import Adapter
from data_adapter.structure import Structure

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclasses.dataclass
class Datapackage:
    """Class to store structured data that will be converted to urbs format."""
    parametrized_elements: dict = dataclasses.field(default_factory=dict)  # scalar data in form of {type:pd.DataFrame(type)}
    parametrized_sequences: dict = dataclasses.field(default_factory=dict)  # timeseries in form of {type:pd.DataFrame(type)}
    foreign_keys: dict = dataclasses.field(default_factory=dict)  # foreign keys for timeseries profiles
    adapter: Adapter = None
    periods: pd.DataFrame = None
    location_to_save_to: str = None
    tsa_parameters: pd.DataFrame = None

class UrbsDataConverter:
    """
    Class to read CSV files from 'collections' directory using data_adapter
    and convert them to urbs Excel format.
    """
    # Define urbs required sheet names and their structures
    URBS_SHEETS = {
        'global': ['param', 'value', 'unit', 'description'],
        'site': ['site', 'lat', 'lon', 'area'],
        'commodity': ['site', 'commodity', 'type', 'price', 'max', 'maxperhour', 'co2'],
        'process': ['site', 'process', 'cap-min', 'cap-max', 'max-grad', 'min-load', 
                   'inv-cost', 'fix-cost', 'var-cost', 'wacc', 'depreciation', 'area-per-cap'],
        'storage': ['site', 'storage', 'commodity', 'cap-min', 'cap-max', 'eff-in', 'eff-out', 
                   'self-discharge', 'inv-cost-p', 'inv-cost-e', 'fix-cost-p', 'fix-cost-e', 
                   'var-cost-p', 'var-cost-e', 'wacc', 'depreciation', 'initial'],
        'transmission': ['site-in', 'site-out', 'transmission', 'commodity', 'cap-min', 'cap-max', 
                        'eff', 'inv-cost', 'fix-cost', 'var-cost', 'wacc', 'depreciation', 'distance']
    }
    
    def __init__(
        self, 
        collections_directory="collections/SEDOS_industry_sector",
        output_file="results",
        structure_name="SEDOS_Modellstruktur_steel_industry",
        process_sheet="Process_Set",
        helper_sheet="Helper_Set",
        scenario="default",

    ):
        """
        Initialize the converter with collections directory path and output file.
        
        Args:
            collections_directory: Path to the 'collections' directory containing subdirectories with CSV files
            output_file: Path to save the output Excel file
            structure_name: Name of the structure file
            process_sheet: Name of the process sheet
            helper_sheet: Name of the helper sheet
            scenario: Scenario name
        """
        self.collections_directory = Path(collections_directory)
        self.output_file = Path(output_file)
        self.structure_name = structure_name
        self.process_sheet = process_sheet
        self.helper_sheet = helper_sheet
        self.scenario = scenario
        
        # Data containers
        self.datapackage = Datapackage()
        self.urbs_dataframes = {}
        
        # Initialize structure and adapter
        self._init_data_adapter()
    
    def _init_data_adapter(self):
        """Initialize the data_adapter components."""
        try:

            self.collection_name = "SEDOS_industry_sector"
            
            # Initialize structure
            self.structure = Structure(
                self.structure_name,
                process_sheet=self.process_sheet,
                helper_sheet=self.helper_sheet,
            )
            
            # Initialize adapter
            self.adapter = Adapter(
                self.collection_name,
                structure=self.structure,
            )
            
            self.datapackage.adapter = self.adapter
            logger.info(f"Initialized data_adapter with collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error initializing data_adapter: {e}")
            raise

    def find_csv_files(self):
        """
        Find all CSV files in the collections directory.
        """
        logger.info(f"Searching for data files in {self.collections_directory}")

        # Check if collections directory exists
        if not self.collections_directory.exists():
            logger.error(f"Collections directory '{self.collections_directory}' does not exist.")
            return False

        try:
            # Find files with .csv extension in the collections directory
            collection_files = list(self.collections_directory.rglob("*.csv"))

            if not collection_files:
                logger.warning(f"No files found in collection: {self.collection_name}")
                return False

            logger.info(f"Found {len(collection_files)} files in collection")
            return collection_files  # Return the list of files
        except Exception as e:
            logger.error(f"An error occurred while searching for files: {e}")
            return False

    def read_collection_data(self):
        """
        Read data from the collection using data_adapter.
        """
        self.processes = {
            process: self.adapter.get_process(process)
            for process in list(self.structure.processes.keys())
        }
        try:
            # Use the adapter to read collection data
            scalar_data = self.adapter.get_process(self,process=scalar)
            if scalar_data is not None and not scalar_data.empty:
                logger.info(f"Read scalar data with {len(scalar_data)} entries")
                self.datapackage.parametrized_elements = self._group_scalar_data(scalar_data)
            else:
                logger.warning("No scalar data found")
            
            # Read time series data
            timeseries_data = self.adapter.get_process(self, process=timeseries)
            if timeseries_data:
                logger.info(f"Read {len(timeseries_data)} time series")
                self.datapackage.parametrized_sequences = timeseries_data
            else:
                logger.warning("No time series data found")
                
            return len(self.datapackage.parametrized_elements) > 0 or len(self.datapackage.parametrized_sequences) > 0
            
        except Exception as e:
            logger.error(f"Error reading collection data: {e}")
            return False
    
    def _group_scalar_data(self, scalar_data):
        """
        Group scalar data by type for better organization.
        
        Args:
            scalar_data: Pandas DataFrame with scalar data
            
        Returns:
            Dictionary of DataFrames grouped by type
        """
        grouped_data = {}
        
        if 'type' in scalar_data.columns:
            # Group by type if available
            for data_type, group_df in scalar_data.groupby('type'):
                grouped_data[data_type] = group_df.reset_index(drop=True)
        else:
            # Otherwise, try to infer types from other columns
            grouped_data['default'] = scalar_data
            
        return grouped_data
    
    def transform_to_urbs_format(self):
        """
        Transform the data from data_adapter format to urbs format.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # 1. Create site data
            self._create_site_sheet()
            
            # 2. Create commodity data
            self._create_commodity_sheet()
            
            # 3. Create process data
            self._create_process_sheet()
            
            # 4. Create storage data if applicable
            self._create_storage_sheet()
            
            # 5. Create transmission data if applicable
            self._create_transmission_sheet()
            
            # 6. Create global data
            self._create_global_sheet()
            
            # 7. Transform time series data
            self._transform_time_series()
            
            logger.info("Completed transformation to urbs format")
            return len(self.urbs_dataframes) > 0
            
        except Exception as e:
            logger.error(f"Error transforming to urbs format: {e}")
            return False
    
    def _create_site_sheet(self):
        """Create the 'site' sheet for urbs."""
        site_data = {
            'site': [],
            'lat': [],
            'lon': [],
            'area': []
        }
        
        # Extract site information from parametrized elements
        for data_type, df in self.datapackage.parametrized_elements.items():
            if 'site' in df.columns:
                for site in df['site'].unique():
                    if site and site not in site_data['site']:
                        site_data['site'].append(site)
                        site_data['lat'].append(None)  # Could be populated if latitude data is available
                        site_data['lon'].append(None)  # Could be populated if longitude data is available
                        site_data['area'].append(None)  # Could be populated if area data is available
        
        # If no sites found, create a default site
        if not site_data['site']:
            site_data['site'].append('default')
            site_data['lat'].append(0)
            site_data['lon'].append(0)
            site_data['area'].append(None)
        
        self.urbs_dataframes['site'] = pd.DataFrame(site_data)
        logger.info(f"Created site sheet with {len(site_data['site'])} sites")
    
    def _create_commodity_sheet(self):
        """Create the 'commodity' sheet for urbs."""
        commodity_data = {
            'site': [],
            'commodity': [],
            'type': [],
            'price': [],
            'max': [],
            'maxperhour': [],
            'co2': []
        }
        
        # Extract commodity information from parametrized elements
        commodities_found = set()
        
        for data_type, df in self.datapackage.parametrized_elements.items():
            if 'commodity' in df.columns:
                for _, row in df.iterrows():
                    site = row.get('site', 'default')
                    commodity = row.get('commodity')
                    
                    if commodity and (site, commodity) not in commodities_found:
                        commodities_found.add((site, commodity))
                        
                        commodity_data['site'].append(site)
                        commodity_data['commodity'].append(commodity)
                        
                        # Try to determine commodity type based on available information
                        if 'type' in df.columns and not pd.isna(row.get('type')):
                            # Map to urbs commodity types: Stock, SupIm, Demand, Env, Buy, Sell
                            type_map = {
                                'input': 'Stock',
                                'output': 'Demand',
                                'emission': 'Env',
                                'renewable': 'SupIm'
                            }
                            commodity_data['type'].append(type_map.get(row.get('type').lower(), 'Stock'))
                        else:
                            commodity_data['type'].append('Stock')
                        
                        # Add price and other attributes
                        price = row.get('price', 0) if 'price' in df.columns else 0
                        commodity_data['price'].append(price)
                        
                        max_val = row.get('max', None) if 'max' in df.columns else None
                        commodity_data['max'].append(max_val)
                        
                        maxperhour = row.get('maxperhour', None) if 'maxperhour' in df.columns else None
                        commodity_data['maxperhour'].append(maxperhour)
                        
                        co2 = row.get('co2', 0) if 'co2' in df.columns else 0
                        commodity_data['co2'].append(co2)
        
        if commodity_data['site']:
            self.urbs_dataframes['commodity'] = pd.DataFrame(commodity_data)
            logger.info(f"Created commodity sheet with {len(commodity_data['site'])} entries")
        else:
            logger.warning("No commodity data found")
    
    def _create_process_sheet(self):
        """Create the 'process' sheet for urbs."""
        process_data = {
            'site': [],
            'process': [],
            'cap-min': [],
            'cap-max': [],
            'max-grad': [],
            'min-load': [],
            'inv-cost': [],
            'fix-cost': [],
            'var-cost': [],
            'wacc': [],
            'depreciation': [],
            'area-per-cap': []
        }
        
        # Add commodity input/output columns for each commodity
        commodities = set()
        if 'commodity' in self.urbs_dataframes:
            for _, row in self.urbs_dataframes['commodity'].iterrows():
                commodities.add(row['commodity'])
        
        for commodity in commodities:
            process_data[f'{commodity}.in'] = []
            process_data[f'{commodity}.out'] = []
        
        # Extract process information from parametrized elements
        processes_found = set()
        
        for data_type, df in self.datapackage.parametrized_elements.items():
            if 'process' in df.columns:
                for _, row in df.iterrows():
                    site = row.get('site', 'default')
                    process = row.get('process')
                    
                    if process and (site, process) not in processes_found:
                        processes_found.add((site, process))
                        
                        process_data['site'].append(site)
                        process_data['process'].append(process)
                        
                        # Add process parameters
                        process_data['cap-min'].append(row.get('cap-min', 0) if 'cap-min' in df.columns else 0)
                        process_data['cap-max'].append(row.get('cap-max', None) if 'cap-max' in df.columns else None)
                        process_data['max-grad'].append(row.get('max-grad', None) if 'max-grad' in df.columns else None)
                        process_data['min-load'].append(row.get('min-load', None) if 'min-load' in df.columns else None)
                        process_data['inv-cost'].append(row.get('inv-cost', 0) if 'inv-cost' in df.columns else 0)
                        process_data['fix-cost'].append(row.get('fix-cost', 0) if 'fix-cost' in df.columns else 0)
                        process_data['var-cost'].append(row.get('var-cost', 0) if 'var-cost' in df.columns else 0)
                        process_data['wacc'].append(row.get('wacc', 0.07) if 'wacc' in df.columns else 0.07)
                        process_data['depreciation'].append(row.get('depreciation', 20) if 'depreciation' in df.columns else 20)
                        process_data['area-per-cap'].append(row.get('area-per-cap', 0) if 'area-per-cap' in df.columns else 0)
                        
                        # Initialize commodity input/output ratios to zero
                        for commodity in commodities:
                            process_data[f'{commodity}.in'].append(0)
                            process_data[f'{commodity}.out'].append(0)
        
        if process_data['site']:
            self.urbs_dataframes['process'] = pd.DataFrame(process_data)
            logger.info(f"Created process sheet with {len(process_data['site'])} processes")
        else:
            logger.warning("No process data found")
    
    def _create_storage_sheet(self):
        """Create the 'storage' sheet for urbs if storage data is available."""
        storage_data = {
            'site': [],
            'storage': [],
            'commodity': [],
            'cap-min': [],
            'cap-max': [],
            'eff-in': [],
            'eff-out': [],
            'self-discharge': [],
            'inv-cost-p': [],
            'inv-cost-e': [],
            'fix-cost-p': [],
            'fix-cost-e': [],
            'var-cost-p': [],
            'var-cost-e': [],
            'wacc': [],
            'depreciation': [],
            'initial': []
        }
        
        # Check if we have storage data
        has_storage_data = False
        
        for data_type, df in self.datapackage.parametrized_elements.items():
            if 'storage' in df.columns or data_type.lower() == 'storage':
                has_storage_data = True
                
                for _, row in df.iterrows():
                    site = row.get('site', 'default')
                    storage = row.get('storage', row.get('name', f"storage_{len(storage_data['storage']) + 1}"))
                    commodity = row.get('commodity', 'default')
                    
                    storage_data['site'].append(site)
                    storage_data['storage'].append(storage)
                    storage_data['commodity'].append(commodity)
                    
                    # Add storage parameters
                    storage_data['cap-min'].append(row.get('cap-min', 0) if 'cap-min' in df.columns else 0)
                    storage_data['cap-max'].append(row.get('cap-max', None) if 'cap-max' in df.columns else None)
                    storage_data['eff-in'].append(row.get('eff-in', 1.0) if 'eff-in' in df.columns else 1.0)
                    storage_data['eff-out'].append(row.get('eff-out', 1.0) if 'eff-out' in df.columns else 1.0)
                    storage_data['self-discharge'].append(row.get('self-discharge', 0) if 'self-discharge' in df.columns else 0)
                    storage_data['inv-cost-p'].append(row.get('inv-cost-p', 0) if 'inv-cost-p' in df.columns else 0)
                    storage_data['inv-cost-e'].append(row.get('inv-cost-e', 0) if 'inv-cost-e' in df.columns else 0)
                    storage_data['fix-cost-p'].append(row.get('fix-cost-p', 0) if 'fix-cost-p' in df.columns else 0)
                    storage_data['fix-cost-e'].append(row.get('fix-cost-e', 0) if 'fix-cost-e' in df.columns else 0)
                    storage_data['var-cost-p'].append(row.get('var-cost-p', 0) if 'var-cost-p' in df.columns else 0)
                    storage_data['var-cost-e'].append(row.get('var-cost-e', 0) if 'var-cost-e' in df.columns else 0)
                    storage_data['wacc'].append(row.get('wacc', 0.07) if 'wacc' in df.columns else 0.07)
                    storage_data['depreciation'].append(row.get('depreciation', 20) if 'depreciation' in df.columns else 20)
                    storage_data['initial'].append(row.get('initial', 0) if 'initial' in df.columns else 0)
        
        if has_storage_data:
            self.urbs_dataframes['storage'] = pd.DataFrame(storage_data)
            logger.info(f"Created storage sheet with {len(storage_data['storage'])} storages")
    
    def _create_transmission_sheet(self):
        """Create the 'transmission' sheet for urbs if transmission data is available."""
        transmission_data = {
            'site-in': [],
            'site-out': [],
            'transmission': [],
            'commodity': [],
            'cap-min': [],
            'cap-max': [],
            'eff': [],
            'inv-cost': [],
            'fix-cost': [],
            'var-cost': [],
            'wacc': [],
            'depreciation': [],
            'distance': []
        }
        
        # Check if we have transmission data
        has_transmission_data = False
        
        for data_type, df in self.datapackage.parametrized_elements.items():
            if ('transmission' in df.columns or 
                data_type.lower() == 'transmission' or 
                ('site-in' in df.columns and 'site-out' in df.columns)):
                has_transmission_data = True
                
                for _, row in df.iterrows():
                    site_in = row.get('site-in', row.get('site_in', 'default'))
                    site_out = row.get('site-out', row.get('site_out', 'default'))
                    transmission = row.get('transmission', f"{site_in}_{site_out}")
                    commodity = row.get('commodity', 'default')
                    
                    transmission_data['site-in'].append(site_in)
                    transmission_data['site-out'].append(site_out)
                    transmission_data['transmission'].append(transmission)
                    transmission_data['commodity'].append(commodity)
                    
                    # Add transmission parameters
                    transmission_data['cap-min'].append(row.get('cap-min', 0) if 'cap-min' in df.columns else 0)
                    transmission_data['cap-max'].append(row.get('cap-max', None) if 'cap-max' in df.columns else None)
                    transmission_data['eff'].append(row.get('eff', 1.0) if 'eff' in df.columns else 1.0)
                    transmission_data['inv-cost'].append(row.get('inv-cost', 0) if 'inv-cost' in df.columns else 0)
                    transmission_data['fix-cost'].append(row.get('fix-cost', 0) if 'fix-cost' in df.columns else 0)
                    transmission_data['var-cost'].append(row.get('var-cost', 0) if 'var-cost' in df.columns else 0)
                    transmission_data['wacc'].append(row.get('wacc', 0.07) if 'wacc' in df.columns else 0.07)
                    transmission_data['depreciation'].append(row.get('depreciation', 20) if 'depreciation' in df.columns else 20)
                    transmission_data['distance'].append(row.get('distance', 1) if 'distance' in df.columns else 1)
        
        if has_transmission_data:
            self.urbs_dataframes['transmission'] = pd.DataFrame(transmission_data)
            logger.info(f"Created transmission sheet with {len(transmission_data['transmission'])} transmissions")
    
    def _create_global_sheet(self):
        """Create the 'global' sheet for urbs with global parameters."""
        global_data = {
            'param': ['timestep', 'horizon', 'interest rate', 'weight'],
            'value': [1, 8760, 0.07, 1],
            'unit': ['h', 'h', '-', '-'],
            'description': [
                'Duration of one time step',
                'Time horizon of optimization',
                'Interest rate',
                'Scaling factor for all costs'
            ]
        }
        
        # Add any global parameters found in the data
        for data_type, df in self.datapackage.parametrized_elements.items():
            if data_type.lower() in ['global', 'settings', 'parameters']:
                for _, row in df.iterrows():
                    param = row.get('param', row.get('parameter', None))
                    if param and param not in global_data['param']:
                        global_data['param'].append(param)
                        global_data['value'].append(row.get('value', 0))
                        global_data['unit'].append(row.get('unit', '-'))
                        global_data['description'].append(row.get('description', ''))
        
        self.urbs_dataframes['global'] = pd.DataFrame(global_data)
        logger.info(f"Created global sheet with {len(global_data['param'])} parameters")
    
    def _transform_time_series(self):
        """Transform time series data to urbs format."""
        # Process demand time series
        demand_ts = pd.DataFrame()
        
        # Process supim (supply and intermittent) time series
        supim_ts = pd.DataFrame()
        
        # Check if we have time series data
        for ts_name, ts_df in self.datapackage.parametrized_sequences.items():
            # Try to determine if this is a demand or supply time series
            if 'demand' in ts_name.lower():
                if demand_ts.empty:
                    demand_ts = ts_df.copy()
                else:
                    # Try to merge time series
                    common_index = demand_ts.index.intersection(ts_df.index)
                    if not common_index.empty:
                        for col in ts_df.columns:
                            if col not in demand_ts.columns:
                                demand_ts[col] = ts_df[col]
                    else:
                        # If indices don't match, append as new columns with NaN values
                        demand_ts = pd.concat([demand_ts, ts_df], axis=1)
            elif any(term in ts_name.lower() for term in ['supim', 'solar', 'wind', 'hydro', 'supply']):
                if supim_ts.empty:
                    supim_ts = ts_df.copy()
                else:
                    # Try to merge time series
                    common_index = supim_ts.index.intersection(ts_df.index)
                    if not common_index.empty:
                        for col in ts_df.columns:
                            if col not in supim_ts.columns:
                                supim_ts[col] = ts_df[col]
                    else:
                        # If indices don't match, append as new columns with NaN values
                        supim_ts = pd.concat([supim_ts, ts_df], axis=1)
        
        # Add time series to urbs dataframes
        if not demand_ts.empty:
            self.urbs_dataframes['demand'] = demand_ts
            logger.info(f"Created demand time series with {demand_ts.shape[0]} timestamps and {demand_ts.shape[1]} columns")
        
        if not supim_ts.empty:
            self.urbs_dataframes['supim'] = supim_ts
            logger.info(f"Created supim time series with {supim_ts.shape[0]} timestamps and {supim_ts.shape[1]} columns")
    
    def export_to_excel(self):
        """
        Export all urbs dataframes to an Excel file.
        
        Returns:
            True if export was successful, False otherwise
        """
        if not self.urbs_dataframes:
            logger.warning("No urbs dataframes to export")
            return False
        
        try:
            logger.info(f"Exporting to Excel file: {self.output_file}")
            
            # Create directory if it doesn't exist
            os.makedirs(self.output_file.parent, exist_ok=True)
            
            # Write dataframes to Excel file
            with pd.ExcelWriter(self.output_file, engine='openpyxl') as writer:
                for sheet_name, df in self.urbs_dataframes.items():
                    # For time series, write with index
                    if sheet_name in ['demand', 'supim']:
                        df.to_excel(writer, sheet_name=sheet_name)
                    else:
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    logger.info(f"Exported {sheet_name} sheet")
            
            logger.info(f"Successfully exported urbs data to {self.output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to Excel: {e}")
            return False
    
    def run(self):
        """
        Run the complete conversion process.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Find files
            if self.find_csv_files():
                # Read data using data_adapter
                if self.read_collection_data():
                    # Transform to urbs format
                    if self.transform_to_urbs_format():
                        # Export to Excel
                        return self.export_to_excel()
            
            return False
            
        except Exception as e:
            logger.error(f"Error during conversion process: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(
        description='Convert CSV files from collections directory to urbs Excel format using data_adapter'
    )
    parser.add_argument('--collections_directory', default='collections', help='Directory containing input files')
    parser.add_argument('--output_file', default='results', help='Directory to save adapted files')
    parser.add_argument('--structure_name', default='SEDOS_Modellstruktur_steel_industry')
    parser.add_argument('--process_sheet', default='Process_Set', help='Process sheet name (default: Process_Set)')
    parser.add_argument('--helper_sheet', default='Helper_Set', help='Helper sheet name (default: Helper_Set)')
    parser.add_argument('--scenario', default='default', help='Helper sheet name (default: Helper_Set)')

    args = parser.parse_args()

    # Create adapter
    adapter = UrbsDataConverter(
        collections_directory=args.collections_directory,
        output_file=args.output_file,
        structure_name=args.structure_name,
        process_sheet=args.process_sheet,
        helper_sheet=args.helper_sheet,
        scenario=args.scenario,
    )

    # Process all files
    adapter.run()

    print(f"\nAll processing completed! Adapted files are in the '{args.output_file}' directory.")


if __name__ == "__main__":
    main()
