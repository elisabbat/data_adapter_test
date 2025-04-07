import os
import sys
import pandas as pd
import glob
from typing import List, Dict, Optional, Union, Tuple
import argparse

from data_adapter import databus, collection, main
from data_adapter.preprocessing import Adapter
from data_adapter.structure import Structure
#reads from directory in PC
class UrbsDataAdapter:
    """
    A class to read files from a directory using data_adapter package
    and adapt the sheets to fit urbs package requirements.
    """
    
    # Define urbs required sheets based on documentation
    URBS_REQUIRED_SHEETS = [
        'commodity', 'process', 'storage', 'demand', 'supim', 
        'buy_sell_price', 'capacity', 'time series'
    ]
    
    def __init__(
            self,
            input_dir: str = "collections",
            output_dir: str = "results",
            structure_name: str = "steel_industry_test",
            process_sheet: str = "Process_Set",
            helper_sheet: str = "Helper_Set"
    ):
        """
        Initialize the UrbsDataAdapter.
        
        Parameters:
        -----------
        input_dir : str
            Directory containing input files
        output_dir : str
            Directory to save adapted files
        structure_name : str
            Name of the structure to use
        process_sheet : str
            Name of the process sheet
        helper_sheet : str
            Name of the helper sheet
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        
        # Initialize structure and adapter
        print(f"Initializing structure with name: {structure_name}")
        self.structure = Structure(
            structure_name,
            process_sheet=process_sheet,
            helper_sheet=helper_sheet,
        )
        
        # Initialize adapter with a placeholder collection name
        print("Initializing adapter...")
        self.adapter = Adapter(
            "urbs_adaptation",
            structure=self.structure,
        )
        
        # Standard units for urbs
        self.standard_units = {
            'None': None,
            'year': 'a',
            'cost': 'MEUR',
            'energy': 'GWh',
            'power': 'GW',
            'efficiency': '%',
            'emissions': 'Mt',
            'self_discharge': '%/h',
        }
    
    def find_files(self, extensions: List[str] = ['.csv', '.xlsx', '.xls']) -> List[str]:
        """
        Find all files with specified extensions in the input directory.
        
        Parameters:
        -----------
        extensions : List[str]
            List of file extensions to look for
            
        Returns:
        --------
        List[str]
            List of paths to found files
        """
        all_files = []
        for ext in extensions:
            pattern = os.path.join(self.input_dir, f"**/*{ext}")
            files = glob.glob(pattern, recursive=True)
            all_files.extend(files)
        
        print(f"Found {len(all_files)} files in {self.input_dir}")
        return all_files
    
    def read_file(self, file_path: str) -> Dict[str, pd.DataFrame]:
        """
        Read file (CSV or Excel) and return dictionary of DataFrames.
        
        Parameters:
        -----------
        file_path : str
            Path to the file
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary of sheet names and their DataFrames
        """
        try:
            print(f"Reading file: {file_path}")
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.csv':
                # For CSV, we'll just use the filename as the sheet name
                df = pd.read_csv(file_path)
                sheet_name = os.path.basename(os.path.splitext(file_path)[0])
                return {sheet_name: df}
            elif file_ext in ['.xlsx', '.xls']:
                # For Excel, read all sheets
                excel_file = pd.ExcelFile(file_path)
                sheet_names = excel_file.sheet_names
                sheets_dict = {}
                
                for sheet in sheet_names:
                    sheets_dict[sheet] = pd.read_excel(file_path, sheet_name=sheet)
                
                return sheets_dict
            else:
                print(f"Unsupported file extension: {file_ext}")
                return {}
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return {}
    
    def adapt_sheet_for_urbs(self, df: pd.DataFrame, sheet_name: str) -> Tuple[pd.DataFrame, str]:
        """
        Adapt a dataframe to fit urbs package requirements for a specific sheet type.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to adapt
        sheet_name : str
            Name of the sheet (used to determine adaptation method)
            
        Returns:
        --------
        Tuple[pd.DataFrame, str]
            Adapted DataFrame and determined urbs sheet type
        """
        # Lowercase the sheet name for easier matching
        sheet_name_lower = sheet_name.lower()
        urbs_sheet_type = None
        
        # Try to map the sheet to one of the urbs required sheets
        for urbs_sheet in self.URBS_REQUIRED_SHEETS:
            if urbs_sheet.lower() in sheet_name_lower:
                urbs_sheet_type = urbs_sheet
                break
        
        # If we couldn't determine the sheet type, use a default
        if urbs_sheet_type is None:
            if "time" in sheet_name_lower or "series" in sheet_name_lower:
                urbs_sheet_type = "time series"
            else:
                return df, "unknown"
        
        # Apply specific adaptations based on sheet type
        adapted_df = self._apply_specific_adaptations(df, urbs_sheet_type)
        
        return adapted_df, urbs_sheet_type
    
    def _apply_specific_adaptations(self, df: pd.DataFrame, sheet_type: str) -> pd.DataFrame:
        """
        Apply specific adaptations to a DataFrame based on the urbs sheet type.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to adapt
        sheet_type : str
            Type of urbs sheet
            
        Returns:
        --------
        pd.DataFrame
            Adapted DataFrame
        """
        # Create a copy to avoid modifying the original
        adapted_df = df.copy()
        
        # Apply specific adaptations based on sheet type
        if sheet_type == "commodity":
            # Ensure required columns
            required_cols = ['Site', 'Commodity', 'Type', 'Cost', 'CO2']
            for col in required_cols:
                if col not in adapted_df.columns:
                    adapted_df[col] = None
                    
        elif sheet_type == "process":
            # Ensure required columns
            required_cols = ['Site', 'Process', 'Commodity In/Out', 'Direction', 'Ratio', 'Cost-Var', 'CO2']
            for col in required_cols:
                if col not in adapted_df.columns:
                    adapted_df[col] = None
                    
        elif sheet_type == "storage":
            # Ensure required columns
            required_cols = ['Site', 'Storage', 'Commodity', 'Cost1', 'Cost2', 'Cost-Fix', 'Cost-Var', 'C-rate', 'Efficiency', 'Self-discharge']
            for col in required_cols:
                if col not in adapted_df.columns:
                    adapted_df[col] = None
                    
        elif sheet_type == "demand":
            # Ensure required columns
            required_cols = ['Site', 'Commodity', 'Year']
            for col in required_cols:
                if col not in adapted_df.columns:
                    adapted_df[col] = None
                    
        elif sheet_type == "supim":
            # Ensure required columns
            required_cols = ['Site', 'Commodity', 'Year']
            for col in required_cols:
                if col not in adapted_df.columns:
                    adapted_df[col] = None
                    
        elif sheet_type == "buy_sell_price":
            # Ensure required columns
            required_cols = ['Site', 'Commodity', 'Direction', 'Price']
            for col in required_cols:
                if col not in adapted_df.columns:
                    adapted_df[col] = None
                    
        elif sheet_type == "capacity":
            # Ensure required columns
            required_cols = ['Site', 'Process/Storage', 'Installed', 'Cost-Fix']
            for col in required_cols:
                if col not in adapted_df.columns:
                    adapted_df[col] = None
                    
        elif sheet_type == "time series":
            # For time series, we might need to pivot or restructure based on specific format
            # This would depend on the exact requirements
            pass
            
        # Add any other sheet-specific adaptations here
            
        return adapted_df
    
    def save_adapted_files(self, adapted_sheets: Dict[str, pd.DataFrame], original_filename: str) -> str:
        """
        Save adapted sheets to an Excel file.
        
        Parameters:
        -----------
        adapted_sheets : Dict[str, pd.DataFrame]
            Dictionary of adapted sheets
        original_filename : str
            Name of the original file (used to construct output filename)
            
        Returns:
        --------
        str
            Path to the saved Excel file
        """
        # Create output filename
        base_name = os.path.splitext(os.path.basename(original_filename))[0]
        output_file = os.path.join(self.output_dir, f"{base_name}_urbs_adapted.xlsx")
        
        print(f"Saving adapted sheets to {output_file}")
        
        # Create a new Excel writer
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            for sheet_name, df in adapted_sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"Successfully saved adapted file: {output_file}")
        return output_file
    
    def process_all_files(self) -> List[str]:
        """
        Process all files in the input directory.
        
        Returns:
        --------
        List[str]
            List of paths to adapted files
        """
        # Find all files
        input_files = self.find_files()
        
        # Process each file
        adapted_files = []
        
        for file_path in input_files:
            try:
                # Read file
                sheets_dict = self.read_file(file_path)
                
                if not sheets_dict:
                    print(f"Skipping empty or unreadable file: {file_path}")
                    continue
                
                # Adapt each sheet
                adapted_sheets = {}
                
                for sheet_name, df in sheets_dict.items():
                    adapted_df, urbs_sheet_type = self.adapt_sheet_for_urbs(df, sheet_name)
                    
                    # Only include known sheet types
                    if urbs_sheet_type != "unknown":
                        adapted_sheets[urbs_sheet_type] = adapted_df
                        print(f"Adapted sheet '{sheet_name}' as '{urbs_sheet_type}'")
                    else:
                        print(f"Skipping sheet '{sheet_name}': could not determine urbs sheet type")
                
                # Save adapted sheets
                if adapted_sheets:
                    adapted_file = self.save_adapted_files(adapted_sheets, file_path)
                    adapted_files.append(adapted_file)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
        
        # Print summary
        print("\nProcessing Summary:")
        print(f"Total files processed: {len(input_files)}")
        print(f"Successfully adapted files: {len(adapted_files)}")
        
        return adapted_files


def main():
    """Command line interface for the UrbsDataAdapter."""
    parser = argparse.ArgumentParser(description='Adapt data files to fit urbs package requirements')
    parser.add_argument('--input-dir', default='collections', help='Directory containing input files')
    parser.add_argument('--output-dir', default='results', help='Directory to save adapted files')
    parser.add_argument('--structure', default='steel_industry_test')
    parser.add_argument('--process-sheet', default='Process_Set', help='Process sheet name (default: Process_Set)')
    parser.add_argument('--helper-sheet', default='Helper_Set', help='Helper sheet name (default: Helper_Set)')
    
    args = parser.parse_args()
    
    # Create adapter
    adapter = UrbsDataAdapter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        structure_name=args.structure,
        process_sheet=args.process_sheet,
        helper_sheet=args.helper_sheet
    )
    
    # Process all files
    adapter.process_all_files()
    
    print(f"\nAll processing completed! Adapted files are in the '{args.output_dir}' directory.")


if __name__ == "__main__":
    main()
