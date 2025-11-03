import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import logging
import traceback
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XlsxConversionError(Exception):
    """Custom exception for XLSX to CSV conversion errors"""
    pass

class BmkgXlsxConverter:
    """Converts single BMKG XLSX file to CSV format"""
    
    def __init__(self):
        self.logger = logger
    
    def process_bmkg_excel(
        self,
        file_buffer: bytes,
        filename: str = None,
    ) -> Dict[str, Any]:
        """
        Converts BMKG Excel to CSV format.
        
        Args: 
            file_buffer (bytes): Excel file content as bytes.
            filename (str, optional): Name of the file for logging purposes.
            
        Returns:
            Dictionary containing csv data and metadata
        """ 

        try:
            self.logger.info(f"Processing XLSX file: {filename or 'uploaded file'}")
            
            # Read Excel file
            df = pd.read_excel(io.BytesIO(file_buffer), header=None)
            
            # Find header row containing TANGGAL
            header_row_idx = None
            for i in range(10):
                row = df.iloc[i].astype(str)
                if "TANGGAL" in row.values:
                    header_row_idx = i
                    break
            
            if header_row_idx is None:
                raise XlsxConversionError(f"Header TANGGAL tidak ditemukan di file {filename or 'uploaded file'}")
            
            # Extract headers and data
            headers = df.iloc[header_row_idx].tolist()
            df = df.iloc[header_row_idx+1:].reset_index(drop=True)
            df.columns = headers
            
            # Clean missing values only for empty NaN or '-'
            missing_mask = df.isin(['-', 'nan', 'NaN', 'NULL', ''])
            df = df.mask(missing_mask, np.nan)
            
            # Find last valid data row before KETERANGAN
            last_row_idx = len(df)
            for i in range(len(df)):
                row_data = df.iloc[i].dropna()
                if len(row_data) == 0 or (isinstance(df.iloc[i, 0], str) and 
                                        ("KETERANGAN" in str(df.iloc[i, 0]) or str(df.iloc[i, 0]).strip() == "")):
                    last_row_idx = i
                    break
            df = df.iloc[:last_row_idx]
            
            # Validate TANGGAL format
            if "TANGGAL" not in df.columns:
                raise XlsxConversionError(f"kolom TANGGAL tidak ditemukan.")
            
            # Convert date dolumn
            df['Date'] = pd.to_datetime(df['TANGGAL'], format='%d-%m-%Y', errors='coerce')
            if df['Date'].isna().all():
                df['Date'] = pd.to_datetime(df['TANGGAL'], errors='coerce')
                
            # Extract date components
            df['Year'] = df['Date'].dt.year.astype('Int64')
            df['month'] = df['Date'].dt.month.astype('Int64')
            df['day'] = df['Date'].dt.day.astype('Int64')
            
            # Remove old date columns
            df = df.drop(columns=['Month', 'Day'], errors='ignore')
            
            # Reorganize columns (Date first, then Year, month, day)
            date_cols = ['Date', 'Year', 'month', 'day']
            other_cols = [col for col in df.columns if col not in date_cols + ['TANGGAL']]
            df = df[date_cols + other_cols]
            
            # ✅ KEEP ORIGINAL BMKG DATA - NO FILL VALUE CONVERSION
            # Don't convert 8888, 9999 or any other BMKG codes
            # Keep all original values as they are
            
            # Convert numeric columns appropriately (but preserve original values)
            for col in df.columns:
                if col not in ['Date', 'TANGGAL']:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        non_na_values = df[col].dropna()
                        if len(non_na_values) > 0:
                            # Only convert to integer if ALL values are actually integers
                            # This preserves decimal values and BMKG codes
                            try:
                                if all(non_na_values == non_na_values.astype(int)):
                                    df[col] = df[col].astype('Int64')
                            except:
                                # Keep as float if conversion fails
                                pass
            
            # Remove completely empty rows
            df = df.dropna(how='all')
            
            # Generate CSV string
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()
            
            # Extract date range for metadata
            min_date = df['Date'].min()
            max_date = df['Date'].max()
            
            # Critial fix: convert DataFrame to JSON-safe records
            records = []
            for _, row in df.iterrows():
                record = {}
                for col in df.columns:
                    value = row[col]
                    if pd.isna(value):
                        record[col] = None  # Convert NaN to None (becomes null in JSON)
                    elif isinstance(value, (np.integer, np.int64, np.int32)):
                        record[col] = int(value)
                    elif isinstance(value, (np.floating, np.float64, np.float32)):
                        if np.isnan(value):
                            record[col] = None
                        else:
                            record[col] = float(value)
                    elif isinstance(value, pd.Timestamp):
                        record[col] = value.isoformat() if pd.notna(value) else None
                    elif hasattr(value, 'item'):  # Handle numpy scalars
                        try:
                            record[col] = value.item()
                        except:
                            record[col] = str(value)
                    else:
                        record[col] = value
                records.append(record)
            
            result = {
                "status": "success",
                "csv_content": csv_content,
                "records": records,
                "record_count": len(records),
                "columns": list(df.columns),
                "date_range": {
                    "start": min_date.strftime('%Y-%m-%d') if pd.notna(min_date) else None,
                    "end": max_date.strftime('%Y-%m-%d') if pd.notna(max_date) else None
                },
                "file_info": {
                    "original_filename": filename,
                    "rows_processed": len(df),
                    "columns_found": len(df.columns)
                }
            }
            # ✅ SAFETY CHECK: Test JSON serialization before returning
            try:
                json.dumps(result)
                self.logger.info(f"✓ Conversion successful: {len(records)} records, {len(df.columns)} columns")
                return result
            except (TypeError, ValueError) as json_error:
                self.logger.error(f"JSON serialization failed: {json_error}")
                raise XlsxConversionError(f"Data contains non-serializable values: {json_error}")
            
        except Exception as e:
            error_msg = f"Error converting XLSX: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            raise XlsxConversionError(error_msg)

def convert_single_xlsx(file_buffer: bytes, filename: str = None) -> Dict[str, Any]:
    """
    Main function to convert single XLSX file to CSV
    
    Args:
        file_buffer: Excel file content as bytes
        filename: Original filename
        
    Returns:
        Conversion result dictionary
    """
    converter = BmkgXlsxConverter()
    return converter.process_bmkg_excel(file_buffer, filename)