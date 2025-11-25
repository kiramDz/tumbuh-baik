import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
import traceback
import json
import io
from .xlsx_to_csv import BmkgXlsxConverter, XlsxConversionError


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BmkgMultiXlsxMerger:
    """Merges multiple BMKG XLSX files into single CSV dataset"""
    
    def __init__(self):
        self.converter = BmkgXlsxConverter()
        self.logger = logger
        
    def process_multiple_xlsx(self, files_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process multiple XLSX files and merge into single CSV
        
        Args:
            files_data: List of dicts with 'buffer' and 'filename' keys
            
        Returns:
            Dictionary containing merged CSV data and metadata
        """
        try:
            self.logger.info(f"Processing {len(files_data)} XLSX files for merging")
            
            if not files_data:
                raise XlsxConversionError("No files provided for processing")
            
            if len(files_data) > 50:  # Safety limit
                raise XlsxConversionError("Terlalu banyak file (maksimal 50 file per batch)")
            
            processed_dfs = []
            file_info = []
            total_records = 0
            failed_files = []
            
            # Process each file
            for idx, file_data in enumerate(files_data):
                try:
                    file_buffer = file_data['buffer']
                    filename = file_data['filename']
                    
                    self.logger.info(f"Processing file {idx+1}/{len(files_data)}: {filename}")
                    
                    # Convert single file
                    result = self.converter.process_bmkg_excel(file_buffer, filename)
                    
                    if result['status'] == 'success':
                        # Convert records back to DataFrame
                        df = pd.DataFrame(result['records'])
                        processed_dfs.append(df)
                        total_records += result['record_count']
                        
                        file_info.append({
                            "filename": filename,
                            "records": result['record_count'],
                            "date_range": result['date_range'],
                            "status": "success"
                        })
                        
                        self.logger.info(f"✓ {filename}: {result['record_count']} records")
                    else:
                        failed_files.append(filename)
                        
                except Exception as e:
                    error_msg = f"Error processing {filename}: {str(e)}"
                    self.logger.warning(error_msg)
                    failed_files.append(filename)
                    file_info.append({
                        "filename": filename,
                        "records": 0,
                        "status": "failed",
                        "error": str(e)
                    })
            
            if not processed_dfs:
                raise XlsxConversionError("Tidak ada file yang berhasil diproses")
            
            # Merge all DataFrames
            self.logger.info(f"Merging {len(processed_dfs)} processed files...")
            combined_df = pd.concat(processed_dfs, ignore_index=True)
            
            # Sort by date
            combined_df = combined_df.sort_values(by='Date').reset_index(drop=True)
            
            # Remove duplicate dates (if any)
            before_dedup = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=['Date'], keep='first')
            after_dedup = len(combined_df)
            
            if before_dedup != after_dedup:
                self.logger.info(f"Removed {before_dedup - after_dedup} duplicate dates")
            
            # Generate merged CSV
            csv_buffer = io.StringIO()
            combined_df.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()
            
            # Calculate date range
            min_date = combined_df['Date'].min()
            max_date = combined_df['Date'].max()
            
            # Calculate file size estimate (CSV content size)
            csv_size = len(csv_content.encode('utf-8'))
            
            # Generate auto dataset name
            dataset_name = self._generate_dataset_name(min_date, max_date, len(files_data))
            
            records = []
            for _, row in combined_df.iterrows():
                record = {}
                for col in combined_df.columns:
                    value = row[col]
                    if pd.isna(value):
                        record[col] = None
                    elif isinstance(value, (np.integer, np.int64, np.int32)):
                        record[col] = int(value)
                    elif isinstance(value, (np.floating, np.float64, np.float32)):
                        if np.isnan(value):
                            record[col] = None
                        else:
                            record[col] = float(value)
                    elif isinstance(value, pd.Timestamp):
                        record[col] = value.isoformat() if pd.notna(value) else None
                    elif hasattr(value, 'item'):
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
                "columns": list(combined_df.columns),
                "dataset_metadata": {
                    "name": dataset_name,
                    "source": "Data BMKG (https://dataonline.bmkg.go.id/)",
                    "filename": f"{dataset_name.lower().replace(' ', '_')}.csv",
                    "fileSize": csv_size,
                    "totalRecords": len(combined_df),
                    "fileType": "csv",
                    "status": "raw",
                    "columns": [col for col in combined_df.columns if col not in ['_id', '__v']],
                    "isAPI": False,
                    "uploadDate": datetime.now().isoformat()
                },
                "processing_summary": {
                    "files_processed": len(processed_dfs),
                    "files_failed": len(failed_files),
                    "total_files": len(files_data),
                    "total_records": len(combined_df),
                    "duplicates_removed": before_dedup - after_dedup,
                    "date_range": {
                        "start": min_date.strftime('%Y-%m-%d') if pd.notna(min_date) and hasattr(min_date, 'strftime') else (
                            str(min_date)[:10] if pd.notna(min_date) and str(min_date) != 'NaT' else None
                        ),
                        "end": max_date.strftime('%Y-%m-%d') if pd.notna(max_date) and hasattr(max_date, 'strftime') else (
                            str(max_date)[:10] if pd.notna(max_date) and str(max_date) != 'NaT' else None
                        ),
                        "years": int(max_date.year - min_date.year + 1) if (
                            pd.notna(min_date) and pd.notna(max_date) and 
                            hasattr(min_date, 'year') and hasattr(max_date, 'year')
                        ) else 0
                    }
                },
                "file_details": file_info,
                "failed_files": failed_files,
                "warnings": []
            }
            
            # Add warnings if any files failed
            if failed_files:
                result["warnings"].append(f"{len(failed_files)} file(s) gagal diproses: {', '.join(failed_files)}")

             # Test JSON serialization
            try:
                json.dumps(result)
                self.logger.info(f"✓ Merge completed: {len(records)} total records from {len(processed_dfs)} files")
                return result
            except (TypeError, ValueError) as json_error:
                self.logger.error(f"JSON serialization failed: {json_error}")
                raise XlsxConversionError(f"Merged data contains non-serializable values: {json_error}")
            
        except Exception as e:
            error_msg = f"Error merging XLSX files: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            raise XlsxConversionError(error_msg)
    
    def _generate_dataset_name(self, min_date: pd.Timestamp, max_date: pd.Timestamp, file_count: int) -> str:
        """Generate automatic dataset name based on date range"""
        try:
            if pd.isna(min_date) or pd.isna(max_date):
                return f"BMKG Dataset {file_count} Files"
            
            start_year = min_date.year
            end_year = max_date.year
            
            if start_year == end_year:
                return f"BMKG Data {start_year}"
            else:
                return f"BMKG Data {start_year}-{end_year}"
                
        except Exception:
            return f"BMKG Dataset {file_count} Files"

def merge_multiple_xlsx(files_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Main function to merge multiple XLSX files into single CSV
    
    Args:
        files_data: List of file data dictionaries
        
    Returns:
        Merge result dictionary
    """
    merger = BmkgMultiXlsxMerger()
    return merger.process_multiple_xlsx(files_data)