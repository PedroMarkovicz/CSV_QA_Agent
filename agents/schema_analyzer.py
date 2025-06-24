"""
SchemaAnalyzerAgent - Analisa os DataFrames carregados, gera estatísticas e sugere relações entre arquivos
"""

import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

# Configure logging for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler if not exists
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

class SchemaAnalyzerAgent:
    """Agent responsible for analyzing DataFrame schemas and suggesting relationships"""
    
    def __init__(self):
        logger.info("Initializing SchemaAnalyzerAgent...")
        self.analysis_results: Dict[str, dict] = {}
        self.relationships: List[dict] = []
        logger.info("SchemaAnalyzerAgent initialized successfully")
    
    def analyze_dataframe(self, df: pd.DataFrame, filename: str) -> dict:
        """
        Analyze a single DataFrame and extract schema information
        
        Args:
            df: DataFrame to analyze
            filename: Name of the file
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"Starting schema analysis for {filename}")
        start_time = time.time()
        
        try:
            logger.debug(f"DataFrame shape: {df.shape}")
            logger.debug(f"Columns: {list(df.columns)}")
            
            analysis = {
                'filename': filename,
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'null_counts': df.isnull().sum().to_dict(),
                'null_percentages': (df.isnull().sum() / len(df) * 100).to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'duplicate_rows': df.duplicated().sum(),
                'column_stats': {},
                'potential_keys': [],
                'data_quality_score': 0.0
            }
            
            logger.debug(f"Memory usage: {analysis['memory_usage']:,} bytes")
            logger.debug(f"Duplicate rows found: {analysis['duplicate_rows']}")
            
            # Analyze each column
            logger.info(f"Analyzing {len(df.columns)} columns...")
            for i, col in enumerate(df.columns, 1):
                logger.debug(f"Analyzing column {i}/{len(df.columns)}: {col}")
                col_analysis = self._analyze_column(df[col], col)
                analysis['column_stats'][col] = col_analysis
                
                # Check if column could be a key
                if col_analysis['unique_count'] == len(df) and col_analysis['null_count'] == 0:
                    analysis['potential_keys'].append(col)
                    logger.debug(f"Column '{col}' identified as potential key")
            
            # Calculate data quality score
            logger.debug("Calculating data quality score...")
            analysis['data_quality_score'] = self._calculate_quality_score(analysis)
            
            # Store results
            self.analysis_results[filename] = analysis
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Schema analysis completed for {filename} in {processing_time:.2f}s")
            logger.info(f"   - Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
            logger.info(f"   - Potential keys: {len(analysis['potential_keys'])}")
            logger.info(f"   - Data quality score: {analysis['data_quality_score']:.1f}/100")
            logger.info(f"   - Memory usage: {analysis['memory_usage']:,} bytes")
            
            return analysis
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error analyzing DataFrame {filename}: {str(e)}"
            logger.error(f"❌ {error_msg} (after {processing_time:.2f}s)", exc_info=True)
            return {}
    
    def _analyze_column(self, series: pd.Series, col_name: str) -> dict:
        """Analyze a single column"""
        logger.debug(f"Starting detailed analysis for column: {col_name}")
        
        analysis = {
            'name': col_name,
            'dtype': str(series.dtype),
            'null_count': series.isnull().sum(),
            'unique_count': series.nunique(),
            'unique_values': [],
            'sample_values': [],
            'pattern_type': 'unknown',
            'statistics': {}
        }
        
        logger.debug(f"Column {col_name}: dtype={analysis['dtype']}, nulls={analysis['null_count']}, unique={analysis['unique_count']}")
        
        # Get sample values
        non_null_values = series.dropna()
        if len(non_null_values) > 0:
            analysis['sample_values'] = non_null_values.head(5).tolist()
            logger.debug(f"Sample values for {col_name}: {analysis['sample_values']}")
            
            # For categorical data with few unique values, store all unique values
            if analysis['unique_count'] <= 20:
                analysis['unique_values'] = sorted(non_null_values.unique().tolist())
                logger.debug(f"All unique values for {col_name}: {len(analysis['unique_values'])} values")
        
        # Detect pattern type
        analysis['pattern_type'] = self._detect_pattern_type(series)
        logger.debug(f"Pattern type detected for {col_name}: {analysis['pattern_type']}")
        
        # Calculate statistics based on data type
        try:
            if pd.api.types.is_numeric_dtype(series):
                logger.debug(f"Calculating numeric statistics for {col_name}")
                analysis['statistics'] = {
                    'mean': series.mean(),
                    'median': series.median(),
                    'std': series.std(),
                    'min': series.min(),
                    'max': series.max(),
                    'q25': series.quantile(0.25),
                    'q75': series.quantile(0.75)
                }
            elif pd.api.types.is_datetime64_any_dtype(series):
                logger.debug(f"Calculating datetime statistics for {col_name}")
                analysis['statistics'] = {
                    'min_date': series.min(),
                    'max_date': series.max(),
                    'date_range_days': (series.max() - series.min()).days if series.max() and series.min() else 0
                }
            else:
                logger.debug(f"Calculating string statistics for {col_name}")
                analysis['statistics'] = {
                    'avg_length': series.astype(str).str.len().mean(),
                    'max_length': series.astype(str).str.len().max(),
                    'min_length': series.astype(str).str.len().min()
                }
        except Exception as e:
            logger.warning(f"Failed to calculate statistics for column {col_name}: {str(e)}")
            analysis['statistics'] = {}
        
        return analysis
    
    def _detect_pattern_type(self, series: pd.Series) -> str:
        """Detect the pattern type of a column"""
        non_null = series.dropna()
        if len(non_null) == 0:
            logger.debug("Pattern detection: empty column")
            return 'empty'
        
        # Check for common patterns
        sample_str = non_null.astype(str).iloc[0] if len(non_null) > 0 else ""
        
        if pd.api.types.is_numeric_dtype(series):
            if series.dtype == 'int64' and series.nunique() == len(series):
                logger.debug("Pattern detected: ID (unique integers)")
                return 'id'
            elif series.nunique() < 20:
                logger.debug("Pattern detected: categorical numeric")
                return 'categorical_numeric'
            else:
                logger.debug("Pattern detected: numeric")
                return 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(series):
            logger.debug("Pattern detected: datetime")
            return 'datetime'
        elif series.nunique() < 20:
            logger.debug("Pattern detected: categorical")
            return 'categorical'
        elif '@' in sample_str:
            logger.debug("Pattern detected: email")
            return 'email'
        elif sample_str.isdigit() and len(sample_str) > 8:
            logger.debug("Pattern detected: phone or ID")
            return 'phone_or_id'
        else:
            logger.debug("Pattern detected: text")
            return 'text'
    
    def _calculate_quality_score(self, analysis: dict) -> float:
        """Calculate a data quality score (0-100)"""
        logger.debug("Calculating data quality score...")
        
        score = 100.0
        
        # Penalize for missing data
        avg_null_percentage = np.mean(list(analysis['null_percentages'].values()))
        score -= avg_null_percentage
        logger.debug(f"Average null percentage: {avg_null_percentage:.2f}% (penalty applied)")
        
        # Penalize for duplicate rows
        if analysis['shape'][0] > 0:
            duplicate_percentage = (analysis['duplicate_rows'] / analysis['shape'][0]) * 100
            score -= duplicate_percentage
            logger.debug(f"Duplicate rows: {duplicate_percentage:.2f}% (penalty applied)")
        
        # Bonus for having potential keys
        if analysis['potential_keys']:
            score += 5
            logger.debug(f"Potential keys bonus: +5 points (keys: {analysis['potential_keys']})")
        
        final_score = max(0.0, min(100.0, score))
        logger.debug(f"Final quality score: {final_score:.2f}/100")
        
        return final_score
    
    def get_analysis_summary(self) -> dict:
        """Get a summary of all analysis results"""
        logger.info("Generating analysis summary...")
        
        if not self.analysis_results:
            logger.warning("No analysis results available for summary")
            return {}
        
        total_rows = sum(analysis['shape'][0] for analysis in self.analysis_results.values())
        total_columns = sum(analysis['shape'][1] for analysis in self.analysis_results.values())
        avg_quality = np.mean([analysis['data_quality_score'] for analysis in self.analysis_results.values()])
        
        summary = {
            'total_files': len(self.analysis_results),
            'total_rows': total_rows,
            'total_columns': total_columns,
            'average_quality_score': avg_quality,
            'total_relationships': len(self.relationships),
            'files_with_keys': len([a for a in self.analysis_results.values() if a['potential_keys']])
        }
        
        logger.info(f"Analysis summary generated:")
        logger.info(f"   - Total files: {summary['total_files']}")
        logger.info(f"   - Total rows: {summary['total_rows']:,}")
        logger.info(f"   - Total columns: {summary['total_columns']}")
        logger.info(f"   - Average quality: {summary['average_quality_score']:.1f}/100")
        logger.info(f"   - Files with keys: {summary['files_with_keys']}")
        
        return summary
    
    def get_file_analysis(self, filename: str) -> Optional[dict]:
        """Get analysis results for a specific file"""
        logger.debug(f"Retrieving analysis for file: {filename}")
        
        result = self.analysis_results.get(filename)
        if result:
            logger.debug(f"Analysis found for {filename}")
        else:
            logger.warning(f"No analysis found for {filename}")
        
        return result
    
    def clear(self):
        """Clear all analysis results"""
        logger.info("Clearing all analysis results...")
        files_count = len(self.analysis_results)
        relationships_count = len(self.relationships)
        
        self.analysis_results.clear()
        self.relationships.clear()
        
        logger.info(f"Cleared {files_count} file analyses and {relationships_count} relationships") 