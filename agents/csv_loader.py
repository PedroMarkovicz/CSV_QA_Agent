"""
CSVLoaderAgent - Enhanced with LLM capabilities

This module provides advanced CSV file loading and analysis capabilities for the CSV Q&A Agent.
It handles the complete pipeline from file upload to data preparation and initial analysis.

Key Features:
- Intelligent encoding detection with chardet and LLM fallback
- Multiple delimiter auto-detection for CSV parsing
- ZIP file support with multi-file extraction
- Comprehensive data quality assessment
- Schema analysis with semantic type detection
- Cross-dataset relationship discovery
- LLM-powered insights and recommendations
- Security validation and file size limits
- Robust error handling with detailed logging

Architecture:
- LoadResult dataclass: Structured results container
- CSVLoaderAgent: Main orchestrator class
- Multiple specialized methods for different analysis aspects
- Integration with external LLM service for enhanced insights

Processing Pipeline:
1. File validation (security, size, format)
2. Encoding detection (chardet -> fallback -> LLM)
3. CSV parsing (multi-delimiter, quote-aware)
4. Schema analysis (types, semantics, statistics)
5. Quality assessment (completeness, duplicates, issues)
6. LLM insights generation (when available)
7. Cross-dataset relationship analysis
"""

import pandas as pd
import zipfile
import io
import logging
import chardet
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass

# Import configuration and utility modules
from config import Config
import utils.llm_integration as llm_module

# Configure logging for this module with appropriate verbosity
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler if not exists to ensure logging output
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


@dataclass
class LoadResult:
    """
    Structured container for CSV loading operation results.

    This dataclass encapsulates all possible outcomes from a file loading operation,
    providing a consistent interface for success/failure handling and comprehensive
    analysis results.

    Attributes:
        success (bool): Whether the loading operation succeeded
        dataframe (Optional[pd.DataFrame]): The loaded pandas DataFrame
        metadata (Optional[Dict]): File and parsing metadata (size, encoding, etc.)
        schema_analysis (Optional[Dict]): Column types, statistics, semantic analysis
        quality_assessment (Optional[Dict]): Data quality scores and recommendations
        relationships (Optional[List[Dict]]): Detected relationships with other datasets
        errors (Optional[List[str]]): Error messages if operation failed
        warnings (Optional[List[str]]): Non-fatal warnings about data issues
        processing_time (Optional[float]): Time taken to process the file
        llm_insights (Optional[Dict]): AI-generated insights and recommendations
    """

    success: bool
    dataframe: Optional[pd.DataFrame] = None
    metadata: Optional[Dict[str, Any]] = None
    schema_analysis: Optional[Dict[str, Any]] = None
    quality_assessment: Optional[Dict[str, Any]] = None
    relationships: Optional[List[Dict[str, Any]]] = None
    errors: Optional[List[str]] = None
    warnings: Optional[List[str]] = None
    processing_time: Optional[float] = None
    llm_insights: Optional[Dict[str, Any]] = None


class CSVLoaderAgent:
    """
    Enhanced CSV Loader Agent with LLM capabilities for intelligent file processing.

    This agent is responsible for the complete file loading and initial analysis pipeline.
    It combines traditional data processing techniques with modern LLM capabilities to
    provide robust, intelligent file handling that can adapt to various data formats
    and quality issues.

    Key Capabilities:
    - Multi-format support (CSV, ZIP archives)
    - Intelligent encoding detection with multiple fallback strategies
    - Automatic delimiter and quote character detection
    - Comprehensive data quality assessment
    - Schema analysis with semantic type detection
    - Cross-dataset relationship discovery
    - LLM-enhanced error analysis and recommendations
    - Security validation and resource management

    Processing Flow:
    1. Security and format validation
    2. Encoding detection and content decoding
    3. CSV parsing with intelligent parameter detection
    4. Data quality assessment and schema analysis
    5. LLM-powered insights generation (when available)
    6. Cross-dataset relationship analysis (for multiple files)
    """

    def __init__(self):
        """
        Initialize the CSV Loader Agent with configuration and settings.

        Sets up all necessary parameters for file processing, including:
        - File size limits and security constraints
        - Encoding detection candidates and strategies
        - CSV parsing options and delimiters
        - LLM integration status and capabilities
        """
        logger.info("Initializing CSVLoaderAgent...")

        # File processing constraints from configuration
        self.max_file_size = Config.MAX_FILE_SIZE_MB * 1024 * 1024  # Convert to bytes
        self.supported_extensions = Config.SUPPORTED_EXTENSIONS
        self.default_encoding = Config.DEFAULT_ENCODING
        self.max_retries = Config.MAX_RETRIES

        # Encoding detection settings - ordered by likelihood and reliability
        # Covers most common encodings found in real-world CSV files
        self.encoding_candidates = [
            "utf-8",
            "utf-8-sig",
            "latin1",
            "cp1252",
            "iso-8859-1",
            "cp850",
            "windows-1252",
        ]

        # CSV parsing settings - common delimiters in order of prevalence
        self.csv_delimiters = [",", ";", "\t", "|"]
        self.sample_size = 10000  # Bytes for encoding detection sample

        # Note: LLM availability is checked dynamically, not fixed at initialization
        # This allows the agent to adapt when providers are changed by the user

        # Log initialization status for debugging and monitoring
        logger.info(f"CSVLoaderAgent initialized successfully")
        logger.info(f"Max file size: {Config.MAX_FILE_SIZE_MB}MB")
        logger.info(f"Supported extensions: {self.supported_extensions}")
        logger.info(f"Encoding candidates: {len(self.encoding_candidates)} options")
        logger.info(f"CSV delimiters: {self.csv_delimiters}")

        # Log current LLM status at initialization
        current_llm_status = llm_module.llm_integration.is_available()
        logger.info(
            f"LLM features: {'enabled' if current_llm_status else 'disabled'} (checked dynamically)"
        )

    @property
    def use_llm_analysis(self) -> bool:
        """
        Dynamically check if LLM analysis should be used.
        This allows the agent to adapt when the user changes providers.
        """
        return llm_module.llm_integration.is_available()

    def load_files(self, uploaded_files: List[Any]) -> Dict[str, LoadResult]:
        """
        Load multiple uploaded files with enhanced analysis and cross-file relationships.

        This is the main entry point for batch file processing. It handles multiple files
        efficiently, provides comprehensive analysis for each file, and discovers
        relationships between datasets when multiple files are processed.

        Processing Steps:
        1. Iterate through each uploaded file
        2. Perform individual file loading and analysis
        3. Track processing metrics and success/failure rates
        4. Analyze relationships between successfully loaded datasets
        5. Return comprehensive results for all files

        Args:
            uploaded_files (List[Any]): List of uploaded file objects from Streamlit

        Returns:
            Dict[str, LoadResult]: Dictionary mapping filenames to their analysis results

        Features:
        - Parallel processing capability (when needed)
        - Comprehensive error handling per file
        - Cross-dataset relationship analysis
        - Performance monitoring and logging
        - Graceful degradation on partial failures
        """
        logger.info(
            f"Starting batch file loading process for {len(uploaded_files)} files"
        )
        start_batch_time = time.time()

        # Initialize tracking variables for batch processing metrics
        results = {}
        successful_files = 0
        failed_files = 0

        # Process each file individually with comprehensive error handling
        for i, uploaded_file in enumerate(uploaded_files, 1):
            logger.info(
                f"Processing file {i}/{len(uploaded_files)}: {uploaded_file.name}"
            )

            try:
                # Time individual file processing for performance monitoring
                start_time = time.time()
                result = self._load_single_file(uploaded_file)
                result.processing_time = time.time() - start_time
                results[uploaded_file.name] = result

                # Log processing results and update metrics
                if result.success:
                    successful_files += 1
                    logger.info(
                        f"✅ Successfully processed {uploaded_file.name} in {result.processing_time:.2f}s"
                    )
                    if result.dataframe is not None:
                        logger.info(f"   - Shape: {result.dataframe.shape}")
                        logger.info(
                            f"   - Memory usage: {result.metadata.get('memory_usage', 'unknown')} bytes"
                        )
                else:
                    failed_files += 1
                    logger.error(
                        f"❌ Failed to process {uploaded_file.name}: {result.errors}"
                    )

            except Exception as e:
                # Handle unexpected errors gracefully without stopping batch processing
                failed_files += 1
                error_msg = (
                    f"Unexpected error processing {uploaded_file.name}: {str(e)}"
                )
                logger.error(error_msg, exc_info=True)
                results[uploaded_file.name] = LoadResult(
                    success=False,
                    errors=[error_msg],
                    processing_time=time.time() - start_time
                    if "start_time" in locals()
                    else 0,
                )

        # Analyze relationships between loaded datasets (only with LLM and multiple files)
        if len(results) > 1 and self.use_llm_analysis:
            logger.info("Starting cross-dataset relationship analysis...")
            self._analyze_cross_dataset_relationships(results)

        # Log final batch processing statistics
        batch_time = time.time() - start_batch_time
        logger.info(f"Batch processing completed in {batch_time:.2f}s")
        logger.info(f"Results: {successful_files} successful, {failed_files} failed")

        return results

    def _load_single_file(self, uploaded_file) -> LoadResult:
        """
        Load a single file with comprehensive analysis and validation.

        This method handles the complete pipeline for processing a single file:
        1. Security validation to prevent malicious uploads
        2. File size validation against configured limits
        3. Format detection and appropriate handler selection
        4. Comprehensive error handling and logging

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            LoadResult: Complete analysis results for the file

        Security Features:
        - File extension validation
        - Content validation
        - Size limit enforcement
        - Path traversal protection
        """
        logger.debug(f"Starting single file load for: {uploaded_file.name}")

        # Initialize error and warning collectors
        errors = []
        warnings = []

        # STEP 1: Security validation to prevent malicious file uploads
        logger.debug("Performing security validation...")
        if not self._validate_file_security(uploaded_file):
            error_msg = "File failed security validation"
            logger.warning(f"Security validation failed for {uploaded_file.name}")
            return LoadResult(success=False, errors=[error_msg])

        # STEP 2: File size validation against configured limits
        if hasattr(uploaded_file, "size"):
            file_size_mb = uploaded_file.size / (1024 * 1024)
            logger.debug(
                f"File size: {file_size_mb:.2f}MB (limit: {Config.MAX_FILE_SIZE_MB}MB)"
            )

            if uploaded_file.size > self.max_file_size:
                error_msg = f"File size ({file_size_mb:.1f}MB) exceeds limit ({Config.MAX_FILE_SIZE_MB}MB)"
                logger.warning(error_msg)
                return LoadResult(success=False, errors=[error_msg])

        # STEP 3: Determine file type and route to appropriate handler
        file_extension = Path(uploaded_file.name).suffix.lower()
        logger.debug(f"File extension detected: {file_extension}")

        if file_extension == ".zip":
            # Handle ZIP archives containing multiple CSV files
            logger.info("Processing as ZIP file...")
            return self._load_zip_file(uploaded_file)
        elif file_extension == ".csv":
            # Handle individual CSV files
            logger.info("Processing as CSV file...")
            return self._load_csv_file(uploaded_file)
        else:
            # Unsupported file format
            error_msg = f"Unsupported file type: {file_extension}"
            logger.error(error_msg)
            return LoadResult(success=False, errors=[error_msg])

    def _load_csv_file(self, uploaded_file) -> LoadResult:
        """
        Load CSV file with intelligent parsing and comprehensive analysis.

        This method implements the complete CSV processing pipeline:
        1. File content reading and validation
        2. Intelligent encoding detection with multiple strategies
        3. CSV parsing with automatic delimiter detection
        4. Schema analysis and semantic type detection
        5. Data quality assessment with scoring
        6. LLM-powered insights generation (when available)

        The method uses a multi-stage approach to handle various CSV formats
        and encoding issues commonly found in real-world data files.

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            LoadResult: Complete analysis results including DataFrame and metadata

        Processing Stages:
        - Content Reading: Safe file content extraction
        - Encoding Detection: chardet -> fallback -> LLM suggestions
        - CSV Parsing: Multi-delimiter testing with quote handling
        - Analysis Pipeline: Schema, quality, insights generation
        """
        logger.debug(f"Starting CSV loading process for {uploaded_file.name}")

        # Initialize error and warning collectors
        errors = []
        warnings = []

        # STAGE 1: Read file content safely with error handling
        logger.debug("Reading file content...")
        try:
            file_content = uploaded_file.read()
            uploaded_file.seek(0)  # Reset file pointer for potential re-reading
            logger.debug(f"File content read successfully: {len(file_content)} bytes")
        except Exception as e:
            error_msg = f"Failed to read file: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return LoadResult(success=False, errors=[error_msg])

        # STAGE 2: Detect file encoding using multiple strategies
        logger.debug("Starting encoding detection...")
        encoding_result = self._detect_encoding(file_content, uploaded_file.name)
        if not encoding_result["success"]:
            logger.error(
                f"Encoding detection failed: {encoding_result.get('errors', [])}"
            )
            return LoadResult(
                success=False,
                errors=encoding_result.get("errors", ["Encoding detection failed"]),
                llm_insights=encoding_result.get("llm_analysis"),
            )

        encoding = encoding_result["encoding"]
        logger.info(
            f"Encoding detected: {encoding} (confidence: {encoding_result.get('confidence', 'unknown')})"
        )

        # STAGE 3: Parse CSV content with intelligent delimiter detection
        logger.debug("Starting CSV parsing...")
        parsing_result = self._parse_csv_content(
            file_content, encoding, uploaded_file.name
        )
        if not parsing_result["success"]:
            logger.error(f"CSV parsing failed: {parsing_result.get('errors', [])}")
            return LoadResult(
                success=False,
                errors=parsing_result.get("errors", ["CSV parsing failed"]),
                warnings=parsing_result.get("warnings", []),
                llm_insights=parsing_result.get("llm_analysis"),
            )

        df = parsing_result["dataframe"]
        parsing_metadata = parsing_result["metadata"]
        logger.info(
            f"CSV parsed successfully - Shape: {df.shape}, Delimiter: {parsing_metadata.get('delimiter')}"
        )

        # STAGE 4: Generate comprehensive metadata about the file and parsing
        logger.debug("Generating metadata...")
        metadata = self._generate_metadata(
            df, uploaded_file.name, encoding, parsing_metadata
        )

        # STAGE 5: Perform schema analysis to understand data structure
        logger.debug("Performing schema analysis...")
        schema_analysis = self._analyze_schema(df, uploaded_file.name)
        logger.debug(
            f"Schema analysis completed - {len(schema_analysis.get('columns', []))} columns analyzed"
        )

        # STAGE 6: Assess data quality with scoring and recommendations
        logger.debug("Assessing data quality...")
        quality_assessment = self._assess_data_quality(df, uploaded_file.name)
        quality_score = quality_assessment.get("overall_score", 0)
        logger.info(
            f"Data quality assessment completed - Overall score: {quality_score}/100"
        )

        # STAGE 7: Generate LLM-enhanced insights (when available)
        llm_insights = None
        if self.use_llm_analysis:
            logger.debug("Generating LLM insights...")
            llm_insights = self._generate_llm_insights(df, uploaded_file.name, metadata)
            logger.debug("LLM insights generated")
        else:
            logger.debug("LLM analysis disabled - skipping insights generation")

        logger.info(f"CSV file {uploaded_file.name} loaded and analyzed successfully")

        # Return comprehensive results
        return LoadResult(
            success=True,
            dataframe=df,
            metadata=metadata,
            schema_analysis=schema_analysis,
            quality_assessment=quality_assessment,
            errors=errors,
            warnings=warnings,
            llm_insights=llm_insights,
        )

    def _detect_encoding(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Detect file encoding with multiple strategies and LLM fallback for complex cases.

        This method implements a sophisticated encoding detection pipeline:
        1. chardet library for automatic detection
        2. Common encoding fallbacks with decode testing
        3. LLM-enhanced analysis for problematic files

        The multi-stage approach ensures maximum compatibility with various
        file sources and encoding issues commonly encountered in real-world data.

        Args:
            file_content (bytes): Raw file content as bytes
            filename (str): Filename for logging and LLM context

        Returns:
            Dict[str, Any]: Detection result with encoding, confidence, and method used

        Detection Strategies:
        - chardet: Automatic detection with confidence scoring
        - Fallback: Testing common encodings manually
        - LLM: AI analysis of encoding patterns and suggestions
        """
        logger.debug(f"Starting encoding detection for {filename}")

        # STRATEGY 1: Try chardet for automatic detection
        try:
            logger.debug("Trying chardet encoding detection...")
            detected = chardet.detect(file_content[: self.sample_size])
            confidence = detected.get("confidence", 0)
            encoding = detected.get("encoding")

            logger.debug(f"Chardet result: {encoding} (confidence: {confidence})")

            # Only trust chardet results with high confidence
            if encoding and confidence > 0.7:
                # Validate the encoding by testing decode operation
                try:
                    file_content.decode(encoding)
                    logger.info(
                        f"Encoding successfully detected via chardet: {encoding}"
                    )
                    return {
                        "success": True,
                        "encoding": encoding,
                        "confidence": confidence,
                        "method": "chardet",
                    }
                except UnicodeDecodeError:
                    logger.debug(
                        f"Chardet suggested encoding {encoding} failed decode test"
                    )
        except Exception as e:
            logger.warning(f"Chardet failed for {filename}: {str(e)}")

        # STRATEGY 2: Try common encoding fallbacks manually
        logger.debug("Trying common encoding fallbacks...")
        for encoding in self.encoding_candidates:
            try:
                file_content.decode(encoding)
                logger.info(f"Encoding detected via fallback: {encoding}")
                return {
                    "success": True,
                    "encoding": encoding,
                    "confidence": 0.5,
                    "method": "fallback",
                }
            except UnicodeDecodeError:
                logger.debug(f"Encoding {encoding} failed")
                continue

        # STRATEGY 3: LLM-enhanced encoding analysis for complex cases
        llm_analysis = None
        if self.use_llm_analysis:
            logger.debug("Attempting LLM-enhanced encoding analysis...")
            try:
                # Provide hex representation of file bytes for LLM analysis
                file_bytes_hex = file_content[:100].hex()
                llm_analysis = llm_module.llm_integration.analyze_encoding_error(
                    filename=filename,
                    attempted_encodings=self.encoding_candidates,
                    error_message="All standard encodings failed",
                    file_bytes_hex=file_bytes_hex,
                )

                # Test LLM-suggested encodings
                if llm_analysis and llm_analysis.get("suggested_encodings"):
                    logger.debug(
                        f"LLM suggested encodings: {llm_analysis['suggested_encodings']}"
                    )
                    for encoding in llm_analysis["suggested_encodings"]:
                        try:
                            file_content.decode(encoding)
                            logger.info(
                                f"Encoding detected via LLM suggestion: {encoding}"
                            )
                            return {
                                "success": True,
                                "encoding": encoding,
                                "confidence": llm_analysis.get("confidence", 0.3),
                                "method": "llm_suggestion",
                                "llm_analysis": llm_analysis,
                            }
                        except UnicodeDecodeError:
                            logger.debug(f"LLM suggested encoding {encoding} failed")
                            continue
            except Exception as e:
                logger.warning(f"LLM encoding analysis failed: {str(e)}")

        # All encoding detection methods failed
        logger.error(f"All encoding detection methods failed for {filename}")
        return {
            "success": False,
            "errors": ["Could not determine file encoding"],
            "llm_analysis": llm_analysis,
        }

    def _parse_csv_content(
        self, file_content: bytes, encoding: str, filename: str
    ) -> Dict[str, Any]:
        """
        Parse CSV content with intelligent delimiter detection and error recovery.

        This method implements sophisticated CSV parsing that can handle various
        formats and problematic files by testing multiple parsing strategies:
        1. Standard delimiter detection with quote character testing
        2. Error recovery with bad line skipping
        3. LLM-assisted parsing for complex formats

        Args:
            file_content (bytes): Raw file content
            encoding (str): Detected encoding for content decoding
            filename (str): Filename for logging and context

        Returns:
            Dict[str, Any]: Parsing result with DataFrame and metadata

        Parsing Strategies:
        - Multi-delimiter testing: Common separators (comma, semicolon, tab, pipe)
        - Quote character detection: Various quote styles
        - Error handling: Skip bad lines, handle malformed data
        - LLM assistance: AI-powered parsing suggestions for edge cases
        """
        logger.debug(
            f"Starting CSV content parsing for {filename} with encoding {encoding}"
        )

        # Decode file content using detected encoding
        try:
            text_content = file_content.decode(encoding)
            logger.debug(
                f"Content decoded successfully: {len(text_content)} characters"
            )
        except UnicodeDecodeError as e:
            error_msg = f"Failed to decode with {encoding}: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "errors": [error_msg]}

        # STRATEGY 1: Try different parsing options systematically
        logger.debug("Attempting CSV parsing with different delimiters...")
        for delimiter in self.csv_delimiters:
            logger.debug(f"Trying delimiter: '{delimiter}'")
            try:
                # Test different quote characters for each delimiter
                for quotechar in ['"', "'", None]:
                    logger.debug(f"  Trying quotechar: {repr(quotechar)}")
                    try:
                        df = pd.read_csv(
                            io.StringIO(text_content),
                            delimiter=delimiter,
                            quotechar=quotechar,
                            on_bad_lines="skip",  # Skip problematic lines
                        )

                        # Validate parsing result quality
                        if len(df.columns) > 1 and len(df) > 0:
                            metadata = {
                                "delimiter": delimiter,
                                "quotechar": quotechar,
                                "encoding": encoding,
                                "parsing_method": "standard",
                            }

                            logger.info(
                                f"CSV parsing successful with delimiter '{delimiter}' and quotechar {repr(quotechar)}"
                            )
                            logger.debug(f"Parsed DataFrame shape: {df.shape}")

                            return {
                                "success": True,
                                "dataframe": df,
                                "metadata": metadata,
                            }

                    except Exception as parse_error:
                        logger.debug(
                            f"    Parsing failed with {repr(quotechar)}: {str(parse_error)}"
                        )
                        continue

            except Exception as e:
                logger.debug(f"  Delimiter '{delimiter}' failed: {str(e)}")
                continue

        # STRATEGY 2: LLM-enhanced parsing error analysis for complex formats
        llm_analysis = None
        if self.use_llm_analysis:
            logger.debug("Attempting LLM-enhanced parsing analysis...")
            try:
                # Provide file preview for LLM to analyze structure
                preview = text_content[:500]
                llm_analysis = llm_module.llm_integration.analyze_parsing_error(
                    filename=filename,
                    error_message="Standard parsing methods failed",
                    error_line="N/A",
                    delimiter="auto-detect",
                    file_preview=preview,
                )

                # Apply LLM-suggested parsing options
                if llm_analysis and llm_analysis.get("parsing_options"):
                    options = llm_analysis["parsing_options"]
                    logger.debug(f"LLM suggested parsing options: {options}")

                    try:
                        df = pd.read_csv(
                            io.StringIO(text_content),
                            delimiter=options.get("delimiter", ","),
                            quotechar=options.get("quotechar", '"'),
                            skiprows=options.get("skiprows", 0),
                            header=options.get("header", 0),
                            on_bad_lines="skip",
                        )

                        metadata = {
                            "delimiter": options.get("delimiter", ","),
                            "quotechar": options.get("quotechar", '"'),
                            "encoding": encoding,
                            "parsing_method": "llm_assisted",
                            "skiprows": options.get("skiprows", 0),
                            "header": options.get("header", 0),
                        }

                        logger.info(f"CSV parsing successful using LLM suggestions")
                        logger.debug(f"LLM-assisted DataFrame shape: {df.shape}")

                        return {
                            "success": True,
                            "dataframe": df,
                            "metadata": metadata,
                            "llm_analysis": llm_analysis,
                        }

                    except Exception as e:
                        logger.debug(f"LLM-assisted parsing failed: {str(e)}")

            except Exception as e:
                logger.warning(f"LLM parsing analysis failed: {str(e)}")

        # All parsing strategies failed
        logger.error(f"All CSV parsing methods failed for {filename}")
        return {
            "success": False,
            "errors": ["Could not parse CSV content with any known method"],
            "llm_analysis": llm_analysis,
        }

    def _analyze_schema(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Analyze DataFrame schema with LLM enhancement"""
        logger.debug(f"Starting schema analysis for {filename}")

        schema_info = {
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "shape": df.shape,
            "column_analysis": {},
        }

        logger.debug(f"Analyzing {len(df.columns)} columns: {df.columns.tolist()}")

        # Analyze each column
        for i, column in enumerate(df.columns, 1):
            logger.debug(f"Analyzing column {i}/{len(df.columns)}: {column}")

            col_analysis = {
                "dtype": str(df[column].dtype),
                "null_count": int(df[column].isnull().sum()),
                "null_percentage": float(df[column].isnull().sum() / len(df) * 100),
                "unique_count": int(df[column].nunique()),
                "unique_percentage": float(df[column].nunique() / len(df) * 100),
                "sample_values": df[column].dropna().head(10).tolist(),
            }

            logger.debug(
                f"  Column {column}: {col_analysis['dtype']}, {col_analysis['unique_count']} unique, {col_analysis['null_count']} nulls"
            )

            # LLM-enhanced column type analysis
            if self.use_llm_analysis and len(df) > 0:
                try:
                    logger.debug(f"  Running LLM analysis for column {column}")
                    llm_analysis = llm_module.llm_integration.analyze_column_type(
                        column_name=column,
                        sample_data=col_analysis["sample_values"],
                        total_count=len(df),
                        unique_count=col_analysis["unique_count"],
                        null_count=col_analysis["null_count"],
                        current_dtype=str(df[column].dtype),
                    )

                    if llm_analysis:
                        col_analysis["llm_analysis"] = llm_analysis
                        col_analysis["semantic_type"] = llm_analysis.get(
                            "semantic_type", "unknown"
                        )
                        col_analysis["recommended_dtype"] = llm_analysis.get(
                            "recommended_dtype", str(df[column].dtype)
                        )
                        col_analysis["quality_issues"] = llm_analysis.get(
                            "quality_issues", []
                        )
                        logger.debug(
                            f"  LLM detected semantic type: {col_analysis['semantic_type']}"
                        )
                except Exception as e:
                    logger.warning(f"LLM column analysis failed for {column}: {str(e)}")

            schema_info["column_analysis"][column] = col_analysis

        logger.info(
            f"Schema analysis completed for {filename} - {len(df.columns)} columns analyzed"
        )
        return schema_info

    def _assess_data_quality(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Assess data quality with comprehensive metrics"""
        logger.debug(f"Starting data quality assessment for {filename}")

        # Basic quality metrics
        total_cells = df.size
        null_cells = df.isnull().sum().sum()
        completeness = (
            float((total_cells - null_cells) / total_cells * 100)
            if total_cells > 0
            else 0
        )

        quality_metrics = {
            "completeness": completeness,
            "duplicate_rows": int(df.duplicated().sum()),
            "duplicate_percentage": float(df.duplicated().sum() / len(df) * 100)
            if len(df) > 0
            else 0,
            "empty_columns": [col for col in df.columns if df[col].isnull().all()],
            "constant_columns": [col for col in df.columns if df[col].nunique() <= 1],
            "high_cardinality_columns": [
                col for col in df.columns if df[col].nunique() > len(df) * 0.9
            ],
        }

        logger.debug(f"Basic quality metrics calculated:")
        logger.debug(f"  Completeness: {completeness:.1f}%")
        logger.debug(f"  Duplicate rows: {quality_metrics['duplicate_rows']}")
        logger.debug(f"  Empty columns: {len(quality_metrics['empty_columns'])}")
        logger.debug(f"  Constant columns: {len(quality_metrics['constant_columns'])}")
        logger.debug(
            f"  High cardinality columns: {len(quality_metrics['high_cardinality_columns'])}"
        )

        # Calculate overall quality score
        completeness_score = quality_metrics["completeness"]
        duplicate_penalty = min(quality_metrics["duplicate_percentage"] * 2, 20)
        empty_col_penalty = len(quality_metrics["empty_columns"]) * 10
        constant_col_penalty = len(quality_metrics["constant_columns"]) * 5

        quality_score = max(
            0,
            completeness_score
            - duplicate_penalty
            - empty_col_penalty
            - constant_col_penalty,
        )
        quality_metrics["overall_score"] = round(quality_score, 2)

        logger.debug(
            f"Quality score calculation: {completeness_score:.1f} - {duplicate_penalty:.1f} - {empty_col_penalty:.1f} - {constant_col_penalty:.1f} = {quality_score:.1f}"
        )

        # LLM-enhanced quality assessment
        if self.use_llm_analysis:
            try:
                logger.debug("Running LLM-enhanced quality assessment...")
                sample_data = {}
                for col in df.columns[
                    :10
                ]:  # Limit to first 10 columns for API efficiency
                    sample_data[col] = df[col].dropna().head(5).tolist()

                llm_assessment = llm_module.llm_integration.assess_data_quality(
                    filename=filename,
                    rows=len(df),
                    columns=len(df.columns),
                    column_names=list(df.columns),
                    dtypes=df.dtypes.astype(str).to_dict(),
                    null_counts=df.isnull().sum().to_dict(),
                    duplicate_rows=int(df.duplicated().sum()),
                    sample_data=sample_data,
                )

                if llm_assessment:
                    quality_metrics["llm_assessment"] = llm_assessment
                    quality_metrics["quality_issues"] = llm_assessment.get(
                        "quality_issues", []
                    )
                    quality_metrics["recommendations"] = llm_assessment.get(
                        "recommended_actions", []
                    )
                    quality_metrics["insights"] = llm_assessment.get(
                        "data_insights", []
                    )
                    logger.debug(
                        f"LLM found {len(quality_metrics['quality_issues'])} quality issues"
                    )
            except Exception as e:
                logger.warning(f"LLM quality assessment failed: {str(e)}")

        logger.info(
            f"Data quality assessment completed for {filename} - Overall score: {quality_score:.1f}/100"
        )
        return quality_metrics

    def _generate_llm_insights(
        self, df: pd.DataFrame, filename: str, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive LLM insights about the dataset"""
        logger.debug(f"Generating LLM insights for {filename}")

        if not self.use_llm_analysis:
            logger.debug("LLM analysis disabled, skipping insights generation")
            return None

        try:
            # Generate basic insights
            insights = {
                "summary": f"Dataset {filename} contains {len(df)} rows and {len(df.columns)} columns",
                "key_observations": [],
                "recommended_next_steps": [],
                "potential_use_cases": [],
            }

            # Add column type insights
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            text_cols = df.select_dtypes(include=["object"]).columns.tolist()

            logger.debug(
                f"Column types: {len(numeric_cols)} numeric, {len(text_cols)} text"
            )

            if numeric_cols:
                insights["key_observations"].append(
                    f"Contains {len(numeric_cols)} numeric columns: {', '.join(numeric_cols[:3])}{'...' if len(numeric_cols) > 3 else ''}"
                )

            if text_cols:
                insights["key_observations"].append(
                    f"Contains {len(text_cols)} text columns: {', '.join(text_cols[:3])}{'...' if len(text_cols) > 3 else ''}"
                )

            # Add data quality insights
            null_percentage = (df.isnull().sum().sum() / df.size) * 100
            if null_percentage > 10:
                insights["key_observations"].append(
                    f"Data has {null_percentage:.1f}% missing values - consider data cleaning"
                )

            # Suggest next steps
            if numeric_cols:
                insights["recommended_next_steps"].append(
                    "Perform statistical analysis on numeric columns"
                )

            if text_cols:
                insights["recommended_next_steps"].append(
                    "Analyze text patterns and categories"
                )

            if len(df) > 1000:
                insights["recommended_next_steps"].append(
                    "Consider sampling for exploratory analysis"
                )

            # Suggest use cases
            if len(numeric_cols) > 2:
                insights["potential_use_cases"].append(
                    "Regression analysis or predictive modeling"
                )

            if text_cols and numeric_cols:
                insights["potential_use_cases"].append(
                    "Classification or clustering analysis"
                )

            logger.info(
                f"LLM insights generated for {filename}: {len(insights['key_observations'])} observations, {len(insights['recommended_next_steps'])} recommendations"
            )
            return insights

        except Exception as e:
            logger.error(
                f"Failed to generate LLM insights for {filename}: {str(e)}",
                exc_info=True,
            )
            return {
                "summary": f"Dataset {filename} contains {len(df)} rows and {len(df.columns)} columns",
                "error": f"Insight generation failed: {str(e)}",
            }

    def _analyze_cross_dataset_relationships(self, results: Dict[str, LoadResult]):
        """Analyze relationships between multiple loaded datasets"""
        logger.info("Starting cross-dataset relationship analysis...")

        if not self.use_llm_analysis:
            logger.debug("LLM analysis disabled, skipping relationship analysis")
            return

        successful_results = {
            name: result for name, result in results.items() if result.success
        }

        if len(successful_results) < 2:
            logger.debug(
                f"Only {len(successful_results)} successful results, need at least 2 for relationship analysis"
            )
            return

        logger.info(
            f"Analyzing relationships between {len(successful_results)} datasets"
        )

        try:
            # Compare columns across datasets to identify potential relationships
            datasets = list(successful_results.items())
            relationships_found = 0

            for i in range(len(datasets)):
                for j in range(i + 1, len(datasets)):
                    name1, result1 = datasets[i]
                    name2, result2 = datasets[j]

                    logger.debug(f"Comparing {name1} with {name2}")

                    # Find potential relationship columns
                    relationships = self._find_column_relationships(
                        name1, result1.dataframe, name2, result2.dataframe
                    )

                    if relationships:
                        relationships_found += len(relationships)
                        logger.info(
                            f"Found {len(relationships)} relationships between {name1} and {name2}"
                        )

                        if not result1.relationships:
                            result1.relationships = []
                        if not result2.relationships:
                            result2.relationships = []

                        result1.relationships.extend(relationships)
                        result2.relationships.extend(relationships)

            logger.info(
                f"Cross-dataset relationship analysis completed - {relationships_found} relationships found"
            )

        except Exception as e:
            logger.error(
                f"Cross-dataset relationship analysis failed: {str(e)}", exc_info=True
            )

    def _find_column_relationships(
        self, table1_name: str, df1: pd.DataFrame, table2_name: str, df2: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Find relationships between columns in two DataFrames"""
        logger.debug(
            f"Finding column relationships between {table1_name} and {table2_name}"
        )

        relationships = []
        comparisons_made = 0

        try:
            for col1 in df1.columns:
                for col2 in df2.columns:
                    comparisons_made += 1
                    logger.debug(
                        f"Comparing {table1_name}.{col1} with {table2_name}.{col2}"
                    )

                    # Calculate overlap
                    values1 = set(df1[col1].dropna().astype(str))
                    values2 = set(df2[col2].dropna().astype(str))

                    overlap = values1.intersection(values2)
                    overlap_count = len(overlap)

                    if overlap_count > 0:
                        overlap_percentage = (
                            overlap_count / max(len(values1), len(values2))
                        ) * 100
                        logger.debug(
                            f"  Overlap found: {overlap_count} values ({overlap_percentage:.1f}%)"
                        )

                        if overlap_percentage > 10:  # Only analyze significant overlaps
                            logger.debug(
                                f"  Significant overlap detected, running LLM analysis..."
                            )
                            # LLM analysis of relationship
                            try:
                                llm_analysis = (
                                    llm_module.llm_integration.detect_relationships(
                                        table1_name=table1_name,
                                        column1_name=col1,
                                        column1_samples=list(values1)[:10],
                                        table2_name=table2_name,
                                        column2_name=col2,
                                        column2_samples=list(values2)[:10],
                                        overlap_count=overlap_count,
                                        overlap_percentage=overlap_percentage,
                                    )
                                )

                                if llm_analysis and llm_analysis.get(
                                    "has_relationship"
                                ):
                                    relationship = {
                                        "table1": table1_name,
                                        "column1": col1,
                                        "table2": table2_name,
                                        "column2": col2,
                                        "overlap_count": overlap_count,
                                        "overlap_percentage": overlap_percentage,
                                        "relationship_type": llm_analysis.get(
                                            "relationship_type"
                                        ),
                                        "strength": llm_analysis.get("strength"),
                                        "confidence": llm_analysis.get("confidence"),
                                        "explanation": llm_analysis.get("explanation"),
                                        "recommendations": llm_analysis.get(
                                            "recommendations", []
                                        ),
                                    }
                                    relationships.append(relationship)
                                    logger.info(
                                        f"  ✅ Relationship detected: {col1} ↔ {col2} ({llm_analysis.get('relationship_type', 'unknown')})"
                                    )
                            except Exception as e:
                                logger.warning(
                                    f"LLM relationship analysis failed for {col1} ↔ {col2}: {str(e)}"
                                )
                    else:
                        logger.debug(f"  No overlap found")

            logger.debug(
                f"Column relationship analysis completed: {comparisons_made} comparisons, {len(relationships)} relationships found"
            )

        except Exception as e:
            logger.error(
                f"Column relationship analysis failed: {str(e)}", exc_info=True
            )

        return relationships

    def _load_zip_file(self, uploaded_file) -> LoadResult:
        """Load ZIP file containing CSV files"""
        logger.info(f"Loading ZIP file: {uploaded_file.name}")

        try:
            zip_content = uploaded_file.read()
            uploaded_file.seek(0)
            logger.debug(f"ZIP content read: {len(zip_content)} bytes")

            with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_file:
                all_files = zip_file.namelist()
                csv_files = [
                    name for name in all_files if name.lower().endswith(".csv")
                ]

                logger.debug(
                    f"ZIP contains {len(all_files)} files, {len(csv_files)} CSV files"
                )
                logger.debug(f"CSV files found: {csv_files}")

                if not csv_files:
                    error_msg = "No CSV files found in ZIP archive"
                    logger.error(error_msg)
                    return LoadResult(success=False, errors=[error_msg])

                if len(csv_files) > 1:
                    error_msg = "ZIP contains multiple CSV files. Please upload individual files for multi-file analysis."
                    logger.warning(error_msg)
                    return LoadResult(success=False, errors=[error_msg])

                # Extract and process the single CSV file
                csv_name = csv_files[0]
                logger.info(f"Extracting CSV file: {csv_name}")
                csv_content = zip_file.read(csv_name)
                logger.debug(f"Extracted CSV content: {len(csv_content)} bytes")

                # Create a mock uploaded file object
                class MockUploadedFile:
                    def __init__(self, content, name):
                        self.content = content
                        self.name = name
                        self.size = len(content)

                    def read(self):
                        return self.content

                    def seek(self, pos):
                        pass

                mock_file = MockUploadedFile(csv_content, csv_name)
                logger.debug(f"Created mock file object for {csv_name}")

                result = self._load_csv_file(mock_file)
                logger.info(
                    f"ZIP file processed {'successfully' if result.success else 'with errors'}"
                )
                return result

        except zipfile.BadZipFile:
            error_msg = "Invalid ZIP file"
            logger.error(error_msg)
            return LoadResult(success=False, errors=[error_msg])
        except Exception as e:
            error_msg = f"Failed to process ZIP file: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return LoadResult(success=False, errors=[error_msg])

    def _validate_file_security(self, uploaded_file) -> bool:
        """Validate file for security concerns"""
        logger.debug(f"Validating file security for: {uploaded_file.name}")

        # Basic filename validation
        if ".." in uploaded_file.name or uploaded_file.name.startswith("/"):
            logger.warning(
                f"Security violation - suspicious filename: {uploaded_file.name}"
            )
            return False

        # Extension validation
        file_extension = Path(uploaded_file.name).suffix.lower()
        if file_extension not in self.supported_extensions:
            logger.warning(
                f"Security violation - unsupported extension: {file_extension}"
            )
            return False

        logger.debug(f"File security validation passed for: {uploaded_file.name}")
        return True

    def _generate_metadata(
        self,
        df: pd.DataFrame,
        filename: str,
        encoding: str,
        parsing_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate comprehensive metadata for the loaded DataFrame"""
        logger.debug(f"Generating metadata for {filename}")

        memory_usage = df.memory_usage(deep=True).sum()

        metadata = {
            "filename": filename,
            "encoding": encoding,
            "parsing_info": parsing_metadata,
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "memory_usage": memory_usage,
            "load_timestamp": time.time(),
        }

        logger.debug(
            f"Metadata generated - Memory usage: {memory_usage} bytes, Shape: {df.shape}"
        )
        return metadata

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        logger.debug(f"Returning supported formats: {self.supported_extensions}")
        return self.supported_extensions.copy()

    def get_llm_status(self) -> Dict[str, Any]:
        """Get LLM integration status and usage statistics"""
        logger.debug("Getting LLM status and usage statistics")

        status = {
            "enabled": self.use_llm_analysis,
            "available": llm_module.llm_integration.is_available(),
            "usage_stats": llm_module.llm_integration.get_usage_stats(),
        }

        logger.debug(
            f"LLM status: enabled={status['enabled']}, available={status['available']}"
        )
        return status
