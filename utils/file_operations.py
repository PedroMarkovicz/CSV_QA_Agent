"""
File operations utilities for CSV Q&A Agent system
Handles secure file reading and validation
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

logger = logging.getLogger(__name__)

def safe_read_file(file_path: Union[str, Path], max_size_mb: int = 100) -> Optional[bytes]:
    """
    Safely read a file with size validation
    
    Args:
        file_path: Path to the file to read
        max_size_mb: Maximum file size in MB
        
    Returns:
        File content as bytes or None if failed
    """
    try:
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            logger.error(f"File does not exist: {file_path}")
            return None
        
        # Check file size
        file_size = file_path.stat().st_size
        max_size_bytes = max_size_mb * 1024 * 1024
        
        if file_size > max_size_bytes:
            logger.error(f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds limit ({max_size_mb}MB)")
            return None
        
        # Read file
        with open(file_path, 'rb') as f:
            content = f.read()
        
        logger.info(f"Successfully read file: {file_path} ({len(content)} bytes)")
        return content
        
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {str(e)}")
        return None

def validate_file_security(file_path: Union[str, Path], 
                         allowed_extensions: List[str] = None) -> bool:
    """
    Validate file for security concerns
    
    Args:
        file_path: Path to the file to validate
        allowed_extensions: List of allowed file extensions
        
    Returns:
        True if file passes security validation
    """
    try:
        file_path = Path(file_path)
        
        # Check for path traversal attempts
        if '..' in str(file_path) or str(file_path).startswith('/'):
            logger.warning(f"Suspicious file path: {file_path}")
            return False
        
        # Check file extension
        if allowed_extensions:
            file_extension = file_path.suffix.lower()
            if file_extension not in allowed_extensions:
                logger.warning(f"Disallowed file extension: {file_extension}")
                return False
        
        # Check for suspicious filenames
        suspicious_patterns = ['..', '~', '$', '%']
        filename = file_path.name
        
        for pattern in suspicious_patterns:
            if pattern in filename:
                logger.warning(f"Suspicious filename pattern '{pattern}' in: {filename}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating file {file_path}: {str(e)}")
        return False

def get_file_hash(file_content: bytes, algorithm: str = 'sha256') -> str:
    """
    Calculate hash of file content
    
    Args:
        file_content: File content as bytes
        algorithm: Hash algorithm to use
        
    Returns:
        Hex digest of the hash
    """
    try:
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(file_content)
        return hash_obj.hexdigest()
    except Exception as e:
        logger.error(f"Failed to calculate {algorithm} hash: {str(e)}")
        return ""

def ensure_directory_exists(directory_path: Union[str, Path]) -> bool:
    """
    Ensure a directory exists, create if it doesn't
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        True if directory exists or was created successfully
    """
    try:
        directory_path = Path(directory_path)
        directory_path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {directory_path}: {str(e)}")
        return False

def clean_filename(filename: str) -> str:
    """
    Clean a filename by removing or replacing problematic characters
    
    Args:
        filename: Original filename
        
    Returns:
        Cleaned filename
    """
    # Remove/replace problematic characters
    problematic_chars = ['<', '>', ':', '"', '|', '?', '*', '\\', '/']
    cleaned = filename
    
    for char in problematic_chars:
        cleaned = cleaned.replace(char, '_')
    
    # Remove leading/trailing whitespace and dots
    cleaned = cleaned.strip(' .')
    
    # Ensure filename is not empty
    if not cleaned:
        cleaned = "unnamed_file"
    
    return cleaned

def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get comprehensive information about a file
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file information
    """
    try:
        file_path = Path(file_path)
        stat = file_path.stat()
        
        return {
            'name': file_path.name,
            'size_bytes': stat.st_size,
            'size_mb': round(stat.st_size / (1024 * 1024), 2),
            'extension': file_path.suffix.lower(),
            'created_time': stat.st_ctime,
            'modified_time': stat.st_mtime,
            'is_file': file_path.is_file(),
            'is_dir': file_path.is_dir(),
            'exists': file_path.exists()
        }
    except Exception as e:
        logger.error(f"Failed to get file info for {file_path}: {str(e)}")
        return {}

def validate_csv_content(content: bytes, max_lines_to_check: int = 10) -> Dict[str, Any]:
    """
    Basic validation of CSV content
    
    Args:
        content: File content as bytes
        max_lines_to_check: Maximum number of lines to analyze
        
    Returns:
        Dictionary with validation results
    """
    try:
        # Try to decode as text
        try:
            text_content = content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                text_content = content.decode('latin1')
            except UnicodeDecodeError:
                return {
                    'valid': False,
                    'error': 'Could not decode file as text',
                    'encoding_issues': True
                }
        
        lines = text_content.split('\n')[:max_lines_to_check]
        
        if not lines:
            return {
                'valid': False,
                'error': 'File appears to be empty'
            }
        
        # Check for common CSV patterns
        first_line = lines[0] if lines else ""
        
        # Look for common delimiters
        delimiters = [',', ';', '\t', '|']
        delimiter_counts = {d: first_line.count(d) for d in delimiters}
        
        # Basic validation
        likely_delimiter = max(delimiter_counts, key=delimiter_counts.get)
        delimiter_count = delimiter_counts[likely_delimiter]
        
        if delimiter_count == 0:
            return {
                'valid': False,
                'error': 'No common CSV delimiters found',
                'delimiter_counts': delimiter_counts
            }
        
        return {
            'valid': True,
            'likely_delimiter': likely_delimiter,
            'delimiter_counts': delimiter_counts,
            'line_count': len(lines),
            'first_line_preview': first_line[:100] + '...' if len(first_line) > 100 else first_line
        }
        
    except Exception as e:
        return {
            'valid': False,
            'error': f'Validation failed: {str(e)}'
        } 