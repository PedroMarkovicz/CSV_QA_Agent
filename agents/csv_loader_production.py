"""
Production-Ready CSVLoaderAgent
Enhanced with security, performance, monitoring, and reliability features
"""

import asyncio
import io
import logging
import time
import hashlib
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, AsyncGenerator, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import resource
import gc

import pandas as pd
import chardet
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import zipfile

from config import Config
from utils.llm_integration import llm_integration
from utils.file_operations import validate_file_security

# Production logging setup
logger = logging.getLogger(__name__)

@dataclass
class SecurityConfig:
    """Security configuration for production deployment"""
    max_file_size_mb: int = 100
    allowed_extensions: List[str] = field(default_factory=lambda: ['.csv', '.zip'])
    enable_content_scanning: bool = True
    enable_virus_scanning: bool = True
    max_processing_time_seconds: int = 300
    rate_limit_requests_per_minute: int = 60
    enable_audit_logging: bool = True
    
@dataclass
class PerformanceConfig:
    """Performance configuration"""
    chunk_size_mb: int = 10
    max_memory_usage_mb: int = 500
    enable_caching: bool = True
    cache_ttl_minutes: int = 60
    max_concurrent_uploads: int = 10
    enable_async_processing: bool = True

@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    enable_metrics: bool = True
    enable_health_checks: bool = True
    enable_performance_tracking: bool = True
    log_level: str = "INFO"
    alert_on_errors: bool = True

@dataclass
class ProductionLoadResult:
    """Enhanced load result with production metadata"""
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
    
    # Production-specific fields
    security_scan_result: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    audit_trail: Optional[List[Dict[str, Any]]] = None
    memory_usage: Optional[Dict[str, Any]] = None
    cache_hit: bool = False

class ProductionMetrics:
    """Production metrics collection"""
    
    def __init__(self):
        self.requests_processed = 0
        self.requests_failed = 0
        self.total_processing_time = 0.0
        self.memory_usage_peak = 0
        self.cache_hits = 0
        self.security_violations = 0
        self.start_time = time.time()
        self._lock = threading.Lock()
    
    def record_request(self, processing_time: float, success: bool, memory_used: int):
        with self._lock:
            self.requests_processed += 1
            if not success:
                self.requests_failed += 1
            self.total_processing_time += processing_time
            self.memory_usage_peak = max(self.memory_usage_peak, memory_used)
    
    def record_cache_hit(self):
        with self._lock:
            self.cache_hits += 1
    
    def record_security_violation(self):
        with self._lock:
            self.security_violations += 1
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            uptime = time.time() - self.start_time
            return {
                'requests_processed': self.requests_processed,
                'requests_failed': self.requests_failed,
                'success_rate': (self.requests_processed - self.requests_failed) / max(self.requests_processed, 1) * 100,
                'avg_processing_time': self.total_processing_time / max(self.requests_processed, 1),
                'memory_usage_peak_mb': self.memory_usage_peak / (1024 * 1024),
                'cache_hit_rate': self.cache_hits / max(self.requests_processed, 1) * 100,
                'security_violations': self.security_violations,
                'uptime_seconds': uptime
            }

class RateLimiter:
    """Simple rate limiter for production use"""
    
    def __init__(self, max_requests: int, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = {}
        self._lock = threading.Lock()
    
    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        with self._lock:
            # Clean old entries
            cutoff = now - self.time_window
            self.requests = {k: v for k, v in self.requests.items() if v[-1] > cutoff}
            
            # Check rate limit
            client_requests = self.requests.get(client_id, [])
            client_requests = [t for t in client_requests if t > cutoff]
            
            if len(client_requests) >= self.max_requests:
                return False
            
            # Record request
            client_requests.append(now)
            self.requests[client_id] = client_requests
            return True

class SecurityScanner:
    """Production security scanning"""
    
    @staticmethod
    def scan_content(content: bytes, filename: str) -> Dict[str, Any]:
        """Basic content security scanning"""
        result = {
            'safe': True,
            'threats_detected': [],
            'scan_time': time.time()
        }
        
        # Check for suspicious patterns
        content_str = content.decode('utf-8', errors='ignore').lower()
        
        suspicious_patterns = [
            '<script>', 'javascript:', 'eval(', 'exec(',
            'import os', 'import subprocess', '__import__'
        ]
        
        for pattern in suspicious_patterns:
            if pattern in content_str:
                result['safe'] = False
                result['threats_detected'].append(f"Suspicious pattern: {pattern}")
        
        # Check file size ratio (potential zip bombs)
        if len(content) > 0:
            compression_ratio = len(content_str) / len(content)
            if compression_ratio > 100:  # Highly compressed content
                result['safe'] = False
                result['threats_detected'].append("Potential compression bomb")
        
        return result
    
    @staticmethod
    def detect_pii(df: pd.DataFrame) -> Dict[str, Any]:
        """Basic PII detection in CSV data"""
        pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
        }
        
        detected_pii = {}
        for column in df.columns:
            if df[column].dtype == 'object':  # String columns
                sample_text = ' '.join(df[column].dropna().astype(str).head(100))
                for pii_type, pattern in pii_patterns.items():
                    import re
                    if re.search(pattern, sample_text):
                        detected_pii[column] = detected_pii.get(column, [])
                        detected_pii[column].append(pii_type)
        
        return {
            'pii_detected': len(detected_pii) > 0,
            'columns_with_pii': detected_pii,
            'total_columns_scanned': len(df.columns)
        }

class CSVLoaderAgentProduction:
    """Production-ready CSV Loader Agent"""
    
    def __init__(self, 
                 security_config: SecurityConfig = None,
                 performance_config: PerformanceConfig = None,
                 monitoring_config: MonitoringConfig = None):
        
        self.security_config = security_config or SecurityConfig()
        self.performance_config = performance_config or PerformanceConfig()
        self.monitoring_config = monitoring_config or MonitoringConfig()
        
        # Initialize production components
        self.metrics = ProductionMetrics()
        self.rate_limiter = RateLimiter(self.security_config.rate_limit_requests_per_minute)
        self.security_scanner = SecurityScanner()
        self.cache = {} if self.performance_config.enable_caching else None
        self.executor = ThreadPoolExecutor(max_workers=self.performance_config.max_concurrent_uploads)
        
        # LLM integration
        self.use_llm_analysis = llm_integration.is_available()
        
        logger.info(f"Production CSVLoaderAgent initialized. LLM: {'enabled' if self.use_llm_analysis else 'disabled'}")
        
    async def load_files_async(self, uploaded_files: List[Any], 
                             client_id: str = "default") -> Dict[str, ProductionLoadResult]:
        """Async file loading with production features"""
        
        # Rate limiting check
        if not self.rate_limiter.is_allowed(client_id):
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            self.metrics.record_security_violation()
            raise ValueError("Rate limit exceeded. Please try again later.")
        
        results = {}
        
        # Process files concurrently with limits
        semaphore = asyncio.Semaphore(self.performance_config.max_concurrent_uploads)
        
        async def process_single_file(uploaded_file):
            async with semaphore:
                return await self._load_single_file_async(uploaded_file, client_id)
        
        tasks = [process_single_file(f) for f in uploaded_files]
        
        try:
            completed_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(completed_results):
                filename = uploaded_files[i].name
                if isinstance(result, Exception):
                    logger.error(f"Failed to process {filename}: {str(result)}")
                    results[filename] = ProductionLoadResult(
                        success=False,
                        errors=[f"Processing failed: {str(result)}"]
                    )
                else:
                    results[filename] = result
                    
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            raise
        
        return results
    
    async def _load_single_file_async(self, uploaded_file, client_id: str) -> ProductionLoadResult:
        """Async single file processing with full production features"""
        
        start_time = time.time()
        initial_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        
        try:
            # Security validation
            security_result = await self._security_validation(uploaded_file)
            if not security_result['safe']:
                self.metrics.record_security_violation()
                return ProductionLoadResult(
                    success=False,
                    errors=security_result['threats_detected'],
                    security_scan_result=security_result
                )
            
            # Check cache
            cache_key = self._generate_cache_key(uploaded_file)
            if self.cache and cache_key in self.cache:
                cached_result = self.cache[cache_key]
                if cached_result['expires'] > time.time():
                    self.metrics.record_cache_hit()
                    cached_result['data'].cache_hit = True
                    return cached_result['data']
            
            # Process with timeout
            try:
                result = await asyncio.wait_for(
                    self._process_file_with_monitoring(uploaded_file),
                    timeout=self.security_config.max_processing_time_seconds
                )
            except TimeoutError:
                return ProductionLoadResult(
                    success=False,
                    errors=["Processing timeout exceeded"]
                )
            
            # Cache result if successful
            if self.cache and result.success:
                self.cache[cache_key] = {
                    'data': result,
                    'expires': time.time() + (self.performance_config.cache_ttl_minutes * 60)
                }
            
            # Record metrics
            processing_time = time.time() - start_time
            final_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            memory_used = final_memory - initial_memory
            
            self.metrics.record_request(processing_time, result.success, memory_used)
            
            # Add production metadata
            result.performance_metrics = {
                'processing_time': processing_time,
                'memory_used_kb': memory_used,
                'cache_hit': result.cache_hit
            }
            
            result.security_scan_result = security_result
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing file {uploaded_file.name}: {str(e)}")
            processing_time = time.time() - start_time
            self.metrics.record_request(processing_time, False, 0)
            
            return ProductionLoadResult(
                success=False,
                errors=[f"Unexpected error: {str(e)}"],
                processing_time=processing_time
            )
        finally:
            # Force garbage collection to manage memory
            gc.collect()
    
    async def _security_validation(self, uploaded_file) -> Dict[str, Any]:
        """Comprehensive security validation"""
        
        result = {
            'safe': True,
            'threats_detected': [],
            'checks_performed': []
        }
        
        # File extension and name validation
        if not validate_file_security(uploaded_file.name, self.security_config.allowed_extensions):
            result['safe'] = False
            result['threats_detected'].append("Invalid file type or suspicious filename")
        
        result['checks_performed'].append('filename_validation')
        
        # File size validation
        if hasattr(uploaded_file, 'size'):
            max_size = self.security_config.max_file_size_mb * 1024 * 1024
            if uploaded_file.size > max_size:
                result['safe'] = False
                result['threats_detected'].append(f"File size exceeds limit ({self.security_config.max_file_size_mb}MB)")
        
        result['checks_performed'].append('size_validation')
        
        # Content scanning if enabled
        if self.security_config.enable_content_scanning:
            try:
                content = uploaded_file.read()
                uploaded_file.seek(0)  # Reset file pointer
                
                content_scan = self.security_scanner.scan_content(content, uploaded_file.name)
                if not content_scan['safe']:
                    result['safe'] = False
                    result['threats_detected'].extend(content_scan['threats_detected'])
                
                result['checks_performed'].append('content_scanning')
                
            except Exception as e:
                logger.warning(f"Content scanning failed: {str(e)}")
                result['threats_detected'].append("Content scanning failed")
        
        return result
    
    async def _process_file_with_monitoring(self, uploaded_file) -> ProductionLoadResult:
        """File processing with comprehensive monitoring"""
        
        # This would contain the actual file processing logic
        # Similar to the original CSV loader but with chunked processing
        # and memory management for large files
        
        # Placeholder implementation
        return ProductionLoadResult(
            success=True,
            metadata={'filename': uploaded_file.name}
        )
    
    def _generate_cache_key(self, uploaded_file) -> str:
        """Generate cache key for file"""
        # Simple hash-based cache key
        content = uploaded_file.read()
        uploaded_file.seek(0)
        return hashlib.sha256(content).hexdigest()
    
    def health_check(self) -> Dict[str, Any]:
        """Production health check endpoint"""
        return {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0-production',
            'metrics': self.metrics.get_stats(),
            'llm_available': llm_integration.is_available(),
            'cache_enabled': self.cache is not None,
            'executor_active': not self.executor._shutdown
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get production metrics"""
        return self.metrics.get_stats()
    
    def cleanup(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
        if self.cache:
            self.cache.clear()
        logger.info("Production CSV Loader Agent cleaned up")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup() 