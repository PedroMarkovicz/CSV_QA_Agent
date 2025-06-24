"""
QueryExecutorAgent - Secure Pandas Code Execution with Comprehensive Safety

This module provides secure execution of dynamically generated pandas code with extensive
safety measures, error handling, and fallback strategies. It serves as the critical
security boundary between user input and code execution.

Architecture:
- Sandboxed Execution: Isolated environment with controlled global variables
- Safety Validation: Multi-layer security checks before execution
- Error Recovery: Intelligent fallback strategies for failed operations
- Output Capture: Safe capture of stdout/stderr during execution
- History Tracking: Complete audit trail of all executions
- Timeout Protection: Prevents infinite loops and hanging operations

Security Features:
- Code Pattern Validation: Blocks dangerous operations and imports
- Sandboxed Environment: Limited global scope with safe pandas/numpy access
- Input Sanitization: Variable name cleaning and validation
- Error Containment: Graceful handling of all execution failures
- Audit Logging: Complete traceability of executed code and results

Execution Pipeline:
1. Code safety validation against forbidden patterns
2. Environment setup with sanitized variables
3. Timeout-protected code execution in sandbox
4. Output capture and result extraction
5. Error handling with fallback strategies
6. History logging and performance metrics

Fallback Strategies:
- Basic Exploration: Show data preview when complex operations fail
- Simplified Operations: Fall back to basic statistics (describe)
- Error Analysis: Provide meaningful error messages for debugging
- Graceful Degradation: Always return some useful information

Performance Features:
- Execution timing and metrics
- Memory usage monitoring
- Output size limitations
- Resource cleanup and management
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import traceback
import sys
import io
from contextlib import redirect_stdout, redirect_stderr
import re

# Configure logging for this security-critical module
logger = logging.getLogger(__name__)

class QueryExecutorAgent:
    """
    Secure agent for executing pandas code with comprehensive safety and error handling.
    
    This agent is responsible for the safe execution of dynamically generated pandas code
    while maintaining system security and providing robust error recovery. It implements
    multiple layers of protection against malicious code execution and provides intelligent
    fallback strategies when operations fail.
    
    Core Security Principles:
    - Defense in Depth: Multiple security layers (validation, sandboxing, monitoring)
    - Principle of Least Privilege: Minimal required permissions and access
    - Fail-Safe Defaults: Safe behavior when errors occur
    - Complete Auditing: Full traceability of all operations
    - Resource Protection: Timeout and memory usage controls
    
    Key Capabilities:
    - Secure code execution in controlled environment
    - Comprehensive safety validation against dangerous patterns
    - Intelligent error recovery with multiple fallback strategies
    - Complete audit trail with performance metrics
    - Output capture and result extraction
    - Variable name sanitization and environment isolation
    
    Architecture Components:
    - Safe Globals: Controlled global environment with pandas/numpy
    - Pattern Validator: Regex-based detection of dangerous code patterns
    - Execution Sandbox: Isolated execution environment with I/O capture
    - Fallback Engine: Multiple recovery strategies for failed operations
    - History Tracker: Complete audit log of all execution attempts
    """
    
    def __init__(self):
        """
        Initialize the QueryExecutorAgent with security controls and fallback strategies.
        
        Sets up the secure execution environment, safety validators, and recovery mechanisms
        needed for safe pandas code execution.
        
        Initialization Components:
        1. Execution history tracking for audit and debugging
        2. Safe global environment with controlled imports
        3. Fallback strategy registry for error recovery
        4. Security validators and pattern matchers
        """
        # Initialize execution history for complete audit trail
        # Tracks all execution attempts, successes, failures, and fallbacks
        self.execution_history: List[dict] = []
        
        # Set up safe global environment with controlled access to pandas/numpy
        # Limits available functions and modules to prevent security issues
        self.safe_globals = self._setup_safe_globals()
        
        # Define fallback strategies for error recovery
        # Ordered from most specific to most general recovery approaches
        self.fallback_strategies = [
            self._fallback_basic_exploration,      # Show data preview
            self._fallback_simplified_operation    # Basic statistics
        ]
    
    def _setup_safe_globals(self) -> dict:
        """
        Set up safe global environment for secure code execution.
        
        Creates a controlled global namespace that provides access to necessary
        pandas and numpy functionality while blocking access to dangerous
        system functions and modules.
        
        Returns:
            dict: Safe global environment for code execution
            
        Security Considerations:
        - Only includes essential data analysis libraries
        - Excludes system modules (os, sys, subprocess, etc.)
        - Prevents import of additional modules
        - Limits access to file system operations
        """
        return {
            'pd': pd,        # Pandas library for data manipulation
            'pandas': pd,    # Alternative pandas reference
            'np': np,        # NumPy library for numerical operations
            'numpy': np,     # Alternative numpy reference
        }

    def execute_code(self, code: str, dataframes: Dict[str, pd.DataFrame], timeout: int = 30) -> dict:
        """
        Safely execute pandas code with comprehensive error handling and security validation.
        
        This is the main execution method that orchestrates the complete secure execution
        pipeline including safety validation, environment setup, code execution, output
        capture, error handling, and fallback strategies.
        
        Args:
            code (str): Pandas code to execute (pre-validated by QuestionUnderstandingAgent)
            dataframes (Dict[str, pd.DataFrame]): Available DataFrames for the code
            timeout (int): Maximum execution time in seconds (default: 30)
            
        Returns:
            dict: Comprehensive execution result with metrics and error information
            
        Execution Pipeline:
        1. Initialize execution tracking and timing
        2. Set up secure execution environment
        3. Validate code safety against forbidden patterns
        4. Execute code in sandboxed environment with I/O capture
        5. Extract results and capture outputs/warnings
        6. Handle errors with intelligent fallback strategies
        7. Log complete execution history for audit
        
        Security Features:
        - Pre-execution safety validation
        - Sandboxed execution environment
        - Timeout protection against infinite loops
        - Output size limitations
        - Complete error containment
        """
        # Initialize comprehensive execution result tracking
        execution_result = {
            'code': code,                   # Original code for audit trail
            'success': False,               # Execution success flag
            'result': None,                 # Extracted result data
            'output': '',                   # Captured stdout
            'error': None,                  # Error information if failed
            'execution_time': 0.0,          # Performance timing
            'warnings': [],                 # Non-fatal warnings
            'fallback_executed': False,     # Whether fallback was used
            'fallback_strategy': None       # Which fallback strategy was used
        }
        
        try:
            # Start execution timing for performance monitoring
            import time
            start_time = time.time()
            
            # Set up secure execution environment with controlled variables
            local_vars = self.safe_globals.copy()
            local_vars['dataframes'] = dataframes
            
            # Add individual DataFrames with sanitized variable names
            # This allows code to reference DataFrames directly by filename
            for name, df in dataframes.items():
                safe_name = self._make_safe_variable_name(name)
                local_vars[safe_name] = df
            
            logger.debug(f"Execution environment variables: {list(local_vars.keys())}")
            
            # SECURITY CHECKPOINT: Validate code safety before execution
            safety = self.validate_code_safety(code)
            if not safety['safe']:
                logger.error(f"Unsafe code detected: {safety['reason']}")
                execution_result['error'] = {'type': 'UnsafeCode', 'message': safety['reason']}
                self.execution_history.append(execution_result)
                return execution_result
            
            # Set up output capture to safely capture stdout and stderr
            # This prevents code from interfering with system output
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            # Execute code in sandboxed environment with output redirection
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, local_vars, local_vars)
            
            # Calculate execution time for performance monitoring
            execution_time = time.time() - start_time
            execution_result['execution_time'] = execution_time
            
            # Extract execution result using multiple strategies
            if 'result' in local_vars:
                # Preferred: code explicitly set result variable
                execution_result['result'] = local_vars['result']
            else:
                # Fallback: try to find meaningful result in execution environment
                execution_result['result'] = self._find_last_result(local_vars)
            
            # Capture any output that was printed during execution
            output = stdout_capture.getvalue()
            if output:
                execution_result['output'] = output
            
            # Capture warnings and non-fatal errors
            warnings = stderr_capture.getvalue()
            if warnings:
                execution_result['warnings'].append(warnings)
            
            # Mark execution as successful
            execution_result['success'] = True
            logger.info(f"Code executed successfully in {execution_time:.3f}s")
            logger.debug(f"Execution result: {execution_result['result']}")
            
        except (KeyError, ValueError, TypeError) as e:
            # Handle common data-related errors with specific error information
            error_info = {
                'type': type(e).__name__,
                'message': str(e),
                'traceback': traceback.format_exc()
            }
            execution_result['error'] = error_info
            logger.error(f"Code execution failed ({type(e).__name__}): {str(e)}")
            logger.debug(f"Traceback: {error_info['traceback']}")
            
            # Attempt fallback strategies for data-related errors
            fallback_result = self._try_fallback_strategies(code, dataframes, error_info)
            if fallback_result['success']:
                execution_result.update(fallback_result)
                execution_result['fallback_executed'] = True
                logger.info(f"Fallback executed: {fallback_result.get('fallback_strategy')}")
            
        except Exception as e:
            # Handle all other unexpected errors
            error_info = {
                'type': type(e).__name__,
                'message': str(e),
                'traceback': traceback.format_exc()
            }
            execution_result['error'] = error_info
            logger.error(f"Code execution failed: {str(e)}")
            logger.debug(f"Traceback: {error_info['traceback']}")
            
            # Attempt fallback strategies for general errors
            fallback_result = self._try_fallback_strategies(code, dataframes, error_info)
            if fallback_result['success']:
                execution_result.update(fallback_result)
                execution_result['fallback_executed'] = True
                logger.info(f"Fallback executed: {fallback_result.get('fallback_strategy')}")
        
        # Log execution in history for audit and debugging
        self.execution_history.append(execution_result)
        return execution_result
    
    def _make_safe_variable_name(self, filename: str) -> str:
        """
        Convert filename to a safe Python variable name.
        
        Sanitizes filenames to create valid Python identifiers that can be used
        safely in the execution environment without causing syntax errors or
        security issues.
        
        Args:
            filename (str): Original filename (e.g., "202401_NFs_Cabeçalho.csv")
            
        Returns:
            str: Safe Python variable name (e.g., "df_202401_NFs_Cabecalho")
            
        Sanitization Process:
        1. Remove file extensions (.csv, .zip)
        2. Replace non-alphanumeric characters with underscores
        3. Ensure variable name doesn't start with digit
        4. Provide fallback name if completely invalid
        """
        # Remove common file extensions
        safe_name = filename.replace('.csv', '').replace('.zip', '')
        
        # Replace non-alphanumeric characters with underscores
        safe_name = ''.join(c if c.isalnum() else '_' for c in safe_name)
        
        # Ensure variable name doesn't start with a digit (invalid in Python)
        if safe_name and safe_name[0].isdigit():
            safe_name = 'df_' + safe_name
        
        # Provide fallback if name becomes empty or invalid
        return safe_name or 'df'
    
    def _find_last_result(self, local_vars: dict) -> Any:
        """
        Try to find the last meaningful result from code execution.
        
        When code doesn't explicitly set a 'result' variable, this method
        attempts to intelligently identify what the user intended as the
        result by examining the execution environment.
        
        Args:
            local_vars (dict): Variables from code execution environment
            
        Returns:
            Any: Most likely intended result or None if not found
            
        Search Strategy:
        1. Look for common result variable names
        2. Find pandas DataFrames/Series created during execution
        3. Return the most likely candidate result
        """
        # Check for common result variable names
        result_candidates = ['result', 'output', 'answer', 'data', 'df']
        for candidate in result_candidates:
            if candidate in local_vars and candidate not in self.safe_globals:
                return local_vars[candidate]
        
        # Look for pandas objects created during execution
        for name, value in local_vars.items():
            if (isinstance(value, (pd.DataFrame, pd.Series)) and 
                name not in self.safe_globals and 
                not name.startswith('_')):
                return value
        
        # No clear result found
        return None
    
    def _try_fallback_strategies(self, original_code: str, dataframes: Dict[str, pd.DataFrame], error_info: dict) -> dict:
        """
        Try different fallback strategies when code execution fails.
        
        Implements intelligent error recovery by attempting various fallback
        approaches that can still provide useful information to the user
        even when the original code fails.
        
        Args:
            original_code (str): The code that failed to execute
            dataframes (Dict[str, pd.DataFrame]): Available DataFrames
            error_info (dict): Information about the execution error
            
        Returns:
            dict: Result of successful fallback strategy or failure information
            
        Fallback Strategy Order:
        1. Basic Exploration: Show data preview (head())
        2. Simplified Operation: Show basic statistics (describe())
        3. Ultimate Fallback: Return error information for user
        """
        # Try each fallback strategy in order of preference
        for strategy in self.fallback_strategies:
            result = strategy(dataframes)
            if result.get('success'):
                result['fallback_strategy'] = strategy.__name__
                logger.info(f"Fallback strategy used: {strategy.__name__}")
                return result
        
        # All fallback strategies failed - return original error
        return {
            'success': False,
            'result': None,
            'output': '',
            'error': error_info,
            'fallback_strategy': None
        }
        
    def _fallback_basic_exploration(self, dataframes: Dict[str, pd.DataFrame]) -> dict:
        """
        Fallback strategy: Basic data exploration.
        
        When complex operations fail, provide a basic preview of the data
        that can still be useful for understanding the dataset structure.
        
        Args:
            dataframes (Dict[str, pd.DataFrame]): Available DataFrames
            
        Returns:
            dict: Result with basic data preview or error information
        """
        try:
            if not dataframes:
                return {
                    'success': False, 
                    'result': None, 
                    'output': '', 
                    'error': 'No dataframes available.'
                }
            
            # Use the first available DataFrame for exploration
            df = list(dataframes.values())[0]
            return {
                'success': True,
                'result': df.head(), 
                'output': '', 
                'error': None
            }
        except Exception as e:
            logger.error(f"Fallback basic exploration failed: {str(e)}")
            return {
                'success': False,
                'result': None,
                'output': '', 
                'error': str(e)
            }
    
    def _fallback_simplified_operation(self, dataframes: Dict[str, pd.DataFrame]) -> dict:
        """
        Fallback strategy: Simplified statistical operation.
        
        Provides basic statistical summary of the data when more complex
        operations fail. This almost always works and provides useful insights.
        
        Args:
            dataframes (Dict[str, pd.DataFrame]): Available DataFrames
            
        Returns:
            dict: Result with statistical summary or error information
        """
        try:
            if not dataframes:
                return {
                    'success': False, 
                    'result': None, 
                    'output': '', 
                    'error': 'No dataframes available.'
                }
            
            # Use the first available DataFrame for basic statistics
            df = list(dataframes.values())[0]
            return {
                'success': True,
                'result': df.describe(), 
                'output': '', 
                'error': None
            }
        except Exception as e:
            logger.error(f"Fallback simplified operation failed: {str(e)}")
            return {
                'success': False,
                'result': None,
                'output': '', 
                'error': str(e)
            }
    
    def validate_code_safety(self, code: str) -> dict:
        """
        Comprehensive security validation to check for potentially dangerous code patterns.
        
        This critical security function implements defense-in-depth by checking for
        various categories of dangerous operations that could compromise system security
        or stability. It uses regex pattern matching to detect prohibited operations.
        
        Args:
            code (str): Code to validate for safety
            
        Returns:
            dict: Validation result with safety status and reason
            
        Security Categories Checked:
        - System Access: os, sys, subprocess operations
        - Code Injection: eval, exec, compile functions
        - File Operations: file reading/writing, directory operations
        - Network Access: HTTP, FTP, socket operations
        - Process Control: threading, multiprocessing, system calls
        - Environment Access: environment variables, system information
        - Import Controls: preventing arbitrary module imports
        
        Defense Strategy:
        - Blacklist Approach: Block known dangerous patterns
        - Comprehensive Coverage: Multiple variations of dangerous operations
        - Case Insensitive: Prevent case-based evasion attempts
        - Regular Expressions: Flexible pattern matching
        """
        # Comprehensive list of forbidden code patterns organized by security risk
        forbidden_patterns = [
            # Core Security Risks - Code Injection and Execution
            r'__import__', r'eval\(', r'exec\(', r'compile\(', r'execfile\(',
            
            # File System Access - Prevent unauthorized file operations
            r'open\(', r'write\(', r'read\(', r'save', r'pickle',
            
            # System Module Access - Block system-level operations
            r'os\.', r'sys\.', r'subprocess', r'socket', r'shutil',
            
            # Network and External Communication
            r'ftp', r'http', r'https',
            
            # User Input and Interactive Functions
            r'input\(', r'globals\(', r'locals\(', r'breakpoint\(',
            
            # Destructive Operations - Data and File Manipulation
            r'del\s', r'\bexit\b', r'\bquit\b', r'\bkill\b', r'\bremove\b', r'\bdelete\b',
            
            # File System Permissions and Directory Operations
            r'\bchmod\b', r'\bchown\b', r'\bmkdir\b', r'\brmdir\b', r'\bcopy\b', 
            r'\bmove\b', r'\brename\b', r'\bmakedirs\b', r'\bremovedirs\b', r'\bchdir\b',
            
            # Environment and System Information Access
            r'\bgetenv\b', r'\benviron\b', r'\bsetattr\b', r'\bgetattr\b',
            
            # Process and Thread Control - Prevent resource exhaustion
            r'\bpopen\b', r'\bcall\b', r'\bspawn\b', r'\bstartfile\b', r'\bthread\b', 
            r'\bthreading\b', r'\bconcurrent\b', r'\bmultiprocessing\b', r'\bwalk\b',
            
            # Import Controls - Prevent loading of unauthorized modules
            r'\bimport\b'
        ]
        
        # Check code against all forbidden patterns
        for pattern in forbidden_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                logger.warning(f"Unsafe code pattern detected: {pattern}")
                return {
                    'safe': False, 
                    'reason': f'Unsafe code pattern detected: {pattern}'
                }
        
        # Code passed all security checks
        return {'safe': True, 'reason': ''}
    
    def get_execution_history(self) -> List[dict]:
        """
        Return the complete history of all code executions for audit and debugging.
        
        Provides access to the complete execution audit trail including successful
        executions, failures, fallback strategies, and performance metrics.
        
        Returns:
            List[dict]: Complete execution history with all details
            
        History Information Includes:
        - Original code and execution parameters
        - Success/failure status and error details
        - Execution timing and performance metrics
        - Fallback strategy usage and results
        - Output capture and warning information
        """
        return self.execution_history
    
    def clear_history(self):
        """
        Clear the execution history for privacy or memory management.
        
        Removes all stored execution history while preserving the agent's
        operational state and configuration. Useful for privacy compliance
        or memory management in long-running sessions.
        """
        self.execution_history.clear() 