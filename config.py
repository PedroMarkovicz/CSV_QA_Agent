"""
Configuration module for CSV Q&A Agent

This module centralizes all application configuration settings, environment variables,
and system parameters. It provides a structured approach to managing:

- API keys and external service credentials
- File processing constraints and defaults
- Logging configuration and output formatting
- Directory paths and project structure
- Runtime parameters and agent settings

The configuration supports both environment variables and .env file loading,
with sensible defaults for development and production environments.

Features:
- Automatic .env file loading
- Environment variable override support
- Configuration validation methods
- Automatic directory creation
- Type-safe parameter access
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file if it exists
# This allows developers to set configuration locally without modifying code
load_dotenv()


class Config:
    """
    Application configuration settings with environment variable support.

    This class consolidates all configuration parameters for the CSV Q&A Agent,
    providing a single source of truth for application settings. All values
    can be overridden via environment variables for different deployment scenarios.

    Configuration Categories:
    - API Keys: External service authentication
    - File Processing: Upload limits and file handling
    - Logging: Output formatting and verbosity
    - Paths: Directory structure and file locations
    - Runtime: Performance and timeout settings
    """

    # =============================================================================
    # API KEYS AND EXTERNAL SERVICES
    # =============================================================================

    # OpenAI API key for LLM functionality (ChatGPT integration)
    # Required for advanced question understanding and AI insights
    # Can be set via environment variable: OPENAI_API_KEY=sk-...
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    # Groq API key for fallback LLM functionality
    # Used when OpenAI is unavailable or fails
    # Can be set via environment variable: GROQ_API_KEY=gsk-...
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

    # Default model for Groq API
    # Used when falling back to Groq service
    GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")

    # LangChain API key for advanced LLM orchestration (optional)
    # Used for enhanced prompt chaining and model management
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")

    # =============================================================================
    # FILE PROCESSING CONFIGURATION
    # =============================================================================

    # Maximum file size limit in megabytes for uploads
    # Prevents system overload from extremely large files
    # Default: 100MB, can be overridden via MAX_FILE_SIZE_MB environment variable
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "100"))

    # Default character encoding for CSV file parsing
    # Used when automatic encoding detection fails
    # Most common: 'utf-8', alternatives: 'latin1', 'cp1252'
    DEFAULT_ENCODING = os.getenv("DEFAULT_ENCODING", "utf-8")

    # List of supported file extensions for upload validation
    # Currently supports CSV files and ZIP archives containing CSVs
    SUPPORTED_EXTENSIONS = [".csv", ".zip"]

    # =============================================================================
    # LOGGING AND DEBUGGING CONFIGURATION
    # =============================================================================

    # Logging verbosity level - controls what gets logged to files/console
    # Options: DEBUG (most verbose), INFO, WARNING, ERROR (least verbose)
    # DEBUG: All operations including prompts and generated code
    # INFO: Normal operations and important events
    # WARNING: Potential issues and fallback usage
    # ERROR: Only failures and critical problems
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # Standard format string for all log messages
    # Includes timestamp, logger name, severity level, and message content
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # =============================================================================
    # PROJECT DIRECTORY STRUCTURE
    # =============================================================================

    # Root directory of the project (where config.py is located)
    # All other paths are relative to this location
    PROJECT_ROOT = Path(__file__).parent

    # Directory for storing uploaded and sample data files
    # Contains CSV examples and processed datasets
    DATA_DIR = PROJECT_ROOT / "data"

    # Directory for application log files
    # Separated by component: app.log, security.log, performance.log, etc.
    LOGS_DIR = PROJECT_ROOT / "logs"

    # Directory for LLM prompt templates and response formatting
    # Contains structured prompts for different types of analysis
    TEMPLATES_DIR = PROJECT_ROOT / "templates"

    # =============================================================================
    # STREAMLIT WEB SERVER CONFIGURATION
    # =============================================================================

    # Port number for the Streamlit web interface
    # Default: 8501 (Streamlit standard), can be changed for deployment
    STREAMLIT_SERVER_PORT = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))

    # IP address/hostname for Streamlit server binding
    # 'localhost' for development, '0.0.0.0' for production deployment
    STREAMLIT_SERVER_ADDRESS = os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost")

    # =============================================================================
    # AGENT RUNTIME PARAMETERS
    # =============================================================================

    # Maximum number of retry attempts for failed operations
    # Applies to: API calls, file processing, code execution
    # Prevents infinite loops while allowing resilience to transient failures
    MAX_RETRIES = 3

    # Timeout in seconds for long-running operations
    # Applies to: LLM API calls, code execution, file processing
    # Prevents system hang on problematic operations
    TIMEOUT_SECONDS = 30

    # =============================================================================
    # LANGGRAPH INTEGRATION SETTINGS (EXPERIMENTAL)
    # =============================================================================

    # Enable LangGraph workflow orchestration (experimental feature)
    # Set to 'true' to test LangGraph alongside existing system
    # Default: 'false' (uses current working system)
    ENABLE_LANGGRAPH = os.getenv("ENABLE_LANGGRAPH", "false").lower() == "true"

    # Enable A/B testing between current system and LangGraph
    # Runs both systems and compares results (for validation)
    # Default: 'false' (no comparison logging)
    ENABLE_LANGGRAPH_COMPARISON = (
        os.getenv("ENABLE_LANGGRAPH_COMPARISON", "true").lower() == "true"
    )

    # Automatic rollback to current system if LangGraph fails
    # Set to 'false' only when you're confident LangGraph is stable
    # Default: 'true' (always fallback to working system)
    LANGGRAPH_ROLLBACK_ON_ERROR = (
        os.getenv("LANGGRAPH_ROLLBACK_ON_ERROR", "true").lower() == "true"
    )

    # Test LangGraph only on simple questions (for gradual rollout)
    # Filters which questions go to LangGraph vs current system
    # Default: 'true' (only simple patterns like mean, sum, count)
    LANGGRAPH_SIMPLE_QUESTIONS_ONLY = (
        os.getenv("LANGGRAPH_SIMPLE_QUESTIONS_ONLY", "true").lower() == "true"
    )

    # Enable detailed execution path logging
    # Shows which system (LangGraph vs Current) is handling each request
    # Default: 'true' (always log execution paths for monitoring)
    ENABLE_EXECUTION_PATH_LOGGING = (
        os.getenv("ENABLE_EXECUTION_PATH_LOGGING", "true").lower() == "true"
    )

    # Log detailed performance metrics for comparison
    # Includes timing, success rates, and answer comparison between systems
    # Default: 'false' (only enable for detailed analysis)
    ENABLE_PERFORMANCE_LOGGING = (
        os.getenv("ENABLE_PERFORMANCE_LOGGING", "true").lower() == "true"
    )

    @classmethod
    def validate(cls) -> bool:
        """
        Validate that required configuration is present and accessible.

        Checks for critical configuration parameters that are needed for
        core functionality. While some features may work without all parameters,
        this validation helps identify missing configuration early.

        Returns:
            bool: True if all required configuration is present, False otherwise

        Note:
            Missing configuration is logged as warnings but doesn't prevent startup.
            The application uses graceful degradation when optional features
            (like LLM integration) are unavailable.
        """
        # Define configuration keys that are highly recommended for full functionality
        # Note: OPENAI_API_KEY is not strictly required as the system has regex fallback
        required_keys = ["OPENAI_API_KEY"]

        # Check which required keys are missing or empty
        missing_keys = [key for key in required_keys if not getattr(cls, key)]

        if missing_keys:
            # Log warning but don't fail - system can work with degraded functionality
            print(f"Warning: Missing required configuration: {', '.join(missing_keys)}")
            print("The system will work with limited functionality (regex-only mode)")
            return False
        return True

    @classmethod
    def create_directories(cls):
        """
        Create necessary directories if they don't exist.

        Ensures that all required directories for the application exist before
        startup. This prevents runtime errors when trying to write logs or
        access template files.

        Created directories:
        - data/: For uploaded files and datasets
        - logs/: For application log files
        - templates/: For LLM prompt templates

        All directories are created with exist_ok=True to avoid errors if
        they already exist.
        """
        # List of critical directories that must exist for proper operation
        for directory in [cls.DATA_DIR, cls.LOGS_DIR, cls.TEMPLATES_DIR]:
            # Create directory and any missing parent directories
            directory.mkdir(exist_ok=True)


# Initialize directory structure on module import
# This ensures directories exist before any other code tries to use them
Config.create_directories()
