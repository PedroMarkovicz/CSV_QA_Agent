#!/usr/bin/env python3
"""
Setup debug logging for detailed troubleshooting
"""

import logging
import os
import sys

def setup_debug_logging():
    """Setup detailed debug logging to see all internal operations"""
    
    # Set environment variable for debug level
    os.environ['LOG_LEVEL'] = 'DEBUG'
    
    # Configure root logger for debug level
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True  # Override any existing configuration
    )
    
    # Set specific loggers to debug level
    loggers_to_debug = [
        'agents.question_understanding',
        'agents.csv_loader', 
        'agents.query_executor',
        'agents.answer_formatter',
        'agents.langgraph_workflow',
        'root'
    ]
    
    for logger_name in loggers_to_debug:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
    
    print("🔧 Debug logging enabled!")
    print("📝 You should now see detailed logs in the terminal including:")
    print("   - Column identification steps")
    print("   - Question normalization process") 
    print("   - DataFrame analysis details")
    print("   - Code generation steps")
    print("   - All debug messages")
    print()
    print("💡 To run Streamlit with debug logging:")
    print("   python setup_debug_logging.py && streamlit run app.py")

if __name__ == "__main__":
    setup_debug_logging() 