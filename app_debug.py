#!/usr/bin/env python3
"""
DEBUG VERSION of CSV Q&A Agent with enhanced logging
"""

import logging
import os

# FORCE DEBUG LOGGING BEFORE ANY OTHER IMPORTS
os.environ['LOG_LEVEL'] = 'DEBUG'

# Configure debug logging immediately
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

# Set all agent loggers to debug level
debug_loggers = [
    'agents.question_understanding',
    'agents.csv_loader', 
    'agents.query_executor',
    'agents.answer_formatter',
    'agents.langgraph_workflow'
]

for logger_name in debug_loggers:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

print("🔧 DEBUG MODE ENABLED - Detailed logging active")
print("=" * 60)

# Now import and run the regular app
import app

# The app will now run with full debug logging enabled 