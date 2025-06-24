#!/usr/bin/env python3
"""
Debug Streamlit App - Minimal test to check LangGraph configuration

This is a simplified version to debug why LangGraph logs aren't appearing.
"""

import streamlit as st
import sys
import os
import logging
import pandas as pd

# Add the project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Configure logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import configuration and functions
from config import Config
from app import is_simple_question, answer_question

st.title("🔧 Debug: LangGraph Configuration Test")

# Show current configuration
st.subheader("Current Configuration")
col1, col2 = st.columns(2)

with col1:
    st.write(f"**ENABLE_LANGGRAPH:** `{Config.ENABLE_LANGGRAPH}`")
    st.write(f"**SIMPLE_QUESTIONS_ONLY:** `{Config.LANGGRAPH_SIMPLE_QUESTIONS_ONLY}`")
    st.write(f"**ROLLBACK_ON_ERROR:** `{Config.LANGGRAPH_ROLLBACK_ON_ERROR}`")

with col2:
    st.write(f"**COMPARISON:** `{Config.ENABLE_LANGGRAPH_COMPARISON}`")
    st.write(f"**EXECUTION_LOGGING:** `{Config.ENABLE_EXECUTION_PATH_LOGGING}`")
    st.write(f"**PERFORMANCE_LOGGING:** `{Config.ENABLE_PERFORMANCE_LOGGING}`")

# Check LangGraph availability
try:
    import app
    langgraph_available = getattr(app, 'LANGGRAPH_INTEGRATION_AVAILABLE', False)
    if langgraph_available:
        st.success("✅ LangGraph integration is AVAILABLE")
    else:
        st.error("❌ LangGraph integration is NOT AVAILABLE")
except Exception as e:
    st.error(f"❌ Error checking LangGraph: {e}")

# Test question
st.subheader("Test Question")
test_question = st.text_input("Enter your question:", value="Qual é a soma dos valores?")

if test_question:
    # Test question classification
    is_simple = is_simple_question(test_question)
    if is_simple:
        st.success(f"✅ '{test_question}' is classified as SIMPLE")
    else:
        st.warning(f"⚠️ '{test_question}' is classified as COMPLEX")

# Create mock data for testing
if st.button("🧪 Test with Mock Data"):
    st.info("Creating mock CSV data and testing question processing...")
    
    # Create test DataFrame
    mock_data = pd.DataFrame({
        'Produto': ['A', 'B', 'C', 'D', 'E'],
        'Valor': [100.0, 200.0, 300.0, 150.0, 250.0],
        'Quantidade': [1, 2, 3, 1, 2]
    })
    
    st.write("**Mock Data:**")
    st.dataframe(mock_data)
    
    # Create mock analysis results
    class MockResult:
        def __init__(self, df):
            self.success = True
            self.dataframe = df
    
    mock_analysis = {
        'test_data.csv': MockResult(mock_data)
    }
    
    # Test the question processing
    if test_question:
        st.write(f"**Testing question:** '{test_question}'")
        
        # Show what should happen
        st.write("**Expected behavior:**")
        should_use_langgraph = (
            Config.ENABLE_LANGGRAPH and 
            langgraph_available and
            (not Config.LANGGRAPH_SIMPLE_QUESTIONS_ONLY or is_simple_question(test_question))
        )
        
        if should_use_langgraph:
            st.success("🚀 Should trigger LangGraph")
        else:
            st.info("🔧 Should use current system")
        
        # Actually test it
        st.write("**Actual execution:**")
        with st.spinner("Processing..."):
            try:
                # Capture logs
                import io
                import contextlib
                
                log_capture = io.StringIO()
                
                # Create a handler that captures logs
                handler = logging.StreamHandler(log_capture)
                handler.setLevel(logging.INFO)
                
                # Add handler to root logger to capture all logs
                root_logger = logging.getLogger()
                root_logger.addHandler(handler)
                
                try:
                    # Process the question
                    result = answer_question(test_question, mock_analysis)
                    
                    # Show result
                    if result.get('success'):
                        st.success("✅ Question processed successfully")
                        st.write(f"**Answer:** {result.get('answer', 'No answer')}")
                    else:
                        st.error("❌ Question processing failed")
                        st.write(f"**Error:** {result.get('error', 'Unknown error')}")
                    
                    # Show captured logs
                    log_output = log_capture.getvalue()
                    if log_output:
                        st.write("**Captured Logs:**")
                        st.text(log_output)
                    else:
                        st.warning("⚠️ No logs captured - this might indicate a logging configuration issue")
                        
                finally:
                    # Remove handler
                    root_logger.removeHandler(handler)
                    
            except Exception as e:
                st.error(f"❌ Exception during processing: {e}")
                import traceback
                st.code(traceback.format_exc())
    else:
        st.warning("Please enter a question to test")

# Environment variable check
st.subheader("Environment Variables")
env_vars = {}
for key, value in os.environ.items():
    if 'LANGGRAPH' in key or 'ENABLE_' in key:
        env_vars[key] = value

if env_vars:
    st.write("**Relevant environment variables:**")
    for key, value in env_vars.items():
        st.write(f"- `{key}`: {value}")
else:
    st.warning("⚠️ No relevant environment variables found")

# .env file check
st.subheader(".env File Status")
env_file_path = ".env"
if os.path.exists(env_file_path):
    st.success("✅ .env file exists")
    try:
        with open(env_file_path, 'r', encoding='utf-8') as f:
            env_content = f.read()
        st.write("**.env file contents:**")
        st.code(env_content, language="bash")
    except Exception as e:
        st.error(f"❌ Could not read .env file: {e}")
else:
    st.error("❌ .env file not found")

st.markdown("---")
st.markdown("**Instructions:**")
st.markdown("""
1. Check that all configuration values are `True` where expected
2. Verify LangGraph integration is AVAILABLE  
3. Test with the mock data button
4. Look for LangGraph logs in the output
5. If no logs appear, there may be a logging configuration issue
""") 