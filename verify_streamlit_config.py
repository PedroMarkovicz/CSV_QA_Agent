#!/usr/bin/env python3
"""
Streamlit Configuration Verification App

This app shows the exact configuration that Streamlit sees when it starts.
Run with: streamlit run verify_streamlit_config.py
"""

import streamlit as st
import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(__file__))

from config import Config

def main():
    st.title("🔧 LangGraph Configuration Verification")
    st.markdown("This app shows the current LangGraph configuration that Streamlit sees.")
    
    # Show environment file status
    st.subheader("📁 Environment File Status")
    env_path = ".env"
    if os.path.exists(env_path):
        st.success(f"✅ .env file exists at: {os.path.abspath(env_path)}")
        
        # Read and display .env contents
        with open(env_path, 'r') as f:
            env_content = f.read()
        
        st.code(env_content, language="bash")
    else:
        st.error("❌ .env file not found")
    
    st.subheader("⚙️ Current Configuration")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Core Settings:**")
        st.write(f"ENABLE_LANGGRAPH: `{Config.ENABLE_LANGGRAPH}`")
        st.write(f"LANGGRAPH_SIMPLE_QUESTIONS_ONLY: `{Config.LANGGRAPH_SIMPLE_QUESTIONS_ONLY}`")
        st.write(f"LANGGRAPH_ROLLBACK_ON_ERROR: `{Config.LANGGRAPH_ROLLBACK_ON_ERROR}`")
    
    with col2:
        st.markdown("**Monitoring Settings:**")
        st.write(f"ENABLE_LANGGRAPH_COMPARISON: `{Config.ENABLE_LANGGRAPH_COMPARISON}`")
        st.write(f"ENABLE_EXECUTION_PATH_LOGGING: `{Config.ENABLE_EXECUTION_PATH_LOGGING}`")
        st.write(f"ENABLE_PERFORMANCE_LOGGING: `{Config.ENABLE_PERFORMANCE_LOGGING}`")
    
    # Check LangGraph availability
    st.subheader("🔌 LangGraph Integration Status")
    try:
        import app
        if hasattr(app, 'LANGGRAPH_INTEGRATION_AVAILABLE'):
            if app.LANGGRAPH_INTEGRATION_AVAILABLE:
                st.success("✅ LangGraph integration is AVAILABLE")
            else:
                st.error("❌ LangGraph integration is NOT AVAILABLE")
        else:
            st.warning("⚠️ LANGGRAPH_INTEGRATION_AVAILABLE not found in app module")
    except Exception as e:
        st.error(f"❌ Error checking LangGraph availability: {e}")
    
    # Test question classification
    st.subheader("🧪 Question Classification Test")
    test_question = st.text_input("Test a question:", value="Qual é a soma dos valores?")
    
    if test_question:
        try:
            from app import is_simple_question
            is_simple = is_simple_question(test_question)
            
            if is_simple:
                st.success(f"✅ '{test_question}' is classified as a SIMPLE question")
                st.info("This question SHOULD trigger LangGraph (if enabled)")
            else:
                st.warning(f"⚠️ '{test_question}' is classified as a COMPLEX question")
                st.info("This question will NOT trigger LangGraph (if SIMPLE_QUESTIONS_ONLY=true)")
                
            # Show pattern matching
            simple_patterns = [
                'média', 'mean', 'average',
                'soma', 'sum', 'total', 
                'contar', 'count', 'quantos',
                'máximo', 'max', 'maior',
                'mínimo', 'min', 'menor'
            ]
            
            question_lower = test_question.lower()
            matching_patterns = [pattern for pattern in simple_patterns if pattern in question_lower]
            
            if matching_patterns:
                st.write(f"**Matching patterns:** {matching_patterns}")
            else:
                st.write("**No matching patterns found**")
                
        except Exception as e:
            st.error(f"Error testing question: {e}")
    
    # Final decision logic
    st.subheader("🎯 LangGraph Decision Logic")
    
    try:
        should_try_langgraph = (
            Config.ENABLE_LANGGRAPH and 
            app.LANGGRAPH_INTEGRATION_AVAILABLE and
            (not Config.LANGGRAPH_SIMPLE_QUESTIONS_ONLY or is_simple_question(test_question))
        )
        
        if should_try_langgraph:
            st.success("🚀 LangGraph SHOULD be used for this question!")
        else:
            st.error("🔧 Current system will be used (LangGraph NOT triggered)")
            
        # Show logic breakdown
        st.write("**Logic breakdown:**")
        st.write(f"- ENABLE_LANGGRAPH: {Config.ENABLE_LANGGRAPH}")
        st.write(f"- LANGGRAPH_AVAILABLE: {app.LANGGRAPH_INTEGRATION_AVAILABLE}")
        st.write(f"- SIMPLE_QUESTIONS_ONLY: {Config.LANGGRAPH_SIMPLE_QUESTIONS_ONLY}")
        st.write(f"- Is simple question: {is_simple_question(test_question) if test_question else 'N/A'}")
        
    except Exception as e:
        st.error(f"Error in decision logic: {e}")
    
    # Instructions
    st.subheader("📋 Troubleshooting Instructions")
    st.markdown("""
    **If LangGraph is not working in the main app:**
    
    1. **Stop Streamlit** (Ctrl+C in terminal)
    2. **Verify .env file** has correct settings (shown above)
    3. **Restart Streamlit**: `streamlit run app.py`
    4. **Check logs** for LangGraph execution messages
    
    **Expected log messages when LangGraph is working:**
    ```
    🎯 EXECUTION PATH: LangGraph
    🚀 STARTING LANGGRAPH EXECUTION
    🔄 LANGGRAPH WORKFLOW: Processing question...
    ```
    """)

if __name__ == "__main__":
    main() 