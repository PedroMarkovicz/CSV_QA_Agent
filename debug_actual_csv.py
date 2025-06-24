#!/usr/bin/env python3
"""
Debug script to investigate the actual CSV columns during question processing
"""

import streamlit as st
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(__file__))

from agents.question_understanding import QuestionUnderstandingAgent

def debug_actual_question_processing():
    """Debug the actual question processing with the CSV that's loaded"""
    
    # Check if we're in a Streamlit session and have loaded dataframes
    if 'loaded_dataframes' in st.session_state and st.session_state.loaded_dataframes:
        st.write("## 🔍 Debug Information")
        
        st.write("### Loaded DataFrames:")
        for name, df in st.session_state.loaded_dataframes.items():
            st.write(f"**{name}:**")
            st.write(f"- Shape: {df.shape}")
            st.write(f"- Columns: {list(df.columns)}")
            st.write(f"- Sample data:")
            st.dataframe(df.head(3))
            
        st.write("### Question Processing Test:")
        question = "Qual é a soma dos valores?"
        
        agent = QuestionUnderstandingAgent()
        
        # Test the column identification specifically
        for name, df in st.session_state.loaded_dataframes.items():
            st.write(f"**Testing with {name}:**")
            
            clean_question = agent._clean_question(question)
            st.write(f"- Cleaned question: '{clean_question}'")
            
            # Test normalization
            normalized_question = agent._normalize(clean_question)
            st.write(f"- Normalized question: '{normalized_question}'")
            
            # Test column identification step by step
            st.write(f"- Available columns: {list(df.columns)}")
            
            columns_found = []
            normalized_cols = {}
            
            for col in df.columns:
                norm_col = agent._normalize(str(col))
                normalized_cols[col] = norm_col
                st.write(f"  - '{col}' -> '{norm_col}'")
                
                # Check for matches
                if norm_col in normalized_question:
                    columns_found.append(f"{col} (direct match)")
                elif any(word in norm_col for word in normalized_question.split() if len(word) > 3):
                    columns_found.append(f"{col} (partial match)")
                    
            st.write(f"- Columns found: {columns_found}")
            
            # Test the actual method
            identified_columns = agent._identify_columns(clean_question, df)
            st.write(f"- Method result: {identified_columns}")
            
            # Test operations
            operations = agent._identify_operations(clean_question)
            st.write(f"- Operations found: {[op['operation'] for op in operations]}")
    
    else:
        st.write("❌ No loaded dataframes found in session state")

# If this script is being imported into the Streamlit app
if __name__ != "__main__":
    def add_debug_section():
        with st.expander("🔧 Debug Information", expanded=False):
            debug_actual_question_processing()
            
    # Export the function
    __all__ = ['add_debug_section'] 