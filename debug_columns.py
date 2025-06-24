#!/usr/bin/env python3
"""
Debug script to check column identification logic
"""

import os
import sys
import pandas as pd
from typing import Dict

# Add the project root to the path
sys.path.insert(0, os.path.dirname(__file__))

from agents.question_understanding import QuestionUnderstandingAgent

def test_column_identification():
    """Test the column identification logic"""
    
    # Create a test CSV file with typical column names
    print("🔍 Testing Column Identification Logic")
    print("=" * 50)
    
    # Create sample DataFrames with different column name patterns
    test_dataframes = {
        'test1.csv': pd.DataFrame({
            'Valor': [100, 200, 300],
            'Quantidade': [1, 2, 3],
            'Produto': ['A', 'B', 'C']
        }),
        'test2.csv': pd.DataFrame({
            'valor_total': [100.50, 200.75, 300.25],
            'valor_unitario': [50.25, 100.375, 150.125],
            'nome_produto': ['Produto A', 'Produto B', 'Produto C']
        }),
        'test3.csv': pd.DataFrame({
            'Total': [1000, 2000, 3000],
            'Price': [10.99, 20.99, 30.99],
            'Item': ['Item 1', 'Item 2', 'Item 3']
        })
    }
    
    agent = QuestionUnderstandingAgent()
    question = "Qual é a soma dos valores?"
    
    print(f"📝 Question: '{question}'")
    print()
    
    for df_name, df in test_dataframes.items():
        print(f"📊 Testing DataFrame: {df_name}")
        print(f"   Columns: {list(df.columns)}")
        
        # Test the column identification
        clean_question = agent._clean_question(question)
        identified_columns = agent._identify_columns(clean_question, df)
        
        print(f"   Cleaned question: '{clean_question}'")
        print(f"   Identified columns: {identified_columns}")
        
        # Test operations identification
        operations = agent._identify_operations(clean_question)
        print(f"   Identified operations: {[op['operation'] for op in operations]}")
        
        # Test full understanding
        result = agent.understand_question(question, {df_name: df})
        print(f"   Understanding result:")
        print(f"     - Success: {result.get('generated_code') is not None}")
        print(f"     - Code source: {result.get('code_source', 'None')}")
        print(f"     - Confidence: {result.get('confidence', 0):.2f}")
        if result.get('generated_code'):
            print(f"     - Generated code: {result['generated_code']}")
        else:
            print(f"     - Error: {result.get('explanation', 'No explanation')}")
        print()

def test_normalization():
    """Test the text normalization function"""
    print("🔧 Testing Text Normalization")
    print("=" * 30)
    
    agent = QuestionUnderstandingAgent()
    
    test_cases = [
        "Qual é a soma dos valores?",
        "valores",
        "valor",
        "Valor",
        "VALOR",
        "valor_total",
        "Valor_Total"
    ]
    
    for text in test_cases:
        normalized = agent._normalize(text)
        print(f"'{text}' -> '{normalized}'")
    print()

if __name__ == "__main__":
    test_normalization()
    test_column_identification() 