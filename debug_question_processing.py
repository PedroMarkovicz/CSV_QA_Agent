#!/usr/bin/env python3
"""
Debug script to trace question processing and column identification
"""

import os
import sys
import logging

# Add the project root to the path
sys.path.insert(0, os.path.dirname(__file__))

from agents.question_understanding import QuestionUnderstandingAgent

# Enable more detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def debug_question_processing():
    """Debug the question processing with detailed logging"""
    
    print("🔍 Debugging Question Processing")
    print("=" * 50)
    
    # Create the agent
    agent = QuestionUnderstandingAgent()
    
    # Monkey patch the methods to add debug information
    original_identify_columns = agent._identify_columns
    original_clean_question = agent._clean_question
    original_understand_question = agent.understand_question
    
    def debug_identify_columns(self, question, df):
        print(f"🔍 _identify_columns called:")
        print(f"   Question: '{question}'")
        print(f"   DataFrame columns: {list(df.columns)}")
        print(f"   DataFrame shape: {df.shape}")
        
        result = original_identify_columns(question, df)
        print(f"   Result: {result}")
        return result
    
    def debug_clean_question(self, question):
        result = original_clean_question(question)
        print(f"🧹 _clean_question: '{question}' -> '{result}'")
        return result
    
    def debug_understand_question(self, question, dataframes):
        print(f"🧠 understand_question called:")
        print(f"   Question: '{question}'")
        print(f"   DataFrames: {list(dataframes.keys())}")
        for name, df in dataframes.items():
            print(f"     - {name}: shape {df.shape}, columns {list(df.columns)}")
        
        result = original_understand_question(question, dataframes)
        print(f"   Final result: {result}")
        return result
    
    # Apply monkey patches
    agent._identify_columns = lambda question, df: debug_identify_columns(agent, question, df)
    agent._clean_question = lambda question: debug_clean_question(agent, question)
    agent.understand_question = lambda question, dataframes: debug_understand_question(agent, question, dataframes)
    
    return agent

if __name__ == "__main__":
    debug_agent = debug_question_processing()
    print("Debug agent created. You can now import this in your app to trace execution.") 