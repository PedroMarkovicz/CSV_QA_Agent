"""
LangGraph Wrapper Nodes - Phase 1: Pure Wrappers

This file wraps existing agents in LangGraph nodes without changing any logic.
The existing agents are called exactly as they were before, just through LangGraph's framework.

This is a zero-risk migration step that maintains 100% compatibility with current code.
"""

from typing import Dict, Any, List, Optional, TypedDict
import logging
import time
import json

# Import existing agents - no changes to them
from agents.csv_loader import CSVLoaderAgent
from agents.question_understanding import QuestionUnderstandingAgent
from agents.query_executor import QueryExecutorAgent
from agents.answer_formatter import AnswerFormatterAgent

logger = logging.getLogger(__name__)
print("🔧 DEBUG MODE ENABLED - Detailed logging active")
print("=" * 60)

logging.basicConfig(
    level=logging.DEBUG,  # 👈 Necessário para permitir mensagens DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class CSVQAState(TypedDict):
    """State object that flows between LangGraph nodes"""
    # Input
    question: str
    analysis_results: Dict[str, Any]
    
    # Intermediate results (exactly matching current system)
    understanding_result: Optional[Dict[str, Any]]
    execution_result: Optional[Dict[str, Any]]
    formatted_response: Optional[Dict[str, Any]]
    
    # Final output (exactly matching current return format)
    success: bool
    answer: str
    error: Optional[str]
    
    # Metadata
    processing_step: str
    errors: List[str]
    start_time: float

def log_node_entry(node_name: str, state: CSVQAState):
    """Log detailed information when entering a node"""
    logger.info(f"🧩 LANGGRAPH NODE ENTRY: {node_name}")
    logger.debug(f"   📊 State validation:")
    logger.debug(f"     - Current step: {state.get('processing_step', 'unknown')}")
    logger.debug(f"     - Question: {state.get('question', 'N/A')[:50]}{'...' if len(state.get('question', '')) > 50 else ''}")
    logger.debug(f"     - Has analysis_results: {bool(state.get('analysis_results'))}")
    logger.debug(f"     - Analysis results count: {len(state.get('analysis_results', {}))}")
    logger.debug(f"     - Current success status: {state.get('success', False)}")
    logger.debug(f"     - Has existing error: {bool(state.get('error'))}")
    
    # Log state completeness
    required_fields = ['question', 'analysis_results', 'processing_step']
    missing_fields = [field for field in required_fields if not state.get(field)]
    if missing_fields:
        logger.warning(f"   ⚠️  Missing required fields: {missing_fields}")
    
    # Log timing information
    if state.get('start_time'):
        elapsed = time.time() - state['start_time']
        logger.debug(f"   ⏱️  Time since workflow start: {elapsed:.3f}s")

def log_node_exit(node_name: str, state: CSVQAState, execution_time: float):
    """Log detailed information when exiting a node"""
    logger.info(f"🏁 LANGGRAPH NODE EXIT: {node_name} (took {execution_time:.3f}s)")
    logger.debug(f"   📊 State after processing:")
    logger.debug(f"     - New processing step: {state.get('processing_step', 'unknown')}")
    logger.debug(f"     - Success status: {state.get('success', False)}")
    logger.debug(f"     - Has error: {bool(state.get('error'))}")
    logger.debug(f"     - Answer length: {len(state.get('answer', ''))}")
    logger.debug(f"     - Has understanding_result: {bool(state.get('understanding_result'))}")
    logger.debug(f"     - Has execution_result: {bool(state.get('execution_result'))}")
    logger.debug(f"     - Has formatted_response: {bool(state.get('formatted_response'))}")
    print("🔧 DEBUG MODE ENABLED - Detailed logging active")
    print("=" * 90)
    
    # Log performance metrics
    if state.get('start_time'):
        total_elapsed = time.time() - state['start_time']
        node_percentage = (execution_time / total_elapsed * 100) if total_elapsed > 0 else 0
        logger.debug(f"   ⚡ Performance:")
        logger.debug(f"     - Node execution time: {execution_time:.3f}s")
        logger.debug(f"     - Total workflow time: {total_elapsed:.3f}s")
        logger.debug(f"     - Node time percentage: {node_percentage:.1f}%")

def validate_dataframes(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and extract DataFrames from analysis results with detailed logging"""
    logger.debug("   🔍 DATAFRAME VALIDATION: Starting validation...")
    
    dataframes = {}
    validation_summary = {
        'total_files': len(analysis_results),
        'valid_files': 0,
        'invalid_files': 0,
        'total_rows': 0,
        'total_columns': 0,
        'file_details': []
    }
    
    for filename, result in analysis_results.items():
        file_detail = {
            'filename': filename,
            'success': False,
            'has_dataframe': False,
            'rows': 0,
            'columns': 0,
            'error': None
        }
        
        try:
            # Check if result has success attribute and is successful
            if hasattr(result, 'success') and result.success:
                file_detail['success'] = True
                
                # Check if result has a valid dataframe
                if hasattr(result, 'dataframe') and result.dataframe is not None:
                    df = result.dataframe
                    file_detail['has_dataframe'] = True
                    file_detail['rows'] = len(df)
                    file_detail['columns'] = len(df.columns)
                    
                    # Store the DataFrame
                    dataframes[filename] = df
                    validation_summary['valid_files'] += 1
                    validation_summary['total_rows'] += file_detail['rows']
                    validation_summary['total_columns'] += file_detail['columns']
                    
                    logger.debug(f"     ✅ {filename}: {file_detail['rows']} rows, {file_detail['columns']} columns")
                else:
                    file_detail['error'] = "No dataframe attribute or dataframe is None"
                    validation_summary['invalid_files'] += 1
                    logger.warning(f"     ❌ {filename}: No valid dataframe")
            else:
                file_detail['error'] = "Result not successful or missing success attribute"
                validation_summary['invalid_files'] += 1
                logger.warning(f"     ❌ {filename}: Analysis not successful")
                
        except Exception as e:
            file_detail['error'] = str(e)
            validation_summary['invalid_files'] += 1
            logger.error(f"     ❌ {filename}: Validation error - {str(e)}")
        
        validation_summary['file_details'].append(file_detail)
    
    # Log validation summary
    logger.debug(f"   📊 DATAFRAME VALIDATION SUMMARY:")
    logger.debug(f"     - Total files: {validation_summary['total_files']}")
    logger.debug(f"     - Valid files: {validation_summary['valid_files']}")
    logger.debug(f"     - Invalid files: {validation_summary['invalid_files']}")
    logger.debug(f"     - Total rows across all files: {validation_summary['total_rows']:,}")
    logger.debug(f"     - Average columns per file: {validation_summary['total_columns'] / max(validation_summary['valid_files'], 1):.1f}")
    
    logger.info(f"   📊 Extracted {len(dataframes)} valid DataFrames from {len(analysis_results)} files")
    
    return dataframes, validation_summary

def understand_question_node(state: CSVQAState) -> CSVQAState:
    """
    Wrapper around existing QuestionUnderstandingAgent.
    
    This node calls the existing agent exactly as the current system does,
    no logic changes whatsoever.
    """
    node_start_time = time.time()
    log_node_entry("QuestionUnderstanding", state)
    
    try:
        # Extract DataFrames exactly as current system does
        logger.debug("   🔧 UNDERSTANDING: Extracting DataFrames...")
        dataframes, validation_summary = validate_dataframes(state["analysis_results"])
        
        # Validate exactly as current system does
        if not dataframes:
            logger.warning("   ❌ UNDERSTANDING: No valid DataFrames available")
            logger.debug(f"   📊 Validation details: {validation_summary}")
            
            result_state = {
                **state,
                "success": False,
                "error": "No valid DataFrames available for analysis.",
                "answer": "Nenhum dado válido disponível para análise.",
                "processing_step": "understanding_failed"
            }
            
            execution_time = time.time() - node_start_time
            log_node_exit("QuestionUnderstanding", result_state, execution_time)
            return result_state
        
        # Log question analysis
        question = state["question"]
        logger.debug(f"   📝 UNDERSTANDING: Question analysis:")
        logger.debug(f"     - Question: '{question}'")
        logger.debug(f"     - Question length: {len(question)} characters")
        logger.debug(f"     - Question words: {len(question.split())} words")
        logger.debug(f"     - Available DataFrames: {list(dataframes.keys())}")
        
        # Call existing agent exactly as before
        logger.info("   🔄 UNDERSTANDING: Calling QuestionUnderstandingAgent...")
        agent_start_time = time.time()
        
        question_agent = QuestionUnderstandingAgent()
        understanding_result = question_agent.understand_question(question, dataframes)
        
        agent_execution_time = time.time() - agent_start_time
        logger.debug(f"   ⚡ UNDERSTANDING: Agent executed in {agent_execution_time:.3f}s")
        
        # Log detailed understanding results
        logger.info(f"   📋 UNDERSTANDING: Results analysis:")
        logger.info(f"     - Code source: {understanding_result.get('code_source', 'unknown')}")
        logger.info(f"     - Confidence: {understanding_result.get('confidence', 0):.2f}")
        logger.info(f"     - Generated code: {'Yes' if understanding_result.get('generated_code') else 'No'}")
        
        # Log additional details
        logger.debug(f"   🔍 UNDERSTANDING: Detailed results:")
        logger.debug(f"     - Operations detected: {len(understanding_result.get('operations', []))}")
        logger.debug(f"     - Target columns: {understanding_result.get('target_columns', [])}")
        logger.debug(f"     - Explanation: {understanding_result.get('explanation', 'N/A')}")
        
        if understanding_result.get('generated_code'):
            code_preview = understanding_result['generated_code'][:100] + ('...' if len(understanding_result['generated_code']) > 100 else '')
            logger.debug(f"     - Code preview: {code_preview}")
            logger.debug(f"     - Code length: {len(understanding_result['generated_code'])} characters")
        
        # Check for errors exactly as current system does
        if understanding_result.get('error'):
            logger.warning(f"   ❌ UNDERSTANDING: Agent failed: {understanding_result['error']}")
            
            result_state = {
                **state,
                "understanding_result": understanding_result,
                "success": False,
                "error": understanding_result['error'],
                "answer": f"Erro ao entender a pergunta: {understanding_result['explanation']}",
                "processing_step": "understanding_failed"
            }
        else:
            # Success - store result and continue
            logger.info("   ✅ UNDERSTANDING: Agent completed successfully, proceeding to execution")
            
            result_state = {
                **state,
                "understanding_result": understanding_result,
                "processing_step": "understanding_complete"
            }
        
        execution_time = time.time() - node_start_time
        log_node_exit("QuestionUnderstanding", result_state, execution_time)
        return result_state
        
    except Exception as e:
        execution_time = time.time() - node_start_time
        logger.error(f"   ❌ UNDERSTANDING: Node exception after {execution_time:.3f}s: {str(e)}")
        
        # Log full traceback for debugging
        import traceback
        traceback_str = traceback.format_exc()
        logger.debug(f"   🔍 UNDERSTANDING: Full traceback:")
        for line in traceback_str.split('\n'):
            if line.strip():
                logger.debug(f"     {line}")
        
        result_state = {
            **state,
            "success": False,
            "error": str(e),
            "answer": f"Erro interno: {str(e)}",
            "processing_step": "understanding_error"
        }
        
        log_node_exit("QuestionUnderstanding", result_state, execution_time)
        return result_state

def execute_code_node(state: CSVQAState) -> CSVQAState:
    """
    Wrapper around existing QueryExecutorAgent.
    
    Calls the existing executor exactly as the current system does.
    """
    node_start_time = time.time()
    log_node_entry("CodeExecution", state)
    
    try:
        understanding_result = state.get("understanding_result")
        
        # Validate understanding result
        if not understanding_result:
            logger.error("   ❌ EXECUTION: No understanding result available")
            
            result_state = {
                **state,
                "success": False,
                "error": "No understanding result available",
                "answer": "Erro interno: resultado do entendimento não disponível.",
                "processing_step": "execution_failed"
            }
            
            execution_time = time.time() - node_start_time
            log_node_exit("CodeExecution", result_state, execution_time)
            return result_state
        
        # Check if understanding_result is a string (error case) instead of dict
        if isinstance(understanding_result, str):
            logger.error(f"   ❌ EXECUTION: Understanding result is string (error): {understanding_result}")
            
            result_state = {
                **state,
                "success": False,
                "error": f"Understanding result is error string: {understanding_result}",
                "answer": f"Erro no entendimento: {understanding_result}",
                "processing_step": "execution_failed"
            }
            
            execution_time = time.time() - node_start_time
            log_node_exit("CodeExecution", result_state, execution_time)
            return result_state
        
        # Additional validation - ensure it's a dictionary
        if not isinstance(understanding_result, dict):
            logger.error(f"   ❌ EXECUTION: Understanding result is not a dict: {type(understanding_result)}")
            
            result_state = {
                **state,
                "success": False,
                "error": f"Understanding result has wrong type: {type(understanding_result)}",
                "answer": "Erro interno: formato inválido do resultado do entendimento.",
                "processing_step": "execution_failed"
            }
            
            execution_time = time.time() - node_start_time
            log_node_exit("CodeExecution", result_state, execution_time)
            return result_state
        
        # Log code analysis with safe access
        generated_code = understanding_result.get('generated_code', '')
        logger.debug(f"   📝 EXECUTION: Code analysis:")
        logger.debug(f"     - Has generated code: {bool(generated_code)}")
        if generated_code:
            logger.debug(f"     - Code length: {len(generated_code)} characters")
            num_lines = len(generated_code.split('\n'))
            logger.debug(f"     - Code lines: {num_lines}")
            logger.debug(f"     - Code preview: {generated_code[:100]}{'...' if len(generated_code) > 100 else ''}")
        
        # Check for generated code exactly as current system does
        if not generated_code:
            logger.warning("   ❌ EXECUTION: No code generated to execute")
            
            result_state = {
                **state,
                "success": False,
                "error": "No code generated",
                "answer": "Não foi possível gerar código para esta pergunta. Tente reformular de forma mais específica.",
                "processing_step": "execution_failed"
            }
            
            execution_time = time.time() - node_start_time
            log_node_exit("CodeExecution", result_state, execution_time)
            return result_state
        
        # Extract DataFrames exactly as current system does
        logger.debug("   🔧 EXECUTION: Re-extracting DataFrames for execution...")
        dataframes, validation_summary = validate_dataframes(state["analysis_results"])
        
        if not dataframes:
            logger.error("   ❌ EXECUTION: No valid DataFrames available for execution")
            
            result_state = {
                **state,
                "success": False,
                "error": "No valid DataFrames available for execution",
                "answer": "Erro interno: dados não disponíveis para execução.",
                "processing_step": "execution_failed"
            }
            
            execution_time = time.time() - node_start_time
            log_node_exit("CodeExecution", result_state, execution_time)
            return result_state
        
        logger.info(f"   📊 EXECUTION: Executing code with {len(dataframes)} DataFrames")
        
        # Log execution environment
        logger.debug(f"   🌍 EXECUTION: Environment details:")
        logger.debug(f"     - Available DataFrames: {list(dataframes.keys())}")
        for df_name, df in dataframes.items():
            logger.debug(f"     - {df_name}: {df.shape} ({df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB)")
        
        # Call existing agent exactly as before
        logger.info("   🔄 EXECUTION: Calling QueryExecutorAgent...")
        agent_start_time = time.time()
        
        executor_agent = QueryExecutorAgent()
        execution_result = executor_agent.execute_code(generated_code, dataframes)
        
        agent_execution_time = time.time() - agent_start_time
        logger.debug(f"   ⚡ EXECUTION: Agent executed in {agent_execution_time:.3f}s")
        
        # Log detailed execution results
        logger.info(f"   📋 EXECUTION: Results analysis:")
        logger.info(f"     - Success: {execution_result.get('success')}")
        logger.info(f"     - Execution time: {execution_result.get('execution_time', 0):.3f}s")
        logger.info(f"     - Fallback used: {execution_result.get('fallback_executed', False)}")
        
        # Log additional execution details
        logger.debug(f"   🔍 EXECUTION: Detailed results:")
        logger.debug(f"     - Result type: {type(execution_result.get('result', 'N/A'))}")
        logger.debug(f"     - Variables created: {execution_result.get('variables_created', [])}")
        logger.debug(f"     - Security validations: {execution_result.get('security_validations', [])}")
        
        if execution_result.get('result') is not None:
            result_value = execution_result['result']
            logger.debug(f"     - Result value: {str(result_value)[:100]}{'...' if len(str(result_value)) > 100 else ''}")
            logger.debug(f"     - Result size: {len(str(result_value))} characters")
        
        if execution_result.get('error'):
            logger.warning(f"     - Error: {execution_result['error']}")
            logger.debug(f"     - Error type: {execution_result.get('error_type', 'Unknown')}")
        
        if execution_result.get('fallback_executed'):
            logger.debug(f"     - Fallback strategy: {execution_result.get('fallback_strategy', 'Unknown')}")
            logger.debug(f"     - Fallback reason: {execution_result.get('fallback_reason', 'Unknown')}")
        
        # Store result and continue
        result_state = {
            **state,
            "execution_result": execution_result,
            "processing_step": "execution_complete"
        }
        
        logger.info("   ✅ EXECUTION: Completed successfully, proceeding to formatting")
        
        execution_time = time.time() - node_start_time
        log_node_exit("CodeExecution", result_state, execution_time)
        return result_state
        
    except Exception as e:
        execution_time = time.time() - node_start_time
        logger.error(f"   ❌ EXECUTION: Node exception after {execution_time:.3f}s: {str(e)}")
        
        # Log full traceback for debugging
        import traceback
        traceback_str = traceback.format_exc()
        logger.debug(f"   🔍 EXECUTION: Full traceback:")
        for line in traceback_str.split('\n'):
            if line.strip():
                logger.debug(f"     {line}")
        
        result_state = {
            **state,
            "success": False,
            "error": str(e),
            "answer": f"Erro interno: {str(e)}",
            "processing_step": "execution_error"
        }
        
        log_node_exit("CodeExecution", result_state, execution_time)
        return result_state

def format_response_node(state: CSVQAState) -> CSVQAState:
    """
    Wrapper around existing AnswerFormatterAgent.
    
    Calls the existing formatter exactly as the current system does.
    """
    node_start_time = time.time()
    log_node_entry("ResponseFormatting", state)
    
    try:
        understanding_result = state.get("understanding_result")
        execution_result = state.get("execution_result")
        
        # Validate prerequisites with proper type checking
        if not understanding_result:
            logger.error("   ❌ FORMATTING: No understanding result available")
            
            result_state = {
                **state,
                "success": False,
                "error": "No understanding result for formatting",
                "answer": "Erro interno: resultado do entendimento não disponível para formatação.",
                "processing_step": "formatting_error"
            }
            
            execution_time = time.time() - node_start_time
            log_node_exit("ResponseFormatting", result_state, execution_time)
            return result_state
        
        # Check if understanding_result is a string (error case) instead of dict
        if isinstance(understanding_result, str):
            logger.error(f"   ❌ FORMATTING: Understanding result is string (error): {understanding_result}")
            
            result_state = {
                **state,
                "success": False,
                "error": f"Understanding result is error string: {understanding_result}",
                "answer": f"Erro no entendimento: {understanding_result}",
                "processing_step": "formatting_error"
            }
            
            execution_time = time.time() - node_start_time
            log_node_exit("ResponseFormatting", result_state, execution_time)
            return result_state
        
        if not execution_result:
            logger.error("   ❌ FORMATTING: No execution result available")
            
            result_state = {
                **state,
                "success": False,
                "error": "No execution result for formatting",
                "answer": "Erro interno: resultado da execução não disponível para formatação.",
                "processing_step": "formatting_error"
            }
            
            execution_time = time.time() - node_start_time
            log_node_exit("ResponseFormatting", result_state, execution_time)
            return result_state
        
        # Check if execution_result is a string (error case) instead of dict
        if isinstance(execution_result, str):
            logger.error(f"   ❌ FORMATTING: Execution result is string (error): {execution_result}")
            
            result_state = {
                **state,
                "success": False,
                "error": f"Execution result is error string: {execution_result}",
                "answer": f"Erro na execução: {execution_result}",
                "processing_step": "formatting_error"
            }
            
            execution_time = time.time() - node_start_time
            log_node_exit("ResponseFormatting", result_state, execution_time)
            return result_state
        
        # Additional validation - ensure they are dictionaries
        if not isinstance(understanding_result, dict):
            logger.error(f"   ❌ FORMATTING: Understanding result is not a dict: {type(understanding_result)}")
            
            result_state = {
                **state,
                "success": False,
                "error": f"Understanding result has wrong type: {type(understanding_result)}",
                "answer": "Erro interno: formato inválido do resultado do entendimento.",
                "processing_step": "formatting_error"
            }
            
            execution_time = time.time() - node_start_time
            log_node_exit("ResponseFormatting", result_state, execution_time)
            return result_state
        
        if not isinstance(execution_result, dict):
            logger.error(f"   ❌ FORMATTING: Execution result is not a dict: {type(execution_result)}")
            
            result_state = {
                **state,
                "success": False,
                "error": f"Execution result has wrong type: {type(execution_result)}",
                "answer": "Erro interno: formato inválido do resultado da execução.",
                "processing_step": "formatting_error"
            }
            
            execution_time = time.time() - node_start_time
            log_node_exit("ResponseFormatting", result_state, execution_time)
            return result_state
        
        # Log formatting inputs with safe access
        logger.debug(f"   📝 FORMATTING: Input analysis:")
        logger.debug(f"     - Question: {state.get('question', 'N/A')[:50]}{'...' if len(state.get('question', '')) > 50 else ''}")
        logger.debug(f"     - Execution success: {execution_result.get('success', False)}")
        logger.debug(f"     - Has execution result: {execution_result.get('result') is not None}")
        logger.debug(f"     - Understanding confidence: {understanding_result.get('confidence', 0):.2f}")
        logger.debug(f"     - Code source: {understanding_result.get('code_source', 'unknown')}")
        
        # Call existing agent exactly as before
        logger.info("   🔄 FORMATTING: Calling AnswerFormatterAgent...")
        agent_start_time = time.time()
        
        formatter_agent = AnswerFormatterAgent()
        formatted_response = formatter_agent.format_response(
            execution_result, 
            state["question"], 
            understanding_result
        )
        
        agent_execution_time = time.time() - agent_start_time
        logger.debug(f"   ⚡ FORMATTING: Agent executed in {agent_execution_time:.3f}s")
        
        # Validate formatter response
        if not isinstance(formatted_response, dict):
            logger.error(f"   ❌ FORMATTING: Formatter returned non-dict: {type(formatted_response)}")
            
            result_state = {
                **state,
                "success": False,
                "error": f"Formatter returned invalid type: {type(formatted_response)}",
                "answer": "Erro interno: formatação retornou tipo inválido.",
                "processing_step": "formatting_error"
            }
            
            execution_time = time.time() - node_start_time
            log_node_exit("ResponseFormatting", result_state, execution_time)
            return result_state
        
        # Log detailed formatting results with safe access
        logger.info(f"   📋 FORMATTING: Results analysis:")
        logger.info(f"     - Natural language answer generated: {'Yes' if formatted_response.get('natural_language_answer') else 'No'}")
        logger.info(f"     - Visualizations: {len(formatted_response.get('visualizations', []))}")
        logger.info(f"     - Data insights: {len(formatted_response.get('data_insights', []))}")
        
        # Log additional formatting details with safe access
        logger.debug(f"   🔍 FORMATTING: Detailed results:")
        logger.debug(f"     - Response confidence: {formatted_response.get('confidence_score', 0):.2f}")
        logger.debug(f"     - Response type: {formatted_response.get('response_type', 'unknown')}")
        logger.debug(f"     - Language: {formatted_response.get('language', 'unknown')}")
        
        final_answer = formatted_response.get('natural_language_answer', 'Resposta não disponível.')
        logger.debug(f"     - Answer length: {len(final_answer)} characters")
        logger.debug(f"     - Answer words: {len(final_answer.split())} words")
        
        if formatted_response.get('visualizations'):
            visualizations = formatted_response['visualizations']
            try:
                # Safe access to visualization types with type checking
                viz_types = []
                for viz in visualizations:
                    if isinstance(viz, dict):
                        viz_types.append(viz.get('type', 'unknown'))
                    else:
                        viz_types.append(f"non-dict: {type(viz).__name__}")
                logger.debug(f"     - Visualization types: {viz_types}")
            except Exception as e:
                logger.debug(f"     - Visualization types: Error processing - {str(e)}")
        
        if formatted_response.get('data_insights'):
            data_insights = formatted_response['data_insights']
            try:
                # Safe access to insight types with type checking
                insight_types = []
                for insight in data_insights:
                    if isinstance(insight, dict):
                        insight_types.append(insight.get('type', 'unknown'))
                    else:
                        insight_types.append(f"non-dict: {type(insight).__name__}")
                logger.debug(f"     - Insight types: {insight_types}")
            except Exception as e:
                logger.debug(f"     - Insight types: Error processing - {str(e)}")
        
        answer_preview = final_answer[:100] + ('...' if len(final_answer) > 100 else '')
        logger.info(f"   💬 FORMATTING: Final answer: {answer_preview}")
        
        # Return exactly the same format as current system
        result_state = {
            **state,
            "formatted_response": formatted_response,
            "success": True,
            "answer": final_answer,
            "processing_step": "complete"
        }
        
        logger.info("   ✅ FORMATTING: Completed successfully, workflow complete")
        
        execution_time = time.time() - node_start_time
        log_node_exit("ResponseFormatting", result_state, execution_time)
        return result_state
        
    except Exception as e:
        execution_time = time.time() - node_start_time
        logger.error(f"   ❌ FORMATTING: Node exception after {execution_time:.3f}s: {str(e)}")
        
        # Log full traceback for debugging
        import traceback
        traceback_str = traceback.format_exc()
        logger.debug(f"   🔍 FORMATTING: Full traceback:")
        for line in traceback_str.split('\n'):
            if line.strip():
                logger.debug(f"     {line}")
        
        result_state = {
            **state,
            "success": False,
            "error": str(e),
            "answer": f"Erro interno: {str(e)}",
            "processing_step": "formatting_error"
        }
        
        log_node_exit("ResponseFormatting", result_state, execution_time)
        return result_state

def determine_next_step(state: CSVQAState) -> str:
    """
    Routing function that determines the next step in the workflow.
    
    This implements the exact same logic flow as the current system.
    """
    step = state.get("processing_step", "start")
    
    logger.debug(f"🔀 LANGGRAPH ROUTER: Determining next step...")
    logger.debug(f"   📊 Router input analysis:")
    logger.debug(f"     - Current step: '{step}'")
    logger.debug(f"     - State success: {state.get('success', False)}")
    logger.debug(f"     - Has error: {bool(state.get('error'))}")
    logger.debug(f"     - Has understanding result: {bool(state.get('understanding_result'))}")
    logger.debug(f"     - Has execution result: {bool(state.get('execution_result'))}")
    logger.debug(f"     - Has formatted response: {bool(state.get('formatted_response'))}")
    
    if step == "start":
        next_step = "understand"
    elif step == "understanding_complete":
        next_step = "execute" 
    elif step == "execution_complete":
        next_step = "format"
    elif step == "complete":
        next_step = "END"
    else:
        # Any error state goes to END
        next_step = "END"
        logger.debug(f"     - Error state detected: '{step}' -> END")
    
    logger.debug(f"🔀 LANGGRAPH ROUTER: '{step}' -> '{next_step}'")
    
    # Log routing decision rationale
    if next_step == "END":
        total_time = time.time() - state.get('start_time', time.time())
        logger.info(f"🏁 LANGGRAPH ROUTER: Workflow ending after {total_time:.3f}s")
        logger.debug(f"   📊 Final workflow summary:")
        logger.debug(f"     - Final step: {step}")
        logger.debug(f"     - Success: {state.get('success', False)}")
        logger.debug(f"     - Answer length: {len(state.get('answer', ''))}")
    else:
        logger.debug(f"   ➡️  Proceeding to: {next_step}")
    
    return next_step 