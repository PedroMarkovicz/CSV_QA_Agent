"""
LangGraph Workflow Definition - Phase 1: Pure Wrapper Workflow

This file defines the LangGraph workflow that orchestrates the existing agents
through the wrapper nodes. The workflow follows the exact same logic flow
as the current manual orchestration in app.py.

This is a zero-risk migration step that maintains 100% compatibility.
"""

import logging
from typing import Dict, Any
import time
import json

# LangGraph imports (only install if ENABLE_LANGGRAPH=true)
try:
    from langgraph.graph import StateGraph, END
    from .langgraph_nodes import (
        CSVQAState, 
        understand_question_node, 
        execute_code_node, 
        format_response_node,
        determine_next_step
    )
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("LangGraph not available - install with: pip install langgraph")

logger = logging.getLogger(__name__)

def create_csv_qa_workflow():
    """
    Create the CSV Q&A workflow using LangGraph.
    
    This workflow replicates the exact same logic flow as the current
    answer_question function, just using LangGraph's framework.
    
    Returns:
        Compiled LangGraph workflow
    """
    logger.info("🏗️  LANGGRAPH WORKFLOW: Creating workflow graph...")
    workflow_start_time = time.time()
    
    if not LANGGRAPH_AVAILABLE:
        logger.error("❌ LANGGRAPH WORKFLOW: LangGraph not available")
        raise ImportError("LangGraph not available. Install with: pip install langgraph")
    
    try:
        # Create the state graph
        logger.debug("   📊 Creating StateGraph with CSVQAState schema")
        workflow = StateGraph(CSVQAState)
        
        # Add nodes (these are just wrappers around existing agents)
        logger.debug("   🧩 Adding workflow nodes...")
        workflow.add_node("understand", understand_question_node)
        logger.debug("     ✅ Added 'understand' node (QuestionUnderstandingAgent wrapper)")
        
        workflow.add_node("execute", execute_code_node)
        logger.debug("     ✅ Added 'execute' node (QueryExecutorAgent wrapper)")
        
        workflow.add_node("format", format_response_node)
        logger.debug("     ✅ Added 'format' node (AnswerFormatterAgent wrapper)")
        
        # Set entry point (replaces the manual function call)
        workflow.set_entry_point("understand")
        logger.debug("   🚪 Set entry point: 'understand' node")
        
        # Add edges that replicate current logic flow exactly
        logger.debug("   🔗 Adding conditional edges...")
        workflow.add_conditional_edges(
            "understand",
            determine_next_step,
            {
                "execute": "execute",
                "END": END
            }
        )
        logger.debug("     ✅ understand -> [execute|END] based on determine_next_step")
        
        workflow.add_conditional_edges(
            "execute", 
            determine_next_step,
            {
                "format": "format",
                "END": END
            }
        )
        logger.debug("     ✅ execute -> [format|END] based on determine_next_step")
        
        workflow.add_conditional_edges(
            "format",
            determine_next_step,
            {
                "END": END
            }
        )
        logger.debug("     ✅ format -> END based on determine_next_step")
        
        # Compile the workflow
        logger.debug("   ⚙️  Compiling workflow...")
        compiled_workflow = workflow.compile()
        
        workflow_creation_time = time.time() - workflow_start_time
        logger.info(f"✅ LANGGRAPH WORKFLOW: Workflow created successfully in {workflow_creation_time:.3f}s")
        logger.debug(f"   📊 Workflow structure: understand -> execute -> format -> END")
        logger.debug(f"   🔧 Compiled workflow ready for execution")
        
        return compiled_workflow
        
    except Exception as e:
        workflow_creation_time = time.time() - workflow_start_time
        logger.error(f"❌ LANGGRAPH WORKFLOW: Failed to create workflow after {workflow_creation_time:.3f}s")
        logger.error(f"   Exception: {type(e).__name__}: {str(e)}")
        raise

def answer_question_langgraph(question: str, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph version of answer_question that produces identical results.
    
    This function is a drop-in replacement for the current answer_question function.
    It should produce exactly the same results, just using LangGraph internally.
    
    Args:
        question (str): User's natural language question (same as current)
        analysis_results (dict): Results from file analysis (same as current)
        
    Returns:
        dict: Response in exact same format as current system
        
    The return format matches current system exactly:
    {
        'success': bool,
        'understanding': dict (optional),
        'execution': dict (optional), 
        'formatted_response': dict (optional),
        'answer': str
    }
    """
    execution_start_time = time.time()
    logger.info(f"🔄 LANGGRAPH WORKFLOW: Processing question: {question}")
    logger.debug(f"   📝 Question length: {len(question)} characters")
    logger.debug(f"   📊 Analysis results keys: {list(analysis_results.keys())}")
    logger.debug(f"   🕐 Execution started at: {time.strftime('%H:%M:%S')}")
    
    # Log input validation
    if not question or not question.strip():
        logger.warning("⚠️  LANGGRAPH WORKFLOW: Empty or whitespace-only question received")
    
    if not analysis_results:
        logger.warning("⚠️  LANGGRAPH WORKFLOW: No analysis results provided")
    else:
        logger.debug(f"   📊 Analysis results details:")
        for filename, result in analysis_results.items():
            has_dataframe = hasattr(result, 'dataframe') and result.dataframe is not None
            is_successful = hasattr(result, 'success') and result.success
            logger.debug(f"     - {filename}: success={is_successful}, has_dataframe={has_dataframe}")
            if has_dataframe and hasattr(result.dataframe, 'shape'):
                logger.debug(f"       DataFrame shape: {result.dataframe.shape}")
    
    try:
        # Create the workflow
        logger.info(f"🏗️  LANGGRAPH: Creating workflow graph...")
        workflow_creation_start = time.time()
        workflow = create_csv_qa_workflow()
        workflow_creation_time = time.time() - workflow_creation_start
        logger.info(f"✅ LANGGRAPH: Workflow created successfully in {workflow_creation_time:.3f}s")
        
        # Initialize state exactly as current system expects
        logger.info(f"🔧 LANGGRAPH: Initializing workflow state...")
        initial_state = CSVQAState(
            question=question,
            analysis_results=analysis_results,
            understanding_result=None,
            execution_result=None,
            formatted_response=None,
            success=False,
            answer="",
            error=None,
            processing_step="start",
            errors=[],
            start_time=execution_start_time
        )
        
        logger.debug(f"   📊 Initial state created:")
        logger.debug(f"     - question: {question[:50]}{'...' if len(question) > 50 else ''}")
        logger.debug(f"     - analysis_results count: {len(analysis_results)}")
        logger.debug(f"     - processing_step: {initial_state['processing_step']}")
        logger.debug(f"     - start_time: {initial_state['start_time']}")
        
        logger.info(f"🚀 LANGGRAPH: Starting workflow execution...")
        logger.info(f"   📊 Input DataFrames: {list(analysis_results.keys())}")
        logger.info(f"   🎯 Question: {question}")
        logger.info(f"   🔄 Initial processing step: {initial_state['processing_step']}")
        
        # Run the workflow with detailed state tracking
        workflow_execution_start = time.time()
        logger.debug(f"   ⚡ Invoking workflow with initial state...")
        
        try:
            final_state = workflow.invoke(initial_state)
            workflow_execution_time = time.time() - workflow_execution_start
            
            logger.info(f"🏁 LANGGRAPH: Workflow execution completed in {workflow_execution_time:.3f}s")
            logger.debug(f"   📊 Final state received:")
            logger.debug(f"     - processing_step: {final_state.get('processing_step', 'unknown')}")
            logger.debug(f"     - success: {final_state.get('success', False)}")
            logger.debug(f"     - has_understanding_result: {final_state.get('understanding_result') is not None}")
            logger.debug(f"     - has_execution_result: {final_state.get('execution_result') is not None}")
            logger.debug(f"     - has_formatted_response: {final_state.get('formatted_response') is not None}")
            logger.debug(f"     - answer_length: {len(final_state.get('answer', ''))}")
            logger.debug(f"     - error: {final_state.get('error', 'None')}")
            
        except Exception as workflow_error:
            workflow_execution_time = time.time() - workflow_execution_start
            logger.error(f"❌ LANGGRAPH: Workflow execution failed after {workflow_execution_time:.3f}s")
            logger.error(f"   Exception: {type(workflow_error).__name__}: {str(workflow_error)}")
            
            # Create error state to continue with error handling
            final_state = {
                **initial_state,
                'success': False,
                'error': str(workflow_error),
                'answer': f'Erro na execução do workflow: {str(workflow_error)}',
                'processing_step': 'workflow_error'
            }
        
        logger.info(f"📊 LANGGRAPH: Final processing step: {final_state.get('processing_step', 'unknown')}")
        logger.info(f"✅ LANGGRAPH: Success: {final_state['success']}")
        
        if final_state['success']:
            answer_preview = final_state['answer'][:100] + ('...' if len(final_state['answer']) > 100 else '')
            logger.info(f"💬 LANGGRAPH: Answer generated: {answer_preview}")
            
            # Log detailed success metrics
            logger.debug(f"   📊 Success metrics:")
            if final_state.get('understanding_result'):
                understanding = final_state['understanding_result']
                logger.debug(f"     - Understanding confidence: {understanding.get('confidence', 0):.2f}")
                logger.debug(f"     - Code source: {understanding.get('code_source', 'unknown')}")
                logger.debug(f"     - Operations: {len(understanding.get('operations', []))}")
            
            if final_state.get('execution_result'):
                execution = final_state['execution_result']
                logger.debug(f"     - Execution time: {execution.get('execution_time', 0):.3f}s")
                logger.debug(f"     - Fallback used: {execution.get('fallback_executed', False)}")
                
            if final_state.get('formatted_response'):
                formatting = final_state['formatted_response']
                logger.debug(f"     - Visualizations: {len(formatting.get('visualizations', []))}")
                logger.debug(f"     - Data insights: {len(formatting.get('data_insights', []))}")
                
        else:
            error_msg = final_state.get('error', 'Unknown error')
            logger.warning(f"❌ LANGGRAPH: Error: {error_msg}")
            
            # Log detailed error information
            logger.debug(f"   🔍 Error analysis:")
            logger.debug(f"     - Processing step when failed: {final_state.get('processing_step', 'unknown')}")
            logger.debug(f"     - Error message: {error_msg}")
            logger.debug(f"     - Has understanding result: {final_state.get('understanding_result') is not None}")
            logger.debug(f"     - Has execution result: {final_state.get('execution_result') is not None}")
        
        # Convert LangGraph state back to current system format
        logger.debug(f"🔄 LANGGRAPH: Converting state to current system format...")
        result = {
            'success': final_state['success'],
            'understanding': final_state.get('understanding_result'),
            'execution': final_state.get('execution_result'),
            'formatted_response': final_state.get('formatted_response'),
            'answer': final_state['answer'],
            'error': final_state.get('error')
        }
        
        # Log conversion details
        logger.debug(f"   📊 Conversion results:")
        logger.debug(f"     - Result keys: {list(result.keys())}")
        logger.debug(f"     - Non-null values: {[k for k, v in result.items() if v is not None]}")
        
        total_execution_time = time.time() - execution_start_time
        logger.info(f"🔄 LANGGRAPH: Converted to current system format")
        logger.info(f"⏱️  LANGGRAPH: Total execution time: {total_execution_time:.3f}s")
        logger.info(f"   🏗️  Workflow creation: {workflow_creation_time:.3f}s ({(workflow_creation_time/total_execution_time*100):.1f}%)")
        logger.info(f"   ⚡ Workflow execution: {workflow_execution_time:.3f}s ({(workflow_execution_time/total_execution_time*100):.1f}%)")
        logger.info(f"   🔧 Overhead/conversion: {(total_execution_time-workflow_creation_time-workflow_execution_time):.3f}s")
        
        return result
        
    except Exception as e:
        # Fallback error handling (same as current system)
        total_execution_time = time.time() - execution_start_time
        logger.error(f"❌ LANGGRAPH WORKFLOW FAILED after {total_execution_time:.3f}s: {str(e)}")
        logger.error(f"   Exception type: {type(e).__name__}")
        logger.error(f"   Exception details: {str(e)}")
        
        # Log the full traceback for debugging
        import traceback
        error_traceback = traceback.format_exc()
        logger.debug(f"   🔍 Full traceback:")
        for line in error_traceback.split('\n'):
            if line.strip():
                logger.debug(f"     {line}")
        
        # Create detailed error response
        error_result = {
            'success': False,
            'error': str(e),
            'answer': f'Erro interno: {str(e)}',
            'execution_time': total_execution_time,
            'error_type': type(e).__name__
        }
        
        logger.debug(f"   📊 Error result: {error_result}")
        return error_result

def get_workflow_visualization():
    """
    Get a visual representation of the workflow for debugging/documentation.
    
    Returns:
        str: Mermaid diagram of the workflow
    """
    logger.info("🎨 LANGGRAPH: Generating workflow visualization...")
    
    if not LANGGRAPH_AVAILABLE:
        logger.warning("⚠️  LANGGRAPH: Cannot generate visualization - LangGraph not available")
        return "LangGraph not available"
    
    try:
        visualization_start = time.time()
        workflow = create_csv_qa_workflow()
        
        logger.debug("   🔍 Extracting workflow graph...")
        graph = workflow.get_graph()
        
        logger.debug("   🎨 Generating Mermaid diagram...")
        mermaid_diagram = graph.draw_mermaid()
        
        visualization_time = time.time() - visualization_start
        logger.info(f"✅ LANGGRAPH: Visualization generated in {visualization_time:.3f}s")
        logger.debug(f"   📊 Diagram length: {len(mermaid_diagram)} characters")
        
        return mermaid_diagram
        
    except Exception as e:
        visualization_time = time.time() - visualization_start if 'visualization_start' in locals() else 0
        logger.error(f"❌ LANGGRAPH: Visualization failed after {visualization_time:.3f}s: {str(e)}")
        logger.error(f"   Exception type: {type(e).__name__}")
        
        return f"Visualization failed: {str(e)}"

def log_workflow_metrics(start_time: float, node_times: Dict[str, float], final_state: Dict[str, Any]):
    """
    Log detailed workflow performance metrics.
    
    Args:
        start_time: Workflow start timestamp
        node_times: Dictionary of node execution times
        final_state: Final workflow state
    """
    total_time = time.time() - start_time
    
    logger.info("📊 LANGGRAPH: Detailed Performance Metrics")
    logger.info(f"   ⏱️  Total workflow time: {total_time:.3f}s")
    
    if node_times:
        logger.info(f"   🧩 Node execution times:")
        for node_name, node_time in node_times.items():
            percentage = (node_time / total_time * 100) if total_time > 0 else 0
            logger.info(f"     - {node_name}: {node_time:.3f}s ({percentage:.1f}%)")
    
    logger.info(f"   📊 Final state summary:")
    logger.info(f"     - Success: {final_state.get('success', False)}")
    logger.info(f"     - Processing step: {final_state.get('processing_step', 'unknown')}")
    logger.info(f"     - Answer length: {len(final_state.get('answer', ''))}")
    logger.info(f"     - Has error: {final_state.get('error') is not None}")
    
    # Log memory and resource usage if available
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        logger.debug(f"   💾 Resource usage:")
        logger.debug(f"     - Memory RSS: {memory_info.rss / 1024 / 1024:.1f} MB")
        logger.debug(f"     - Memory VMS: {memory_info.vms / 1024 / 1024:.1f} MB")
        logger.debug(f"     - CPU percent: {process.cpu_percent():.1f}%")
    except ImportError:
        logger.debug(f"   💾 Resource monitoring not available (psutil not installed)")
    except Exception as e:
        logger.debug(f"   💾 Resource monitoring failed: {str(e)}") 