"""
AnswerFormatterAgent - Formata a resposta para o usuário em linguagem natural com base no resultado da análise
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

class AnswerFormatterAgent:
    """Agent responsible for formatting responses in natural language with visualizations and localization."""
    
    def __init__(self):
        self.formatting_history: List[dict] = []
        self.visualization_templates = self._load_visualization_templates()
        self.language = 'pt-BR'  # Default language

    def set_language(self, lang: str):
        """Set the language for formatting (pt-BR or en-US)."""
        if lang in ['pt-BR', 'en-US']:
            self.language = lang
        else:
            logger.warning(f"Unsupported language set: {lang}")

    def _load_visualization_templates(self) -> Dict[str, dict]:
        """Load templates for different types of visualizations."""
        return {
            'numeric_summary': {
                'chart_type': 'bar',
                'description': 'Resumo estatístico de valores numéricos'
            },
            'categorical_distribution': {
                'chart_type': 'pie',
                'description': 'Distribuição de categorias'
            },
            'time_series': {
                'chart_type': 'line',
                'description': 'Evolução temporal'
            },
            'comparison': {
                'chart_type': 'bar',
                'description': 'Comparação entre grupos'
            },
            'correlation': {
                'chart_type': 'heatmap',
                'description': 'Correlação entre variáveis'
            }
        }

    def format_response(self, execution_result: dict, question: str, 
                       understanding_result: dict = None, language: Optional[str] = None) -> dict:
        """
        Format execution results into a user-friendly response with localization and logging.
        """
        if language:
            self.set_language(language)
        formatted_response = {
            'original_question': question,
            'success': execution_result.get('success', False),
            'natural_language_answer': '',
            'visualizations': [],
            'data_insights': [],
            'recommendations': [],
            'technical_details': {},
            'confidence_score': 0.0
        }
        try:
            if execution_result.get('success'):
                formatted_response = self._format_successful_response(
                    execution_result, question, understanding_result, formatted_response
                )
            else:
                formatted_response = self._format_error_response(
                    execution_result, question, formatted_response
                )
            self.formatting_history.append(formatted_response)
            logger.info(f"Response formatted with confidence {formatted_response['confidence_score']:.2f}")
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error formatting response ({type(e).__name__}): {str(e)}")
            formatted_response.update({
                'natural_language_answer': self._localize('Ocorreu um erro ao formatar a resposta.', 'An error occurred while formatting the response.'),
                'technical_details': {'error': str(e)}
            })
        except Exception as e:
            logger.error(f"Error formatting response: {str(e)}")
            formatted_response.update({
                'natural_language_answer': self._localize('Ocorreu um erro ao formatar a resposta.', 'An error occurred while formatting the response.'),
                'technical_details': {'error': str(e)}
            })
        return formatted_response

    def _format_successful_response(self, execution_result: dict, question: str,
                                  understanding_result: dict, response: dict) -> dict:
        """Format a successful execution result with localization and logging."""
        result_data = execution_result.get('result')
        if result_data is None:
            response['natural_language_answer'] = self._localize(
                "A análise foi executada, mas não produziu resultados visíveis.",
                "The analysis was executed, but did not produce visible results."
            )
            logger.debug("Result is None; fallback response used.")
            return response
        # Determine response type based on result data
        if isinstance(result_data, (int, float)):
            response = self._format_numeric_result(result_data, question, execution_result, response)
        elif isinstance(result_data, pd.Series):
            response = self._format_series_result(result_data, question, execution_result, response)
        elif isinstance(result_data, pd.DataFrame):
            response = self._format_dataframe_result(result_data, question, execution_result, response)
        elif isinstance(result_data, dict):
            response = self._format_dict_result(result_data, question, execution_result, response)
        else:
            response = self._format_generic_result(result_data, question, execution_result, response)
        response['technical_details'] = {
            'execution_time': execution_result.get('execution_time', 0),
            'code_executed': execution_result.get('code'),
            'fallback_used': execution_result.get('fallback_executed', False)
        }
        response['confidence_score'] = self._calculate_response_confidence(
            execution_result, understanding_result, response
        )
        return response

    def _format_numeric_result(self, result: Union[int, float], question: str,
                             execution_result: dict, response: dict) -> dict:
        """Format a single numeric result with localization and logging."""
        if isinstance(result, float):
            if result.is_integer():
                formatted_value = f"{int(result):,}"
            else:
                formatted_value = f"{result:,.2f}"
        else:
            formatted_value = f"{result:,}"
        question_lower = question.lower()
        if any(word in question_lower for word in ['média', 'average', 'mean']):
            response['natural_language_answer'] = self._localize(
                f"A média calculada é **{formatted_value}**.",
                f"The calculated mean is **{formatted_value}**."
            )
        elif any(word in question_lower for word in ['soma', 'total', 'sum']):
            response['natural_language_answer'] = self._localize(
                f"O total calculado é **{formatted_value}**.",
                f"The calculated total is **{formatted_value}**."
            )
        elif any(word in question_lower for word in ['máximo', 'max', 'maior']):
            response['natural_language_answer'] = self._localize(
                f"O valor máximo encontrado é **{formatted_value}**.",
                f"The maximum value found is **{formatted_value}**."
            )
        elif any(word in question_lower for word in ['mínimo', 'min', 'menor']):
            response['natural_language_answer'] = self._localize(
                f"O valor mínimo encontrado é **{formatted_value}**.",
                f"The minimum value found is **{formatted_value}**."
            )
        elif any(word in question_lower for word in ['count', 'quantos', 'número']):
            response['natural_language_answer'] = self._localize(
                f"O total de registros é **{formatted_value}**.",
                f"The total number of records is **{formatted_value}**."
            )
        else:
            response['natural_language_answer'] = self._localize(
                f"O resultado da análise é **{formatted_value}**.",
                f"The result of the analysis is **{formatted_value}**."
            )
        viz = self._create_single_value_chart(result, question)
        if viz:
            logger.debug("Single value chart created for numeric result.")
            response['visualizations'].append(viz)
        response['data_insights'] = self._generate_numeric_insights(result, question)
        return response

    def _format_series_result(self, result: pd.Series, question: str,
                            execution_result: dict, response: dict) -> dict:
        """Format a pandas Series result with localization and logging."""
        series_length = len(result)
        if series_length == 0:
            response['natural_language_answer'] = self._localize(
                "A análise não retornou dados.",
                "The analysis returned no data."
            )
            logger.debug("Series result is empty; fallback response used.")
            return response
        if series_length == 1:
            value = result.iloc[0]
            response['natural_language_answer'] = self._localize(
                f"O resultado da análise é **{value}** para {result.index[0]}",
                f"The result of the analysis is **{value}** for {result.index[0]}"
            )
        else:
            response['natural_language_answer'] = self._localize(
                f"A análise retornou **{series_length}** resultados.",
                f"The analysis returned **{series_length}** results."
            )
            if result.dtype in ['int64', 'float64']:
                top_value = result.max()
                top_index = result.idxmax()
                response['natural_language_answer'] += self._localize(
                    f" O maior valor é **{top_value:,.2f}** para {top_index}.",
                    f" The highest value is **{top_value:,.2f}** for {top_index}."
                )
        viz = self._create_series_chart(result, question)
        if viz:
            logger.debug("Series chart created for series result.")
            response['visualizations'].append(viz)
        response['data_insights'] = self._generate_series_insights(result, question)
        return response

    def _format_dataframe_result(self, result: pd.DataFrame, question: str,
                               execution_result: dict, response: dict) -> dict:
        """Format a pandas DataFrame result with localization and logging."""
        if result.empty:
            response['natural_language_answer'] = self._localize(
                "A análise retornou um DataFrame vazio.",
                "The analysis returned an empty DataFrame."
            )
            logger.debug("DataFrame result is empty; fallback response used.")
            return response
        response['natural_language_answer'] = self._localize(
            f"A análise retornou um DataFrame com {result.shape[0]} linhas e {result.shape[1]} colunas.",
            f"The analysis returned a DataFrame with {result.shape[0]} rows and {result.shape[1]} columns."
        )
        viz = self._create_dataframe_chart(result, question)
        if viz:
            logger.debug("DataFrame chart created for DataFrame result.")
            response['visualizations'].append(viz)
        response['data_insights'] = self._generate_dataframe_insights(result, question)
        return response

    def _format_dict_result(self, result: dict, question: str,
                          execution_result: dict, response: dict) -> dict:
        """Format a dict result with localization and logging."""
        if not result:
            response['natural_language_answer'] = self._localize(
                "A análise retornou um dicionário vazio.",
                "The analysis returned an empty dictionary."
            )
            logger.debug("Dict result is empty; fallback response used.")
            return response
        response['natural_language_answer'] = self._localize(
            f"A análise retornou um dicionário com {len(result)} entradas.",
            f"The analysis returned a dictionary with {len(result)} entries."
        )
        return response

    def _format_generic_result(self, result: Any, question: str,
                             execution_result: dict, response: dict) -> dict:
        """Format a generic result with localization and logging."""
        response['natural_language_answer'] = self._localize(
            f"O resultado da análise é: {result}",
            f"The result of the analysis is: {result}"
        )
        logger.debug("Generic result formatting used.")
        return response

    def _format_error_response(self, execution_result: dict, question: str, response: dict) -> dict:
        """Format an error response with localization and logging."""
        error = execution_result.get('error')
        if not error:
            response['natural_language_answer'] = self._localize(
                "Ocorreu um erro desconhecido ao executar a análise.",
                "An unknown error occurred while executing the analysis."
            )
            logger.debug("Unknown error; fallback error response used.")
            return response
        response['natural_language_answer'] = self._localize(
            f"Erro ao executar a análise: {error}",
            f"Error executing analysis: {error}"
        )
        logger.debug(f"Error response formatted: {error}")
        return response

    def _localize(self, pt_text: str, en_text: str) -> str:
        """Return text in the current language."""
        return pt_text if self.language == 'pt-BR' else en_text

    def _create_single_value_chart(self, value: Union[int, float], question: str) -> Optional[dict]:
        """Create a simple chart for a single value"""
        try:
            title = self._localize("Resultado", "Result")
            
            # Create a simple gauge or bar chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = value,
                title = {'text': title},
                gauge = {
                    'axis': {'range': [None, value * 1.2]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, value * 0.5], 'color': "lightgray"},
                        {'range': [value * 0.5, value], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': value
                    }
                }
            ))
            
            chart_title = self._localize('Resultado da Análise', 'Analysis Result')
            return {
                'type': 'plotly',
                'data': fig.to_dict(),
                'title': chart_title
            }
        except Exception as e:
            logger.debug(f"Failed to create single value chart: {str(e)}")
            return None
    
    def _create_series_chart(self, series: pd.Series, question: str) -> Optional[dict]:
        """Create a chart for Series data"""
        try:
            chart_title = self._localize("Resultados da Análise", "Analysis Results")
            x_title = self._localize("Categorias", "Categories")
            y_title = self._localize("Valores", "Values")
            
            # Determine chart type based on data
            if len(series) <= 20:
                # Bar chart for small series
                fig = px.bar(
                    x=series.index.astype(str),
                    y=series.values,
                    title=chart_title
                )
            else:
                # Line chart for larger series
                fig = px.line(
                    x=series.index,
                    y=series.values,
                    title=chart_title
                )
            
            fig.update_layout(
                xaxis_title=x_title,
                yaxis_title=y_title,
                showlegend=False
            )
            
            viz_title = self._localize('Visualização dos Resultados', 'Results Visualization')
            return {
                'type': 'plotly',
                'data': fig.to_dict(),
                'title': viz_title
            }
        except Exception as e:
            logger.debug(f"Failed to create series chart: {str(e)}")
            return None
    
    def _create_dataframe_chart(self, df: pd.DataFrame, question: str) -> Optional[dict]:
        """Create a chart for DataFrame data"""
        try:
            # Simple heuristic for chart type
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) >= 2:
                # Scatter plot for numeric columns
                chart_title = self._localize("Visualização dos Dados", "Data Visualization")
                fig = px.scatter(
                    df,
                    x=numeric_cols[0],
                    y=numeric_cols[1],
                    title=chart_title
                )
            elif len(numeric_cols) == 1:
                # Histogram for single numeric column
                chart_title = self._localize("Distribuição dos Dados", "Data Distribution")
                fig = px.histogram(
                    df,
                    x=numeric_cols[0],
                    title=chart_title
                )
            else:
                # Count plot for categorical data
                chart_title = self._localize("Distribuição dos Dados", "Data Distribution")
                first_col = df.columns[0]
                value_counts = df[first_col].value_counts().head(10)
                fig = px.bar(
                    x=value_counts.index.astype(str),
                    y=value_counts.values,
                    title=chart_title
                )
            
            viz_title = self._localize('Visualização dos Dados', 'Data Visualization')
            return {
                'type': 'plotly',
                'data': fig.to_dict(),
                'title': viz_title
            }
        except Exception as e:
            logger.debug(f"Failed to create dataframe chart: {str(e)}")
            return None
    
    def _generate_numeric_insights(self, value: Union[int, float], question: str) -> List[str]:
        """Generate insights for numeric results (modularized for reuse/testing)."""
        insights = []
        if isinstance(value, (int, float)):
            if value == 0:
                insights.append(self._localize("O valor é zero.", "The value is zero."))
            elif value < 0:
                insights.append(self._localize("O valor é negativo.", "The value is negative."))
            else:
                insights.append(self._localize("O valor é positivo.", "The value is positive."))
        return insights
    
    def _generate_series_insights(self, series: pd.Series, question: str) -> List[str]:
        """Generate insights for series results (modularized for reuse/testing)."""
        insights = []
        if series.empty:
            insights.append(self._localize("A série está vazia.", "The series is empty."))
        else:
            insights.append(self._localize(f"A série possui {len(series)} elementos.", f"The series has {len(series)} elements."))
            if series.dtype in ['int64', 'float64'] and len(series) > 1:
                std_dev = series.std()
                mean_val = series.mean()
                if std_dev > mean_val:
                    insights.append(self._localize("Os dados apresentam alta variabilidade.", "The data shows high variability."))
                else:
                    insights.append(self._localize("Os dados são relativamente homogêneos.", "The data is relatively homogeneous."))
        return insights
    
    def _generate_dataframe_insights(self, df: pd.DataFrame, question: str) -> List[str]:
        """Generate insights for DataFrame results (modularized for reuse/testing)."""
        insights = []
        if df.empty:
            insights.append(self._localize("O DataFrame está vazio.", "The DataFrame is empty."))
        else:
            insights.append(self._localize(f"O DataFrame possui {df.shape[0]} linhas e {df.shape[1]} colunas.", f"The DataFrame has {df.shape[0]} rows and {df.shape[1]} columns."))
            # Check for missing data
            missing_data = df.isnull().sum().sum()
            if missing_data > 0:
                insights.append(self._localize(f"Há {missing_data} valores ausentes nos dados.", f"There are {missing_data} missing values in the data."))
        return insights
    
    def _calculate_response_confidence(self, execution_result: dict,
                                     understanding_result: dict, response: dict) -> float:
        """Calculate a confidence score for the formatted response."""
        score = 0.5
        if execution_result.get('success'):
            score += 0.25
        if response.get('natural_language_answer'):
            score += 0.15
        if response.get('visualizations'):
            score += 0.1
        return min(score, 1.0)
    
    def get_formatting_history(self) -> List[dict]:
        """Return the history of all formatted responses."""
        return self.formatting_history
    
    def clear_history(self):
        """Clear the formatting history."""
        self.formatting_history.clear() 