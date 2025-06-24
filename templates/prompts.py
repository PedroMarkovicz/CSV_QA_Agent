"""
Prompt templates for CSV Q&A Agent system
Defines prompts for LLM-enhanced agent capabilities
"""

from typing import Dict, Any, List
import pandas as pd

class CSVLoaderPrompts:
    """Prompts for CSVLoaderAgent LLM-enhanced features"""
    
    # Schema inference prompts
    COLUMN_TYPE_ANALYSIS = """
Você é um especialista em análise de dados. Analise as seguintes amostras de uma coluna CSV e determine:

1. O tipo semântico da coluna (ID, nome, data, valor monetário, categoria, etc.)
2. O padrão dos dados
3. Possíveis problemas de qualidade
4. Sugestões de tipo pandas otimizado

**Nome da coluna:** {column_name}
**Amostras dos dados:**
{sample_data}

**Estatísticas:**
- Total de valores: {total_count}
- Valores únicos: {unique_count}
- Valores nulos: {null_count}
- Tipo atual: {current_dtype}

Responda em JSON:
{{
    "semantic_type": "tipo_semantico",
    "data_pattern": "descrição_do_padrão",
    "quality_issues": ["lista", "de", "problemas"],
    "recommended_dtype": "tipo_pandas_sugerido",
    "confidence": 0.0-1.0,
    "explanation": "explicação_detalhada"
}}
"""

    RELATIONSHIP_DETECTION = """
Você é um especialista em modelagem de dados. Analise estas colunas de diferentes tabelas CSV e identifique possíveis relacionamentos:

**Tabela 1:** {table1_name}
**Coluna 1:** {column1_name}
**Amostras:** {column1_samples}

**Tabela 2:** {table2_name}  
**Coluna 2:** {column2_name}
**Amostras:** {column2_samples}

**Estatísticas de sobreposição:**
- Valores em comum: {overlap_count}
- Percentual de sobreposição: {overlap_percentage}%

Determine se existe relacionamento e qual tipo:

Responda em JSON:
{{
    "has_relationship": true/false,
    "relationship_type": "foreign_key|similar_data|no_relationship",
    "strength": "high|medium|low",
    "confidence": 0.0-1.0,
    "explanation": "explicação_do_relacionamento",
    "recommendations": ["sugestões", "para", "uso"]
}}
"""

    ENCODING_ANALYSIS = """
Você é um especialista em codificação de arquivos. Um arquivo CSV falhou ao ser carregado com as codificações padrão.

**Nome do arquivo:** {filename}
**Codificações tentadas:** {attempted_encodings}
**Erro:** {error_message}
**Primeiros bytes (hex):** {file_bytes_hex}

Com base nesta informação, sugira:
1. Possíveis codificações alternativas
2. Se o arquivo pode estar corrompido
3. Estratégias de recuperação

Responda em JSON:
{{
    "suggested_encodings": ["lista", "de", "codificações"],
    "likely_corruption": true/false,
    "recovery_strategies": ["estratégias", "de", "recuperação"],
    "confidence": 0.0-1.0,
    "explanation": "análise_detalhada"
}}
"""

    PARSING_ERROR_ANALYSIS = """
Você é um especialista em análise de arquivos CSV. Um arquivo falhou ao ser parseado.

**Nome do arquivo:** {filename}
**Erro:** {error_message}
**Linha do erro:** {error_line}
**Delimitador usado:** {delimiter}
**Primeiras linhas do arquivo:**
{file_preview}

Analise o problema e sugira soluções:

Responda em JSON:
{{
    "problem_type": "tipo_do_problema",
    "likely_cause": "causa_provável",
    "suggested_solutions": ["solução1", "solução2"],
    "parsing_options": {{
        "delimiter": "delimitador_sugerido",
        "quotechar": "caractere_aspas",
        "skiprows": número_linhas_pular,
        "header": número_linha_cabeçalho
    }},
    "confidence": 0.0-1.0,
    "explanation": "explicação_detalhada"
}}
"""

    DATA_QUALITY_ASSESSMENT = """
Você é um especialista em qualidade de dados. Analise este DataFrame CSV e avalie sua qualidade geral:

**Nome do arquivo:** {filename}
**Dimensões:** {rows} linhas × {columns} colunas
**Colunas:** {column_names}
**Tipos de dados:** {dtypes}
**Valores nulos por coluna:** {null_counts}
**Linhas duplicadas:** {duplicate_rows}

**Amostras de cada coluna:**
{sample_data}

Avalie a qualidade e forneça recomendações:

Responda em JSON:
{{
    "overall_quality_score": 0.0-100.0,
    "quality_issues": [
        {{
            "type": "tipo_problema",
            "severity": "high|medium|low", 
            "columns_affected": ["colunas"],
            "description": "descrição",
            "recommendations": ["recomendações"]
        }}
    ],
    "data_insights": ["insights", "importantes"],
    "recommended_actions": ["ações", "recomendadas"],
    "usability_assessment": "avaliação_de_usabilidade"
}}
"""

class QuestionUnderstandingPrompts:
    """Prompts for QuestionUnderstandingAgent"""
    
    QUESTION_ANALYSIS = """
Você é um especialista em análise de dados que ajuda usuários a fazer perguntas sobre datasets CSV.

**Pergunta do usuário:** {user_question}

**Datasets disponíveis:**
{available_datasets}

**Colunas disponíveis por dataset:**
{dataset_schemas}

Sua tarefa é:
1. Compreender a intenção da pergunta
2. Identificar qual(is) dataset(s) usar
3. Identificar as colunas relevantes
4. Determinar as operações pandas necessárias
5. Gerar código pandas executável

Responda em JSON:
{{
    "understood_intent": "descrição_da_intenção",
    "target_datasets": ["dataset1", "dataset2"],
    "target_columns": ["coluna1", "coluna2"],
    "operations_needed": ["operação1", "operação2"],
    "pandas_code": "código_pandas_completo",
    "confidence": 0.0-1.0,
    "explanation": "explicação_do_que_será_feito",
    "assumptions": ["premissas", "assumidas"],
    "alternative_interpretations": ["outras", "interpretações"]
}}
"""

    CODE_GENERATION = """
Você é um especialista em pandas que gera código para análise de dados.

**Intenção:** {intent}
**Dataset:** {dataset_name}
**Colunas:** {columns}
**Operações:** {operations}

**Schema do dataset:**
{dataset_schema}

**Amostras dos dados:**
{data_samples}

Gere código pandas otimizado e seguro:

Requisitos:
- Use apenas as colunas disponíveis
- Trate valores nulos adequadamente
- Seja eficiente com memória
- Inclua validações básicas
- Retorne resultado na variável 'result'

Responda em JSON:
{{
    "pandas_code": "código_completo",
    "explanation": "explicação_do_código",
    "potential_issues": ["possíveis", "problemas"],
    "performance_notes": ["notas", "de", "performance"],
    "confidence": 0.0-1.0
}}
"""

class AnswerFormatterPrompts:
    """Prompts for AnswerFormatterAgent"""
    
    RESPONSE_FORMATTING = """
Você é um especialista em comunicação de insights de dados para usuários não-técnicos.

**Pergunta original:** {original_question}
**Resultado da análise:** {analysis_result}
**Tipo de resultado:** {result_type}
**Código executado:** {executed_code}

Sua tarefa é transformar este resultado técnico em uma resposta clara e útil:

1. Responda a pergunta em linguagem natural
2. Destaque os insights mais importantes
3. Forneça contexto quando necessário
4. Sugira próximos passos ou análises relacionadas

Responda em JSON:
{{
    "natural_language_answer": "resposta_em_linguagem_natural",
    "key_insights": ["insight1", "insight2"],
    "context_notes": ["notas", "de", "contexto"],
    "recommendations": ["próximos", "passos"],
    "confidence": 0.0-1.0,
    "visualization_suggestions": ["tipos", "de", "gráficos"]
}}
"""

    INSIGHT_GENERATION = """
Você é um analista de dados experiente. Com base nos resultados abaixo, gere insights valiosos:

**Dados analisados:** {data_description}
**Resultado:** {result}
**Contexto da pergunta:** {question_context}

Gere insights acionáveis e relevantes:

Responda em JSON:
{{
    "primary_insights": ["insight_principal_1", "insight_principal_2"],
    "supporting_details": ["detalhe1", "detalhe2"],
    "business_implications": ["implicação1", "implicação2"],
    "data_quality_notes": ["observação1", "observação2"],
    "recommendations": ["recomendação1", "recomendação2"]
}}
"""

class SystemPrompts:
    """System-level prompts for agent coordination"""
    
    ERROR_HANDLING = """
Você é um sistema de tratamento de erros inteligente. Analise o erro abaixo e forneça uma resposta útil ao usuário:

**Tipo de erro:** {error_type}
**Mensagem:** {error_message}
**Contexto:** {error_context}
**Ação do usuário:** {user_action}

Forneça uma explicação clara e sugestões de solução:

Responda em JSON:
{{
    "user_friendly_message": "mensagem_para_o_usuário",
    "possible_causes": ["causa1", "causa2"],
    "suggested_solutions": ["solução1", "solução2"],
    "prevention_tips": ["dica1", "dica2"],
    "requires_support": true/false
}}
"""

    PERFORMANCE_OPTIMIZATION = """
Você é um especialista em otimização de performance para análise de dados.

**Operação:** {operation_type}
**Tamanho dos dados:** {data_size}
**Recursos disponíveis:** {available_resources}
**Tempo atual:** {current_time}

Sugira otimizações para melhorar a performance:

Responda em JSON:
{{
    "optimization_strategies": ["estratégia1", "estratégia2"],
    "memory_recommendations": ["recomendação1", "recomendação2"],
    "alternative_approaches": ["abordagem1", "abordagem2"],
    "estimated_improvement": "percentual_de_melhoria",
    "implementation_complexity": "baixa|média|alta"
}}
"""

# Utility functions for prompt formatting
def format_prompt(template: str, **kwargs) -> str:
    """Format a prompt template with provided arguments"""
    return template.format(**kwargs)

def get_column_samples(df: pd.DataFrame, column: str, n_samples: int = 5) -> List[Any]:
    """Get sample values from a DataFrame column"""
    if column not in df.columns:
        return []
    return df[column].dropna().head(n_samples).tolist()

def get_dataset_schema_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Get a summary of dataset schema for prompts"""
    return {
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'shape': df.shape,
        'null_counts': df.isnull().sum().to_dict(),
        'sample_data': {col: get_column_samples(df, col) for col in df.columns[:10]}
    } 