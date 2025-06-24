# 🤖 Guia de Integração LLM - Sistema de Seleção de Provedor

## 📋 Visão Geral

A classe `QuestionUnderstandingAgent` foi refatorada para incluir um sistema de **seleção explícita de provedor**, permitindo que o usuário escolha qual motor de análise utilizar:

1. **🧠 OpenAI (GPT-4o)**: Para análises complexas e robustas
2. **⚡ Groq (Qwen3-32B)**: Para processamento rápido e eficiente
3. **🔍 Regex**: Para análises simples baseadas em padrões

**Importante**: Não há mais fallback automático - o provedor selecionado é usado exclusivamente.

---

## 🚀 Configuração

### Pré-requisitos

```bash
pip install langchain-openai langchain-groq
```

### Configuração das API Keys

```bash
# Para OpenAI
export OPENAI_API_KEY=sua_api_key_openai

# Para Groq
export GROQ_API_KEY=sua_api_key_groq

# Opcional: especificar modelo Groq
export GROQ_MODEL=qwen/qwen3-32b
```

### Ou no código Python:

```python
import os
os.environ['OPENAI_API_KEY'] = 'sua_api_key_openai'
os.environ['GROQ_API_KEY'] = 'sua_api_key_groq'
```

---

## 💡 Como Funciona

### 1. Seleção de Provedor

```python
from utils.llm_integration import reset_llm_integration

# Configurar provedor explicitamente
reset_llm_integration("openai")   # Para usar OpenAI GPT-4o
reset_llm_integration("groq")     # Para usar Groq Qwen3-32B
reset_llm_integration("regex")    # Para usar apenas padrões regex
```

### 2. Inicialização do Agente

```python
from agents.question_understanding import QuestionUnderstandingAgent

# O agente usa o provedor configurado globalmente
agent = QuestionUnderstandingAgent()

# Verificar provedor ativo
provider_mode = agent.llm_integration.provider_mode
print(f"Provedor ativo: {provider_mode}")
```

### 3. Processamento de Perguntas

```python
import pandas as pd

# Dados de exemplo
df = pd.DataFrame({
    'valor_total': [100, 200, 300],
    'produto': ['A', 'B', 'C']
})

dataframes = {'vendas.csv': df}

# Fazer pergunta - usa o provedor selecionado
result = agent.understand_question(
    "Qual é a soma dos valores totais?", 
    dataframes
)

print(f"Provedor usado: {result['code_source']}")  # 'openai', 'groq' ou 'regex'
print(f"Código gerado: {result['generated_code']}")
```

---

## 🔧 Funcionalidades por Provedor

### 🧠 OpenAI (GPT-4o)

**Características:**
- 📊 Interpretação mais sofisticada e robusta
- 🎯 Melhor para análises complexas e casos edge
- 🔍 Capacidade superior de entender contexto
- 💰 Custo mais alto por requisição

**Método `_generate_code_with_openai()`:**
```python
def _generate_code_with_openai(self, question: str, df_name: str, df: pd.DataFrame) -> Optional[str]
```

**Exemplo de uso:**
```python
# Para análises complexas
reset_llm_integration("openai")
result = agent.understand_question(
    "Analise a correlação entre vendas e sazonalidade, identificando padrões trimestrais",
    dataframes
)
# Resultado: código pandas sofisticado com análise estatística
```

### ⚡ Groq (Qwen3-32B)

**Características:**
- 🚀 Processamento mais rápido
- ⚡ Eficiente para análises diretas
- 💰 Custo menor por requisição
- 🎯 Otimizado para respostas rápidas

**Método `_generate_code_with_groq()`:**
```python
def _generate_code_with_groq(self, question: str, df_name: str, df: pd.DataFrame) -> Optional[str]
```

**Exemplo de uso:**
```python
# Para análises rápidas
reset_llm_integration("groq")
result = agent.understand_question(
    "Top 10 produtos por volume de vendas",
    dataframes
)
# Resultado: código pandas eficiente e direto
```

### 🔍 Regex (Padrões)

**Características:**
- 🆓 Sempre disponível, sem custos
- ⚡ Resposta instantânea
- 🎯 Ideal para operações básicas
- 🔒 Não depende de APIs externas

**Método `_generate_code_with_regex()`:**
```python
def _generate_code_with_regex(self, question: str, df_name: str, df: pd.DataFrame) -> Optional[str]
```

**Exemplo de uso:**
```python
# Para operações simples
reset_llm_integration("regex")
result = agent.understand_question(
    "Soma da coluna valor_total",
    dataframes
)
# Resultado: código pandas básico e confiável
```

---

## 📊 Estrutura de Resposta

```python
{
    'original_question': 'Qual é a soma dos valores?',
    'target_dataframe': 'vendas.csv',
    'generated_code': 'df = dataframes["vendas.csv"]\nresult = df["valor_total"].sum()',
    'confidence': 0.95,
    'explanation': 'Código gerado usando OpenAI GPT-4o',
    'code_source': 'openai',  # 'openai', 'groq' ou 'regex'
    'understood_intent': 'Soma de valores da coluna especificada',
    'provider_info': {
        'provider': 'openai',
        'model': 'gpt-4o',
        'response_time': 2.3
    }
}
```

---

## 🎯 Exemplos Práticos por Provedor

### Exemplo 1: Análise Complexa (OpenAI)

```python
# Configurar OpenAI
reset_llm_integration("openai")
agent = QuestionUnderstandingAgent()

result = agent.understand_question(
    "Identifique outliers nas vendas e explique possíveis causas baseadas na sazonalidade",
    dataframes
)

# Resultado esperado:
# code_source: 'openai'
# confidence: 0.95
# generated_code: código sofisticado com análise estatística
```

### Exemplo 2: Análise Rápida (Groq)

```python
# Configurar Groq
reset_llm_integration("groq")
agent = QuestionUnderstandingAgent()

result = agent.understand_question(
    "Média de vendas por categoria nos últimos 6 meses",
    dataframes
)

# Resultado esperado:
# code_source: 'groq'
# confidence: 0.90
# generated_code: código eficiente e direto
```

### Exemplo 3: Operação Básica (Regex)

```python
# Configurar Regex
reset_llm_integration("regex")
agent = QuestionUnderstandingAgent()

result = agent.understand_question(
    "Máximo da coluna valor_total",
    dataframes
)

# Resultado esperado:
# code_source: 'regex'
# confidence: 1.0
# generated_code: df['valor_total'].max()
```

---

## 📈 Vantagens da Nova Arquitetura

### 🎯 **Controle Total do Usuário**
- ✅ Escolha explícita do provedor
- ✅ Previsibilidade total dos resultados
- ✅ Controle de custos direto
- ✅ Transparência completa

### 🧠 **OpenAI (GPT-4o)**
- ✅ Interpretação mais robusta
- ✅ Melhor para casos complexos
- ✅ Suporte superior a edge cases
- ✅ Análises mais profundas

### ⚡ **Groq (Qwen3-32B)**
- ✅ Velocidade superior
- ✅ Eficiência de processamento
- ✅ Custo-benefício otimizado
- ✅ Ideal para alto volume

### 🔍 **Regex**
- ✅ Sempre disponível
- ✅ Zero dependências externas
- ✅ Resposta instantânea
- ✅ Custo zero

---

## 🔒 Segurança

### Validações por Provedor

Todos os provedores aplicam as mesmas validações de segurança:

```python
# Elementos obrigatórios
required_elements = [
    f"dataframes['{df_name}']",  # Carregamento do DataFrame
    "result =",                   # Atribuição do resultado
]

# Padrões perigosos bloqueados
dangerous_patterns = [
    'import os', 'import sys', 'exec(', 'eval(', 
    'open(', '__import__', 'subprocess'
]
```

### Isolamento por Provedor

Cada provedor opera de forma isolada:
- **OpenAI**: Validação adicional de prompts
- **Groq**: Otimização de performance
- **Regex**: Validação de padrões conhecidos

---

## 📊 Monitoramento e Logs

### Logs por Provedor

```python
# Logs de seleção
logger.info("Provider selected: openai")
logger.info("Provider switched from groq to regex")

# Logs de processamento
logger.debug("OpenAI Prompt enviado: ...")
logger.debug("Groq Response recebida: ...")
logger.info("✅ Usando código gerado por OpenAI")
logger.info("⚡ Usando código gerado por Groq")
logger.info("🔍 Usando padrão regex identificado")

# Logs de validação
logger.info("✅ Código OpenAI validado com sucesso")
logger.warning("❌ Código Groq inválido para esta pergunta")
```

### Métricas por Provedor

```python
# Acessar estatísticas de uso
from utils.llm_integration import llm_integration

stats = llm_integration.get_usage_stats()
print(f"Provedor ativo: {stats['primary_provider']}")
print(f"Modelo usado: {stats['model']}")
print(f"Modo selecionado: {stats['selected_mode']}")
```

---

## 🎯 Casos de Uso por Provedor

### 1. OpenAI - Análises Complexas
- "Analise tendências sazonais e sugira estratégias de vendas"
- "Identifique correlações entre múltiplas variáveis"
- "Explique anomalias nos dados com contexto de negócio"

### 2. Groq - Análises Rápidas
- "Top 20 produtos mais vendidos por categoria"
- "Resumo de performance mensal por vendedor"
- "Distribuição de clientes por faixa etária"

### 3. Regex - Operações Básicas
- "Soma da coluna receita"
- "Média de idade dos clientes"
- "Contagem de registros por status"

---

## 🚀 Configuração Avançada

### Personalização por Provedor

```python
from utils.llm_integration import LLMIntegration

# Configuração customizada para OpenAI
openai_integration = LLMIntegration(provider="openai")
if openai_integration.client:
    openai_integration.client.temperature = 0.0  # Mais determinístico
    openai_integration.client.max_tokens = 1000   # Respostas mais longas

# Configuração customizada para Groq
groq_integration = LLMIntegration(provider="groq")
if groq_integration.client:
    groq_integration.client.temperature = 0.1   # Levemente criativo
    groq_integration.client.max_tokens = 500    # Respostas concisas
```

### Seleção Dinâmica por Tipo de Pergunta

```python
def select_optimal_provider(question: str) -> str:
    """Seleciona o provedor ideal baseado no tipo de pergunta"""
    
    # Palavras-chave para análises complexas (OpenAI)
    complex_keywords = ['correlação', 'tendência', 'análise', 'explique', 'sugira']
    
    # Palavras-chave para análises rápidas (Groq)
    fast_keywords = ['top', 'ranking', 'resumo', 'média', 'total']
    
    # Palavras-chave para operações básicas (Regex)
    basic_keywords = ['soma', 'máximo', 'mínimo', 'contagem']
    
    question_lower = question.lower()
    
    if any(keyword in question_lower for keyword in complex_keywords):
        return "openai"
    elif any(keyword in question_lower for keyword in fast_keywords):
        return "groq"
    elif any(keyword in question_lower for keyword in basic_keywords):
        return "regex"
    else:
        return "groq"  # Default para equilíbrio velocidade/capacidade

# Uso automático
optimal_provider = select_optimal_provider(user_question)
reset_llm_integration(optimal_provider)
```

---

## 🔄 Migração da Versão Anterior

### Principais Mudanças

1. **Eliminação do Fallback Automático**
   ```python
   # Antes (v2.0): fallback automático
   # O sistema tentava LLM → Regex automaticamente
   
   # Agora (v3.1): seleção explícita
   reset_llm_integration("openai")  # Usa apenas OpenAI
   reset_llm_integration("groq")    # Usa apenas Groq
   reset_llm_integration("regex")   # Usa apenas Regex
   ```

2. **Nova Interface de Seleção**
   ```python
   # Antes: sistema decidia automaticamente
   agent = QuestionUnderstandingAgent()
   
   # Agora: usuário escolhe explicitamente
   reset_llm_integration("groq")
   agent = QuestionUnderstandingAgent()
   ```

3. **Controle de Custos**
   ```python
   # Agora você controla exatamente quando usar APIs pagas
   if budget_available and complex_analysis_needed:
       reset_llm_integration("openai")
   elif speed_required:
       reset_llm_integration("groq")
   else:
       reset_llm_integration("regex")  # Sem custos
   ```

---

## 🎉 Conclusão

A nova arquitetura de **seleção explícita de provedor** oferece:

- 🎯 **Controle Total**: Usuário decide qual provedor usar
- 🧠 **OpenAI (GPT-4o)**: Para análises complexas e robustas
- ⚡ **Groq (Qwen3-32B)**: Para processamento rápido e eficiente
- 🔍 **Regex**: Para disponibilidade garantida sem custos
- 🔒 **Segurança**: Validação robusta em todos os provedores
- 📊 **Transparência**: Rastreabilidade completa do processamento
- 💰 **Controle de Custos**: Uso intencional de APIs pagas

O sistema está pronto para produção e oferece flexibilidade total para diferentes cenários de uso!

## 🔧 **Configuração de Múltiplos Provedores**

O CSV QA Agent agora oferece **seleção explícita de provedor**, permitindo que o usuário escolha qual motor de análise utilizar para cada sessão:

### 🎯 **Provedores Disponíveis**

1. **🧠 OpenAI (GPT-4o)**
   - **Modelo**: `gpt-4o`
   - **Características**: Alta precisão, análises complexas, interpretação robusta
   - **Ideal para**: Casos complexos, análises estratégicas, edge cases

2. **⚡ Groq (Qwen3-32B)**
   - **Modelo**: `qwen/qwen3-32b`
   - **Características**: Alta velocidade, processamento eficiente, custo otimizado
   - **Ideal para**: Análises rápidas, alto volume, operações diretas

3. **🔍 Regex (Padrões)**
   - **Modelo**: Sistema baseado em padrões pré-definidos
   - **Características**: Sempre disponível, sem custos, resposta instantânea
   - **Ideal para**: Operações básicas, ambiente sem internet, controle total de custos

### 🚫 **Eliminação do Fallback Automático**

**Mudança Importante**: O sistema não possui mais fallback automático entre provedores. Cada provedor opera de forma independente conforme a seleção do usuário.

**Benefícios:**
- **Previsibilidade**: Resultados consistentes com o provedor escolhido
- **Controle de Custos**: Evita uso não intencional de APIs pagas
- **Transparência**: Clareza total sobre qual provedor está processando
- **Performance**: Otimização específica para cada caso de uso

### ⚙️ **Como Configurar**

Para habilitar os provedores desejados:

```bash
# OpenAI (para análises robustas)
export OPENAI_API_KEY=sk-sua_chave_openai

# Groq (para análises rápidas)
export GROQ_API_KEY=gsk-sua_chave_groq

# Opcional: especificar modelo Groq diferente
export GROQ_MODEL=qwen/qwen3-32b

# Regex está sempre disponível (não requer configuração)
```

### 🔍 **Interface de Seleção**

Na aplicação Streamlit, o usuário:

1. **Visualiza** os provedores disponíveis na sidebar
2. **Seleciona** o provedor desejado via interface radio button
3. **Confirma** a mudança com feedback visual imediato
4. **Monitora** o status do provedor ativo em tempo real

### 📊 **Monitoramento**

O sistema registra nos logs:
- Qual provedor foi selecionado pelo usuário
- Tempo de resposta por provedor
- Taxa de sucesso por tipo de provedor
- Custos de API (quando aplicável)

Isso permite análise detalhada de performance e otimização de uso baseada em necessidades específicas. 