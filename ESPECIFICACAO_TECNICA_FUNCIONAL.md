# 📋 Especificação Funcional e Técnica
## Sistema CSV Q&A Agent com Seleção de Provedor

**Versão:** 3.1  
**Data:** Junho 2025  

---

## 🎯 1. ESPECIFICAÇÃO FUNCIONAL

### 1.1 Visão Geral do Sistema

O **CSV Q&A Agent** é uma aplicação web inteligente que permite aos usuários fazer perguntas em linguagem natural sobre dados contidos em arquivos CSV, oferecendo **seleção explícita de provedor** entre OpenAI, Groq ou análise baseada em padrões (Regex) para máxima flexibilidade e controle do usuário.

**Objetivo Principal:** Democratizar o acesso à análise de dados, permitindo que usuários não técnicos obtenham insights de dados CSV através de perguntas em linguagem natural, com controle total sobre o motor de processamento utilizado.

### 1.2 Requisitos Funcionais

#### RF001 - Upload e Processamento de Arquivos
- **Descrição:** Sistema deve aceitar upload de arquivos CSV e ZIP
- **Critérios de Aceitação:**
  - Suporte a múltiplos arquivos simultaneamente
  - Detecção automática de encoding (UTF-8, ISO-8859-1, etc.)
  - Validação de integridade dos arquivos
  - Limite máximo de 100MB por arquivo
  - Suporte a arquivos ZIP contendo CSVs

#### RF002 - Análise Automática de Schema
- **Descrição:** Análise automática da estrutura dos dados carregados
- **Critérios de Aceitação:**
  - Identificação de tipos de dados (numérico, texto, data)
  - Detecção de valores nulos e duplicados
  - Cálculo de métricas de qualidade dos dados
  - Identificação de relacionamentos entre datasets
  - Score de qualidade geral (0-100)

#### RF003 - Seleção Explícita de Provedor
- **Descrição:** Sistema deve permitir seleção explícita do provedor de análise
- **Critérios de Aceitação:**
  - Interface de seleção clara na sidebar da aplicação
  - Três opções disponíveis: OpenAI, Groq, Regex
  - Feedback visual do provedor selecionado
  - Possibilidade de alternar entre provedores durante a sessão
  - Indicação de disponibilidade de cada provedor

#### RF004 - Interpretação de Perguntas por Provedor Selecionado
- **Descrição:** Sistema deve interpretar perguntas usando exclusivamente o provedor selecionado
- **Critérios de Aceitação:**
  - Suporte bilíngue (pt-BR e en-US)
  - Processamento exclusivo pelo provedor selecionado (sem fallback)
  - Identificação automática de operações (soma, média, máximo, etc.)
  - Mapeamento de colunas mencionadas nas perguntas
  - Detecção de arquivo de destino quando especificado
  - Transparência total sobre qual provedor foi utilizado

#### RF005 - Geração de Código por Provedor
- **Descrição:** Geração de código pandas executável baseado no provedor selecionado
- **Critérios de Aceitação:**
  - **OpenAI (GPT-4o)**: Código sofisticado para análises complexas
  - **Groq (Qwen3-32B)**: Código eficiente para análises rápidas
  - **Regex**: Código baseado em padrões para operações básicas
  - Código pandas válido e executável
  - Validação de segurança (bloqueio de operações perigosas)
  - Transparência sobre o provedor utilizado

#### RF006 - Execução Segura de Código
- **Descrição:** Execução controlada do código gerado com tratamento de erros
- **Critérios de Aceitação:**
  - Ambiente de execução isolado
  - Validação de código antes da execução
  - Tratamento de exceções com mensagens claras
  - Timeout para operações longas (30 segundos)
  - Logging completo de execução por provedor

#### RF007 - Formatação de Respostas
- **Descrição:** Apresentação de resultados em linguagem natural com visualizações
- **Critérios de Aceitação:**
  - Respostas em linguagem natural
  - Geração automática de gráficos quando apropriado
  - Insights específicos do provedor utilizado
  - Localização em português e inglês
  - Indicadores de confiança das respostas
  - Informação clara sobre qual provedor foi utilizado

#### RF008 - Interface Web Interativa
- **Descrição:** Interface amigável para seleção de provedor e interação
- **Critérios de Aceitação:**
  - Seleção de provedor na sidebar com radio buttons
  - Upload drag-and-drop de arquivos
  - Visualização prévia dos dados carregados
  - Campo de pergunta com sugestões
  - Histórico de perguntas e respostas
  - Feedback em tempo real sobre provedor ativo
  - Design responsivo para diferentes dispositivos

### 1.3 Casos de Uso Principais

#### CU001 - Análise Complexa com OpenAI
**Ator:** Analista Estratégico  
**Fluxo:**
1. Seleciona "OpenAI" na sidebar
2. Upload de arquivo "vendas_2024.csv"
3. Pergunta: "Analise a sazonalidade das vendas e sugira estratégias de crescimento"
4. Sistema gera código sofisticado com análise estatística
5. Exibe resultado com insights estratégicos e visualizações

#### CU002 - Análise Rápida com Groq
**Ator:** Gerente Operacional  
**Fluxo:**
1. Seleciona "Groq" na sidebar
2. Upload de múltiplos arquivos CSV
3. Pergunta: "Top 20 produtos por volume de vendas nos últimos 3 meses"
4. Sistema gera código eficiente
5. Exibe resultado rápido com ranking e gráfico

#### CU003 - Operação Básica com Regex
**Ator:** Usuário Geral  
**Fluxo:**
1. Seleciona "Regex" na sidebar (sem necessidade de API keys)
2. Upload de dataset simples
3. Pergunta: "Qual é a soma da coluna valor_total?"
4. Sistema identifica padrão e gera código básico
5. Exibe resultado instantâneo

### 1.4 Regras de Negócio

#### RN001 - Seleção de Provedor
- Usuário deve selecionar explicitamente o provedor antes de fazer perguntas
- Não há fallback automático entre provedores
- Provedor selecionado é usado exclusivamente para a sessão
- Alteração de provedor requer seleção manual do usuário

#### RN002 - Disponibilidade de Provedores
- **OpenAI**: Requer `OPENAI_API_KEY` configurada
- **Groq**: Requer `GROQ_API_KEY` configurada
- **Regex**: Sempre disponível, não requer configuração

#### RN003 - Processamento de Arquivos
- Arquivos com mais de 100MB são rejeitados
- Encoding é detectado automaticamente com fallback para UTF-8
- Arquivos corrompidos geram mensagem de erro específica

#### RN004 - Segurança de Execução
- Código gerado é validado antes da execução
- Operações de sistema são bloqueadas
- Timeout de 30 segundos para operações longas

#### RN005 - Transparência de Provedor
- Sistema sempre informa qual provedor foi utilizado
- Logs registram todas as operações por provedor
- Interface mostra status do provedor ativo em tempo real

---

## ⚙️ 2. ESPECIFICAÇÃO TÉCNICA

### 2.1 Arquitetura do Sistema

#### 2.1.1 Visão Geral da Arquitetura

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   External      │
│   (Streamlit)   │◄──►│   (Python)      │◄──►│   (APIs)        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Provider Select │    │ Data Processing │    │ Provider APIs   │
│ (User Choice)   │    │   (Pandas)      │    │ OpenAI/Groq     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

#### 2.1.2 Componentes Principais

1. **Presentation Layer**
   - `app.py` - Interface Streamlit com seleção de provedor
   - Gerenciamento de estado da sessão
   - Upload e visualização de arquivos

2. **Business Logic Layer**
   - `CSVLoaderAgent` - Carregamento e validação de arquivos
   - `SchemaAnalyzerAgent` - Análise de estrutura de dados
   - `QuestionUnderstandingAgent` - Interpretação por provedor selecionado
   - `QueryExecutorAgent` - Execução segura de código
   - `AnswerFormatterAgent` - Formatação de respostas

3. **Provider Integration Layer**
   - `LLMIntegration` - Gerenciamento de provedores
   - Configuração específica por provedor
   - Isolamento de operações por provedor

4. **External Services**
   - OpenAI API (GPT-4o)
   - Groq API (Qwen3-32B)
   - LangChain framework

### 2.2 Especificação dos Provedores

#### 2.2.1 OpenAI Provider (GPT-4o)

**Características:**
- Modelo: `gpt-4o`
- Interpretação mais robusta e sofisticada
- Ideal para análises complexas e casos edge
- Maior custo por requisição

**Interface:**
```python
class OpenAIProvider:
    def generate_code(self, question: str, df_info: dict) -> str
    def validate_response(self, response: str) -> bool
    def get_capabilities(self) -> list
```

#### 2.2.2 Groq Provider (Qwen3-32B)

**Características:**
- Modelo: `qwen/qwen3-32b`
- Processamento mais rápido e eficiente
- Ideal para análises diretas e alto volume
- Custo otimizado

**Interface:**
```python
class GroqProvider:
    def generate_code(self, question: str, df_info: dict) -> str
    def optimize_for_speed(self, prompt: str) -> str
    def get_performance_metrics(self) -> dict
```

#### 2.2.3 Regex Provider

**Características:**
- Sistema baseado em padrões pré-definidos
- Sempre disponível, sem dependências externas
- Resposta instantânea
- Zero custos

**Interface:**
```python
class RegexProvider:
    def match_patterns(self, question: str) -> list
    def generate_basic_code(self, pattern: str, columns: list) -> str
    def get_supported_operations(self) -> list
```

### 2.3 Estruturas de Dados

#### 2.3.1 Provider Selection Result
```python
@dataclass
class ProviderResult:
    provider_used: str  # 'openai', 'groq', 'regex'
    model_name: str
    response_time: float
    confidence: float
    generated_code: str
    explanation: str
    capabilities_used: list
```

#### 2.3.2 Estrutura de Resposta Unificada

```python
{
    'original_question': str,           # Pergunta original do usuário
    'target_dataframe': Optional[str],  # Nome do arquivo identificado
    'target_columns': List[str],        # Colunas identificadas
    'operations': List[dict],           # Operações detectadas
    'generated_code': Optional[str],    # Código pandas gerado
    'confidence': float,                # Score de confiança (0.0-1.0)
    'explanation': str,                 # Explicação em linguagem natural
    'code_source': str,                 # 'openai', 'groq', ou 'regex'
    'provider_info': {                  # Informações do provedor
        'provider': str,
        'model': str,
        'response_time': float,
        'capabilities': list
    },
    'understood_intent': Optional[str], # Intenção interpretada
    'error': Optional[str]              # Mensagem de erro se houver
}
```

### 2.4 Algoritmos Principais

#### 2.4.1 Algoritmo de Seleção de Provedor

```python
def process_with_selected_provider(question, dataframes, selected_provider):
    # 1. Validar provedor selecionado
    if not is_provider_available(selected_provider):
        return create_error_response("Provedor não disponível")
    
    # 2. Configurar provedor específico
    provider = initialize_provider(selected_provider)
    
    # 3. Processar pergunta com provedor selecionado
    if selected_provider == "openai":
        result = process_with_openai(question, dataframes, provider)
    elif selected_provider == "groq":
        result = process_with_groq(question, dataframes, provider)
    else:  # regex
        result = process_with_regex(question, dataframes)
    
    # 4. Adicionar metadados do provedor
    result['provider_info'] = get_provider_metadata(selected_provider)
    
    return result
```

#### 2.4.2 Algoritmo de Processamento por Provedor

```python
def process_with_openai(question, dataframes, provider):
    # Otimizado para análises complexas
    context = build_complex_context(dataframes)
    prompt = create_sophisticated_prompt(question, context)
    code = provider.generate_advanced_code(prompt)
    return validate_and_execute(code, 'openai')

def process_with_groq(question, dataframes, provider):
    # Otimizado para velocidade
    context = build_efficient_context(dataframes)
    prompt = create_optimized_prompt(question, context)
    code = provider.generate_fast_code(prompt)
    return validate_and_execute(code, 'groq')

def process_with_regex(question, dataframes):
    # Baseado em padrões conhecidos
    patterns = identify_question_patterns(question)
    code = generate_pattern_based_code(patterns, dataframes)
    return validate_and_execute(code, 'regex')
```

### 2.5 Performance e Escalabilidade

#### 2.5.1 Métricas de Performance por Provedor

| Operação | OpenAI (GPT-4o) | Groq (Qwen3-32B) | Regex |
|----------|-----------------|-------------------|-------|
| Análise Simples | 2-4s | 0.5-1s | <0.1s |
| Análise Complexa | 3-8s | 1-3s | N/A |
| Upload (10MB) | < 2s | < 2s | < 2s |
| Validação | < 0.5s | < 0.5s | < 0.1s |
| Execução | < 1s | < 1s | < 1s |

#### 2.5.2 Limitações por Provedor

- **OpenAI**: Rate limit da API (50 req/min), custo por token
- **Groq**: Rate limit da API (100 req/min), limitações de modelo
- **Regex**: Limitado a padrões pré-definidos, sem aprendizado
- **Geral**: 2GB RAM por sessão, timeout 30s

### 2.6 Segurança

#### 2.6.1 Validação por Provedor

Todos os provedores aplicam validações idênticas:

```python
SECURITY_VALIDATIONS = {
    'code_validation': {
        'required_elements': ['dataframes[', 'result ='],
        'blocked_operations': ['import os', 'exec(', 'eval(', 'open('],
        'timeout': 30
    },
    'provider_isolation': {
        'openai': {'api_key_encryption': True, 'prompt_sanitization': True},
        'groq': {'api_key_encryption': True, 'response_validation': True},
        'regex': {'pattern_validation': True, 'safe_execution': True}
    }
}
```

#### 2.6.2 Isolamento de Provedores

- Cada provedor opera em contexto isolado
- APIs keys armazenadas de forma segura
- Logs separados por provedor
- Validação específica por tipo de provedor

### 2.7 Monitoramento e Logging

#### 2.7.1 Logs por Provedor

```python
LOGGING_BY_PROVIDER = {
    'openai': {
        'file': 'logs/openai_usage.log',
        'metrics': ['response_time', 'token_usage', 'success_rate']
    },
    'groq': {
        'file': 'logs/groq_usage.log',
        'metrics': ['response_time', 'request_count', 'success_rate']
    },
    'regex': {
        'file': 'logs/regex_usage.log',
        'metrics': ['pattern_matches', 'execution_time', 'success_rate']
    }
}
```

#### 2.7.2 Métricas de Monitoramento

- Taxa de uso por provedor
- Tempo médio de resposta por provedor
- Taxa de sucesso por tipo de pergunta
- Custos de API (OpenAI/Groq)
- Padrões de uso e preferências do usuário

### 2.8 Deployment e Configuração

#### 2.8.1 Variáveis de Ambiente por Provedor

```bash
# === PROVEDORES ===
OPENAI_API_KEY=sk-...                    # Habilita OpenAI GPT-4o
GROQ_API_KEY=gsk-...                     # Habilita Groq Qwen3-32B
GROQ_MODEL=qwen/qwen3-32b               # Modelo Groq (padrão)

# === CONFIGURAÇÕES POR PROVEDOR ===
OPENAI_TEMPERATURE=0.1                   # Temperatura OpenAI
OPENAI_MAX_TOKENS=1000                   # Tokens máximos OpenAI
GROQ_TEMPERATURE=0.1                     # Temperatura Groq
GROQ_MAX_TOKENS=500                      # Tokens máximos Groq

# === MONITORAMENTO ===
ENABLE_PROVIDER_METRICS=true             # Métricas por provedor
LOG_PROVIDER_USAGE=true                  # Log de uso por provedor
```

#### 2.8.2 Comando de Execução

```bash
# Desenvolvimento (apenas Regex disponível)
streamlit run app.py

# Produção com OpenAI
OPENAI_API_KEY=$OPENAI_API_KEY streamlit run app.py

# Produção com todos os provedores
OPENAI_API_KEY=$OPENAI_API_KEY GROQ_API_KEY=$GROQ_API_KEY streamlit run app.py
```

---

## 🎯 3. CONSIDERAÇÕES DE IMPLEMENTAÇÃO

### 3.1 Roadmap de Desenvolvimento

#### Fase Atual (v3.1) - COMPLETA ✅
- [x] Sistema de seleção explícita de provedor
- [x] Interface de usuário para escolha de provedor
- [x] Eliminação de fallback automático
- [x] Integração OpenAI (GPT-4o) e Groq (Qwen3-32B)

#### Fase Próxima (v3.2) - Em Planejamento
- [ ] Métricas detalhadas por provedor
- [ ] Cache inteligente baseado no provedor
- [ ] Otimização de custos por provedor
- [ ] Dashboard de performance comparativa

#### Fase Futura (v4.0) - Visão
- [ ] Novos provedores (Claude, Gemini)
- [ ] Auto-seleção baseada em tipo de pergunta
- [ ] Análise comparativa automática
- [ ] Machine learning para otimização de escolha

### 3.2 Riscos e Mitigações

| Risco | Probabilidade | Impacto | Mitigação |
|-------|---------------|---------|-----------|
| Falha de API específica | Média | Médio | Usuário pode alternar manualmente |
| Custos elevados (OpenAI) | Baixa | Alto | Controle explícito pelo usuário |
| Performance variável | Média | Baixo | Métricas claras por provedor |
| Confusão na seleção | Baixa | Baixo | Interface intuitiva e documentação |

### 3.3 Manutenibilidade

- **Isolamento de Provedores**: Cada provedor é independente
- **Logging Detalhado**: Rastreabilidade por provedor
- **Configuração Flexível**: Parâmetros específicos por provedor
- **Testes Abrangentes**: Suíte de testes por provedor
- **Documentação Específica**: Guias detalhados para cada provedor

---

## 📊 4. CONCLUSÃO

O sistema CSV Q&A Agent v3.1 representa uma evolução significativa em análise de dados democratizada, oferecendo:

**Controle Total do Usuário:**
- Seleção explícita de provedor sem fallback automático
- Transparência completa sobre processamento
- Controle direto de custos e performance

**Flexibilidade de Provedores:**
- OpenAI (GPT-4o) para análises complexas e robustas
- Groq (Qwen3-32B) para processamento rápido e eficiente
- Regex para disponibilidade garantida sem dependências

**Arquitetura Robusta:**
- Segurança enterprise com validação multicamada
- Monitoramento detalhado por provedor
- Escalabilidade para diferentes cenários de uso

O sistema está pronto para uso em produção e oferece uma base sólida para futuras expansões, mantendo sempre o princípio de controle total do usuário sobre o processamento de suas análises de dados. 