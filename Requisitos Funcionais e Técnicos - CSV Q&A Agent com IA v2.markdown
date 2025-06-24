# Requisitos Funcionais e Técnicos - CSV Q&A Agent v3.1

## 🎯 Visão Geral

O **CSV Q&A Agent v3.1** é um sistema inteligente de análise de dados que permite aos usuários fazer perguntas em linguagem natural sobre arquivos CSV, oferecendo **seleção explícita de provedor** entre OpenAI, Groq ou análise baseada em padrões (Regex) para máxima flexibilidade e controle do usuário.

### 🆕 Principais Novidades da v3.1

- **🔧 Seleção Explícita de Provedor**: Usuário escolhe qual motor de análise utilizar
- **🚫 Eliminação de Fallback Automático**: Cada provedor opera independentemente
- **🧠 OpenAI (GPT-4o)**: Para análises complexas e robustas
- **⚡ Groq (Qwen3-32B)**: Para processamento rápido e eficiente
- **🔍 Regex**: Para disponibilidade garantida sem dependências
- **📊 Transparência Total**: Usuário sempre sabe qual provedor está sendo usado

---

## 📋 Requisitos Funcionais

### RF001 - Seleção de Provedor de Análise
**Prioridade:** ALTA  
**Descrição:** O sistema deve permitir que o usuário selecione explicitamente qual provedor utilizar para análise de dados.

**Critérios de Aceitação:**
- Interface de seleção na sidebar com radio buttons
- Três opções disponíveis: OpenAI, Groq, Regex
- Feedback visual do provedor selecionado
- Possibilidade de alternar entre provedores durante a sessão
- Indicação clara de disponibilidade de cada provedor
- Status em tempo real do provedor ativo

**Regras de Negócio:**
- OpenAI requer `OPENAI_API_KEY` configurada
- Groq requer `GROQ_API_KEY` configurada  
- Regex está sempre disponível (não requer configuração)
- Não há fallback automático entre provedores
- Provedor selecionado é usado exclusivamente

### RF002 - Upload e Processamento de Arquivos
**Prioridade:** ALTA  
**Descrição:** Sistema deve aceitar upload de arquivos CSV e ZIP com validação robusta.

**Critérios de Aceitação:**
- Suporte a múltiplos arquivos simultaneamente
- Detecção automática de encoding (UTF-8, ISO-8859-1, etc.)
- Validação de integridade dos arquivos
- Limite máximo de 100MB por arquivo
- Suporte a arquivos ZIP contendo CSVs
- Drag-and-drop na interface
- Visualização prévia dos dados carregados

### RF003 - Análise de Schema por Provedor
**Prioridade:** MÉDIA  
**Descrição:** Análise automática da estrutura dos dados com insights específicos do provedor selecionado.

**Critérios de Aceitação:**
- Identificação de tipos de dados (numérico, texto, data)
- Detecção de valores nulos e duplicados
- Cálculo de métricas de qualidade dos dados
- Identificação de relacionamentos entre datasets
- Score de qualidade geral (0-100)
- Insights específicos do provedor selecionado (quando aplicável)

### RF004 - Interpretação de Perguntas por Provedor
**Prioridade:** ALTA  
**Descrição:** Sistema deve interpretar perguntas usando exclusivamente o provedor selecionado pelo usuário.

**Critérios de Aceitação:**
- **OpenAI (GPT-4o)**: Interpretação sofisticada para análises complexas
- **Groq (Qwen3-32B)**: Interpretação eficiente para análises rápidas
- **Regex**: Interpretação baseada em padrões para operações básicas
- Suporte bilíngue (pt-BR e en-US)
- Identificação automática de operações (soma, média, máximo, etc.)
- Mapeamento de colunas mencionadas nas perguntas
- Detecção de arquivo de destino quando especificado
- Transparência total sobre qual provedor foi utilizado

### RF005 - Geração de Código por Provedor
**Prioridade:** ALTA  
**Descrição:** Geração de código pandas executável otimizado para cada provedor.

**Critérios de Aceitação:**
- **OpenAI**: Código sofisticado com análise estatística avançada
- **Groq**: Código eficiente e direto para resultados rápidos
- **Regex**: Código baseado em padrões conhecidos e testados
- Código pandas válido e executável
- Validação de segurança (bloqueio de operações perigosas)
- Informação clara sobre qual provedor gerou o código
- Confiança/score de qualidade do código gerado

### RF006 - Execução Segura de Código
**Prioridade:** ALTA  
**Descrição:** Execução controlada do código gerado com tratamento de erros específico por provedor.

**Critérios de Aceitação:**
- Ambiente de execução isolado (sandbox)
- Validação de código antes da execução
- Tratamento de exceções com mensagens claras
- Timeout para operações longas (30 segundos)
- Logging completo de execução por provedor
- Bloqueio de operações perigosas (import os, exec, eval, etc.)

### RF007 - Formatação de Respostas por Provedor
**Prioridade:** MÉDIA  
**Descrição:** Apresentação de resultados otimizada para cada tipo de provedor.

**Critérios de Aceitação:**
- Respostas em linguagem natural
- Geração automática de gráficos quando apropriado
- Insights específicos do provedor utilizado
- Localização em português e inglês
- Indicadores de confiança das respostas
- Informação clara sobre qual provedor foi utilizado
- Tempo de resposta e métricas de performance

### RF008 - Interface de Usuário com Seleção de Provedor
**Prioridade:** ALTA  
**Descrição:** Interface web intuitiva com controle total sobre seleção de provedor.

**Critérios de Aceitação:**
- Sidebar com seleção de provedor via radio buttons
- Status visual do provedor ativo
- Upload drag-and-drop de arquivos
- Campo de pergunta com sugestões contextuais
- Histórico de perguntas e respostas
- Feedback em tempo real sobre provedor utilizado
- Design responsivo para diferentes dispositivos
- Indicadores de disponibilidade por provedor

### RF009 - Monitoramento por Provedor
**Prioridade:** MÉDIA  
**Descrição:** Sistema de monitoramento específico para cada provedor.

**Critérios de Aceitação:**
- Logs separados por provedor
- Métricas de performance por provedor
- Rastreamento de custos (OpenAI/Groq)
- Taxa de sucesso por tipo de provedor
- Tempo médio de resposta por provedor
- Alertas configuráveis por provedor

---

## ⚙️ Requisitos Técnicos

### RT001 - Arquitetura de Provedores
**Prioridade:** ALTA  
**Descrição:** Arquitetura modular que suporte múltiplos provedores de forma isolada.

**Especificações Técnicas:**
- Padrão Strategy para seleção de provedor
- Interface comum para todos os provedores
- Isolamento completo entre provedores
- Configuração específica por provedor
- Facilidade para adicionar novos provedores

### RT002 - Integração OpenAI (GPT-4o)
**Prioridade:** ALTA  
**Descrição:** Integração robusta com API da OpenAI usando modelo GPT-4o.

**Especificações Técnicas:**
- Modelo: `gpt-4o`
- Framework: LangChain
- Temperatura: 0.1 (configurável)
- Max tokens: 1000 (configurável)
- Retry automático em caso de falha de rede
- Rate limiting respeitando limites da API
- Validação de resposta específica

### RT003 - Integração Groq (Qwen3-32B)
**Prioridade:** ALTA  
**Descrição:** Integração otimizada com API da Groq usando modelo Qwen3-32B.

**Especificações Técnicas:**
- Modelo: `qwen/qwen3-32b` (configurável via `GROQ_MODEL`)
- Framework: LangChain
- Temperatura: 0.1 (configurável)
- Max tokens: 500 (configurável)
- Otimização para velocidade
- Retry automático em caso de falha
- Monitoramento de throughput

### RT004 - Sistema Regex
**Prioridade:** MÉDIA  
**Descrição:** Sistema baseado em padrões regex para garantir disponibilidade.

**Especificações Técnicas:**
- Padrões pré-definidos para operações comuns
- Matching de colunas por nome e tipo
- Geração de código pandas básico
- Validação de padrões conhecidos
- Extensibilidade para novos padrões
- Performance otimizada (< 100ms)

### RT005 - Segurança por Provedor
**Prioridade:** ALTA  
**Descrição:** Validação de segurança específica para cada provedor.

**Especificações Técnicas:**
- Validação de código gerado por todos os provedores
- Sandbox de execução isolado
- Bloqueio de operações perigosas
- Sanitização de inputs específica por provedor
- Criptografia de API keys
- Auditoria de operações por provedor

### RT006 - Performance por Provedor
**Prioridade:** MÉDIA  
**Descrição:** Otimização de performance específica para cada provedor.

**Especificações Técnicas:**
- **OpenAI**: Timeout 10s, retry 3x
- **Groq**: Timeout 5s, retry 2x
- **Regex**: Timeout 1s, sem retry necessário
- Cache de respostas por provedor
- Métricas de performance em tempo real
- Otimização de prompts por provedor

### RT007 - Configuração por Provedor
**Prioridade:** MÉDIA  
**Descrição:** Sistema de configuração flexível para cada provedor.

**Especificações Técnicas:**
```bash
# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
OPENAI_TEMPERATURE=0.1
OPENAI_MAX_TOKENS=1000

# Groq
GROQ_API_KEY=gsk-...
GROQ_MODEL=qwen/qwen3-32b
GROQ_TEMPERATURE=0.1
GROQ_MAX_TOKENS=500

# Sistema
LOG_LEVEL=INFO
ENABLE_PROVIDER_METRICS=true
MAX_FILE_SIZE_MB=100
```

### RT008 - Logging e Monitoramento
**Prioridade:** MÉDIA  
**Descrição:** Sistema de logging específico por provedor.

**Especificações Técnicas:**
- Logs estruturados em JSON
- Separação por provedor (openai.log, groq.log, regex.log)
- Métricas de performance por provedor
- Rastreamento de custos (APIs pagas)
- Alertas configuráveis
- Dashboard de monitoramento

---

## 🔧 Requisitos de Interface

### RI001 - Seleção de Provedor
**Descrição:** Interface clara para seleção de provedor na sidebar.

**Especificações:**
- Radio buttons para OpenAI, Groq, Regex
- Indicador visual do provedor ativo
- Status de disponibilidade de cada provedor
- Feedback imediato ao alternar provedores
- Tooltip com informações sobre cada provedor

### RI002 - Feedback de Provedor
**Descrição:** Informações claras sobre qual provedor está sendo usado.

**Especificações:**
- Badge com nome do provedor na resposta
- Tempo de resposta específico do provedor
- Indicador de confiança da resposta
- Informações sobre o modelo utilizado
- Métricas de performance em tempo real

### RI003 - Gestão de Configuração
**Descrição:** Interface para verificar configuração dos provedores.

**Especificações:**
- Status de API keys (configurada/não configurada)
- Teste de conectividade com APIs
- Informações sobre modelos disponíveis
- Configurações específicas por provedor
- Alertas de configuração incorreta

---

## 🎯 Casos de Uso por Provedor

### CU001 - Análise Estratégica com OpenAI
**Ator:** Diretor de Vendas  
**Cenário:** Análise complexa de tendências de mercado  
**Fluxo:**
1. Seleciona "OpenAI" na sidebar
2. Upload de dados de vendas anuais
3. Pergunta: "Analise a sazonalidade das vendas e sugira estratégias para aumentar receita no próximo trimestre"
4. Sistema usa GPT-4o para gerar análise sofisticada
5. Recebe insights estratégicos com visualizações avançadas

### CU002 - Análise Operacional com Groq
**Ator:** Gerente de Operações  
**Cenário:** Análise rápida para tomada de decisão  
**Fluxo:**
1. Seleciona "Groq" na sidebar
2. Upload de dados de produção diários
3. Pergunta: "Top 20 produtos com maior volume de produção nos últimos 30 dias"
4. Sistema usa Qwen3-32B para resposta rápida
5. Recebe ranking detalhado em menos de 2 segundos

### CU003 - Análise Básica com Regex
**Ator:** Analista Júnior  
**Cenário:** Operação simples sem necessidade de APIs  
**Fluxo:**
1. Seleciona "Regex" na sidebar (sem API keys necessárias)
2. Upload de planilha de vendas
3. Pergunta: "Qual é a soma total da coluna receita?"
4. Sistema usa padrões regex para identificar operação
5. Recebe resultado instantâneo e preciso

### CU004 - Alternância de Provedores
**Ator:** Analista Sênior  
**Cenário:** Uso estratégico de diferentes provedores  
**Fluxo:**
1. Inicia com OpenAI para análise complexa de correlações
2. Alterna para Groq para validação rápida dos resultados
3. Usa Regex para operações básicas de verificação
4. Compara resultados entre provedores
5. Documenta insights obtidos com cada abordagem

---

## 📊 Métricas de Qualidade

### Métricas por Provedor
| Métrica | OpenAI | Groq | Regex | Target |
|---------|--------|------|-------|--------|
| **Tempo de Resposta** | < 5s | < 2s | < 0.1s | ✅ |
| **Taxa de Sucesso** | > 95% | > 90% | 100% | ✅ |
| **Disponibilidade** | 99.9% | 99.9% | 100% | ✅ |
| **Precisão** | > 95% | > 90% | > 85% | ✅ |

### Métricas de Sistema
- **Uptime**: > 99.5%
- **Cobertura de Testes**: > 85%
- **Documentação**: > 95%
- **Segurança**: Score A (sem vulnerabilidades críticas)

---

## 🚀 Roadmap de Implementação

### Fase Atual (v3.1) - COMPLETA ✅
- [x] Sistema de seleção explícita de provedor
- [x] Interface de usuário para escolha
- [x] Integração OpenAI (GPT-4o) e Groq (Qwen3-32B)
- [x] Eliminação de fallback automático
- [x] Documentação atualizada

### Próxima Fase (v3.2) - 4 semanas
- [ ] Dashboard de métricas por provedor
- [ ] Cache inteligente baseado no provedor
- [ ] Otimização de custos e alertas
- [ ] Testes de carga com 100+ usuários

### Fase Futura (v4.0) - 3 meses
- [ ] Novos provedores (Claude, Gemini)
- [ ] Auto-seleção inteligente de provedor
- [ ] API REST com seleção de provedor
- [ ] Análise comparativa automática

---

## 🎉 Conclusão

O **CSV Q&A Agent v3.1** representa uma evolução significativa em sistemas de análise de dados, oferecendo:

- **🎯 Controle Total**: Usuário decide qual provedor utilizar
- **🧠 Flexibilidade**: Três opções otimizadas para diferentes cenários
- **🔒 Segurança**: Validação robusta em todos os provedores
- **📊 Transparência**: Rastreabilidade completa das operações
- **💰 Controle de Custos**: Uso intencional de APIs pagas

O sistema está pronto para produção e oferece uma base sólida para futuras expansões, mantendo sempre o princípio de **controle total do usuário** sobre o processamento de suas análises de dados.