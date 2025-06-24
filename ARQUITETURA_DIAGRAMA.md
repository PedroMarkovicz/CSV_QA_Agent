# 🏗️ Diagramas de Arquitetura - CSV Q&A Agent

## 📊 Diagrama de Componentes Detalhado

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                FRONTEND LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                 │
│  │   File Upload   │  │  Question Input │  │  Results View   │                 │
│  │   Component     │  │   Component     │  │   Component     │                 │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                 │
│                              │                                                  │
│                              ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                     Streamlit Session State                                 │ │
│  │  ┌───────────────┐ ┌─────────────────┐ ┌─────────────────┐                 │ │
│  │  │ uploaded_files│ │ analysis_results│ │  chat_history   │                 │ │
│  │  └───────────────┘ └─────────────────┘ └─────────────────┘                 │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              BUSINESS LOGIC LAYER                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐             │
│  │  CSVLoaderAgent │    │ SchemaAnalyzer  │    │QuestionUnder-   │             │
│  │                 │    │     Agent       │    │standingAgent    │             │
│  │ • load_files()  │───►│ • analyze_df()  │    │ • understand()  │             │
│  │ • detect_enc()  │    │ • quality_score │    │ • _llm_generate │             │
│  │ • validate()    │    │ • relationships │    │ • _regex_parse  │             │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘             │
│           │                       │                       │                    │
│           ▼                       ▼                       ▼                    │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                         Data Processing Pipeline                            │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│           │                                               │                    │
│           ▼                                               ▼                    │
│  ┌─────────────────┐                            ┌─────────────────┐             │
│  │ QueryExecutor   │                            │ AnswerFormatter │             │
│  │     Agent       │                            │     Agent       │             │
│  │ • execute_code()│                            │ • format_resp() │             │
│  │ • validate()    │                            │ • create_viz()  │             │
│  │ • fallback()    │                            │ • localize()    │             │
│  └─────────────────┘                            └─────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               DATA LAYER                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                 │
│  │   DataFrames    │  │  Execution      │  │   Logging       │                 │
│  │   (in memory)   │  │   History       │  │   System        │                 │
│  │                 │  │                 │  │                 │                 │
│  │ Dict[str, df]   │  │ List[results]   │  │ • debug.log     │                 │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            EXTERNAL SERVICES                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐                            ┌─────────────────┐             │
│  │   OpenAI API    │                            │   LangChain     │             │
│  │                 │                            │   Framework     │             │
│  │ • ChatOpenAI    │◄──────────────────────────►│ • HumanMessage  │             │
│  │ • gpt-3.5-turbo │                            │ • Schema        │             │
│  │ • Validation    │                            │ • Utils         │             │
│  └─────────────────┘                            └─────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 🔄 Fluxo de Processamento de Perguntas

```
┌─────────────────┐
│ User Question   │
│ "Soma vendas?"  │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  1. Normalize   │
│  & Clean Text   │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ 2. Identify     │
│ Target DataFrame│
└─────────────────┘
         │
         ▼
┌─────────────────┐    YES   ┌─────────────────┐
│ 3. LLM          │─────────►│ Generate Code   │
│ Available?      │          │ via LLM         │
└─────────────────┘          └─────────────────┘
         │ NO                         │
         ▼                            │
┌─────────────────┐                   │
│ 4. Regex        │                   │
│ Pattern Match   │                   │
└─────────────────┘                   │
         │                            │
         ▼                            │
┌─────────────────┐                   │
│ 5. Generate     │                   │
│ Code via Regex  │                   │
└─────────────────┘                   │
         │                            │
         └─────────┬──────────────────┘
                   ▼
┌─────────────────────────────────────┐
│ 6. Validate Code Safety             │
│ • Check required elements           │
│ • Block dangerous operations        │
│ • Verify syntax                     │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ 7. Execute Code Safely              │
│ • Isolated environment              │
│ • Timeout control                   │
│ • Error handling                    │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ 8. Format Response                  │
│ • Natural language answer           │
│ • Generate visualizations           │
│ • Add insights                      │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ 9. Return to    │
│ User Interface  │
└─────────────────┘
```

## 🔒 Diagrama de Segurança

```
┌─────────────────────────────────────────────────────────────────┐
│                        SECURITY LAYERS                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │   INPUT     │    │    CODE     │    │  EXECUTION  │          │
│  │ VALIDATION  │    │ VALIDATION  │    │  SANDBOX    │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│         │                   │                   │              │
│         ▼                   ▼                   ▼              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │• File size  │    │• Syntax     │    │• Timeout    │          │
│  │• Extension  │    │• Dangerous  │    │• Memory     │          │
│  │• Encoding   │    │• Required   │    │• Isolated   │          │
│  │• Content    │    │  patterns   │    │  elements   │          │
│  │  integrity  │    │  elements   │    │  scope      │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                      BLOCKED OPERATIONS                         │
│                                                                 │
│  ❌ import os, sys         ❌ exec(), eval()                    │
│  ❌ open(), file access    ❌ subprocess                        │
│  ❌ network operations     ❌ __import__                        │
│  ❌ system commands        ❌ environment access               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 Diagrama de Estados da Aplicação

```
┌─────────────────┐
│   INITIAL       │
│   STATE         │
│ • No files      │
│ • Empty history │
└─────────────────┘
         │
         │ upload files
         ▼
┌─────────────────┐
│ FILES_UPLOADED  │
│ • Processing... │
│ • Loading...    │
└─────────────────┘
         │
         │ analysis complete
         ▼
┌─────────────────┐      ask question      ┌─────────────────┐
│ READY_FOR_      │────────────────────────►│ PROCESSING_     │
│ QUESTIONS       │                         │ QUESTION        │
│ • Files loaded  │                         │ • LLM/Regex    │
│ • Schema ready  │                         │ • Generating    │
└─────────────────┘                         └─────────────────┘
         ▲                                           │
         │                                           │
         │ new question                              │
         │                                           ▼
┌─────────────────┐                         ┌─────────────────┐
│ SHOWING_        │◄────────────────────────│ QUESTION_       │
│ RESULTS         │     display results     │ PROCESSED       │
│ • Answer shown  │                         │ • Code executed │
│ • Charts ready  │                         │ • Results ready │
└─────────────────┘                         └─────────────────┘
```

## 🔧 Diagrama de Configuração e Deploy

```
┌─────────────────────────────────────────────────────────────────┐
│                      DEPLOYMENT OPTIONS                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   DEVELOPMENT   │  │   PRODUCTION    │  │     CLOUD       │  │
│  │                 │  │                 │  │                 │  │
│  │ • Local Python  │  │ • Docker        │  │ • Streamlit     │  │
│  │ • streamlit run │  │ • Compose       │  │   Cloud         │  │
│  │ • Debug mode    │  │ • Load balancer │  │ • AWS/GCP       │  │
│  │ • Hot reload    │  │ • Monitoring    │  │ • Kubernetes    │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                    ENVIRONMENT VARIABLES                        │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ # Required                                                  │ │
│  │ STREAMLIT_SERVER_PORT=8501                                  │ │
│  │ STREAMLIT_SERVER_ADDRESS=0.0.0.0                           │ │
│  │                                                             │ │
│  │ # Optional                                                  │ │
│  │ OPENAI_API_KEY=sk-...                                       │ │
│  │ LOG_LEVEL=INFO                                              │ │
│  │ MAX_FILE_SIZE_MB=100                                        │ │
│  │ SESSION_TIMEOUT_HOURS=24                                    │ │
│  │ ENABLE_LLM_INSIGHTS=true                                    │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 📈 Diagrama de Performance e Monitoramento

```
┌─────────────────────────────────────────────────────────────────┐
│                      MONITORING STACK                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   APPLICATION   │  │    METRICS      │  │     LOGS        │  │
│  │    METRICS      │  │   COLLECTION    │  │  AGGREGATION    │  │
│  │                 │  │                 │  │                 │  │
│  │ • Response time │  │ • Prometheus    │  │ • Structured    │  │
│  │ • Success rate  │  │ • Custom        │  │   logging       │  │
│  │ • LLM usage     │  │   counters      │  │ • Log levels    │  │
│  │ • Error count   │  │ • Histograms    │  │ • Correlation   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                     PERFORMANCE TARGETS                         │
│                                                                 │
│  Operation              │ Expected    │ Maximum                 │
│  ──────────────────────┼─────────────┼─────────────            │
│  File Upload (10MB)    │ < 2s        │ 5s                      │
│  Schema Analysis       │ < 1s        │ 3s                      │
│  LLM Generation        │ < 3s        │ 10s                     │
│  Regex Processing      │ < 0.1s      │ 0.5s                    │
│  Code Execution        │ < 1s        │ 30s                     │
│  Response Formatting   │ < 0.5s      │ 2s                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📝 Notas de Implementação

### Patterns Utilizados

1. **Agent Pattern**: Cada agente tem responsabilidade específica
2. **Strategy Pattern**: LLM vs Regex para interpretação
3. **Chain of Responsibility**: Pipeline de processamento
4. **Observer Pattern**: Logging e monitoramento
5. **Facade Pattern**: Interface unificada via Streamlit

### Considerações de Escalabilidade

- **Stateless Design**: Sessões isoladas por usuário
- **Resource Limits**: Timeouts e limites de memória
- **Graceful Degradation**: Fallback quando LLM indisponível
- **Horizontal Scaling**: Múltiplas instâncias Streamlit
- **Cache Strategy**: Cache de resultados para otimização futura 