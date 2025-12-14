# System Architecture

## Overview

The LangGraph RAG Assistant implements a sophisticated multi-step reasoning pipeline using LangGraph's state machine capabilities.

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Layer (FastAPI)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   /query     │    │   /ingest    │    │   /health    │       │
│  └──────┬───────┘    └──────┬───────┘    └──────────────┘       │
│         │                   │                                    │
├─────────┴───────────────────┴────────────────────────────────────┤
│                     LangGraph Workflow                           │
│                                                                  │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐               │
│  │  Query     │──▶│ Retrieval  │──▶│ Relevance  │               │
│  │ Analysis   │   │   Node     │   │   Check    │               │
│  └────────────┘   └────────────┘   └─────┬──────┘               │
│         │                                │                       │
│         ▼                                ▼                       │
│  ┌────────────┐               ┌────────────────┐                │
│  │Clarificat- │               │   Generation   │                │
│  │   ion      │               │     Node       │                │
│  └────────────┘               └───────┬────────┘                │
│                                       │                          │
│                                       ▼                          │
│                               ┌────────────────┐                │
│                               │    Source      │                │
│                               │  Attribution   │                │
│                               └────────────────┘                │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│                     Data Layer                                   │
│                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │   ChromaDB       │    │   Conversation   │                   │
│  │  Vector Store    │    │     Memory       │                   │
│  └──────────────────┘    └──────────────────┘                   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Workflow Nodes

### 1. Query Analysis Node
- Analyzes user intent
- Determines if retrieval is needed
- Identifies if clarification is required
- Extracts key topics from the query

### 2. Retrieval Node
- Performs similarity search on vector store
- Returns top-k relevant documents
- Includes relevance scores for each document

### 3. Relevance Check Node
- Validates retrieved documents are relevant
- Applies minimum score thresholds
- Routes to fallback if no relevant docs found

### 4. Generation Node
- Creates response using LLM
- Incorporates retrieved context
- Maintains conversation history

### 5. Source Attribution Node
- Extracts citations from response
- Formats source references
- Calculates confidence scores

### 6. Clarification Node
- Requests additional information
- Handles ambiguous queries
- Provides clarifying questions

### 7. Fallback Node
- Handles cases with no relevant documents
- Provides helpful suggestions
- Graceful degradation

## State Management

The workflow maintains state through a `GraphState` TypedDict:

```python
class GraphState(TypedDict):
    question: str
    session_id: str
    chat_history: List[Dict]
    documents: List[Document]
    relevance_scores: List[float]
    needs_retrieval: bool
    needs_clarification: bool
    answer: str
    confidence: float
    sources: List[Dict]
```

## Data Flow

1. User submits query via API
2. Query analysis determines intent and retrieval need
3. If retrieval needed, documents are fetched from vector store
4. Relevance check validates document quality
5. Generation creates response with context
6. Source attribution adds citations
7. Response returned to user

## Scalability Considerations

- **Horizontal Scaling**: API can be scaled behind load balancer
- **Vector Store**: ChromaDB can be replaced with distributed solutions
- **Caching**: Implement response caching for common queries
- **Async Processing**: Document ingestion runs asynchronously
