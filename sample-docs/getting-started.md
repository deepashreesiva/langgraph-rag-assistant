# Getting Started with LangGraph RAG Assistant

## Overview

The LangGraph RAG Assistant is an enterprise-ready Retrieval Augmented Generation (RAG) system that combines the power of large language models with your organization's knowledge base.

## Key Features

- **Multi-step Reasoning**: Uses LangGraph to implement sophisticated reasoning workflows
- **Hybrid Search**: Combines semantic and keyword search for better retrieval
- **Source Attribution**: Every answer includes citations to source documents
- **Conversation Memory**: Maintains context across multiple conversation turns
- **Confidence Scoring**: Provides confidence levels for generated answers

## Architecture

The system consists of several components:

1. **Document Ingestion Pipeline**: Processes and stores documents in a vector database
2. **Query Analysis**: Understands user intent and determines retrieval strategy
3. **Retrieval Engine**: Finds relevant documents using similarity search
4. **Generation Engine**: Creates responses using retrieved context
5. **Source Attribution**: Tracks and formats source citations

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- OpenAI API key

### Installation

1. Clone the repository
2. Copy `.env.example` to `.env` and add your OpenAI API key
3. Run `docker-compose up -d`
4. Access the API at http://localhost:8000/docs

## API Endpoints

### Query Documents

```bash
POST /query
{
  "question": "What is LangGraph?",
  "session_id": "optional-session-id"
}
```

### Ingest Documents

```bash
POST /ingest
{
  "documents": [
    {
      "content": "Document text here",
      "metadata": {"source": "manual", "title": "My Document"}
    }
  ]
}
```

### Health Check

```bash
GET /health
```

## Best Practices

1. **Document Preparation**: Clean and structure your documents before ingestion
2. **Chunking Strategy**: Adjust chunk size based on your document types
3. **Query Formulation**: Be specific in your questions for better results
4. **Session Management**: Use session IDs for multi-turn conversations
