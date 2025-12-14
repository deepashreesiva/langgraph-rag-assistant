"""Tests for LangGraph workflow."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from app.graph.state import GraphState
from app.graph.nodes import (
    query_analysis_node,
    retrieval_node,
    relevance_check_node,
    generation_node,
    source_attribution_node,
    fallback_node,
    clarification_node
)
from app.graph.workflow import create_workflow, should_retrieve, should_generate


class TestGraphState:
    """Tests for GraphState initialization."""

    def test_initial_state(self):
        """Test that initial state has correct defaults."""
        state: GraphState = {
            "question": "What is LangGraph?",
            "session_id": "test-session",
            "chat_history": [],
            "documents": [],
            "relevance_scores": [],
            "needs_retrieval": True,
            "needs_clarification": False,
            "answer": "",
            "confidence": 0.0,
            "sources": []
        }

        assert state["question"] == "What is LangGraph?"
        assert state["needs_retrieval"] is True
        assert len(state["documents"]) == 0


class TestQueryAnalysisNode:
    """Tests for query analysis node."""

    @patch('app.graph.nodes.ChatOpenAI')
    def test_simple_question_needs_retrieval(self, mock_llm):
        """Test that factual questions trigger retrieval."""
        mock_response = Mock()
        mock_response.content = '{"needs_retrieval": true, "needs_clarification": false}'
        mock_llm.return_value.invoke.return_value = mock_response

        state: GraphState = {
            "question": "What are the system requirements?",
            "session_id": "test",
            "chat_history": [],
            "documents": [],
            "relevance_scores": [],
            "needs_retrieval": False,
            "needs_clarification": False,
            "answer": "",
            "confidence": 0.0,
            "sources": []
        }

        result = query_analysis_node(state)
        assert result.get("needs_retrieval", state["needs_retrieval"]) is True

    @patch('app.graph.nodes.ChatOpenAI')
    def test_vague_question_needs_clarification(self, mock_llm):
        """Test that vague questions trigger clarification."""
        mock_response = Mock()
        mock_response.content = '{"needs_retrieval": false, "needs_clarification": true}'
        mock_llm.return_value.invoke.return_value = mock_response

        state: GraphState = {
            "question": "How do I do it?",
            "session_id": "test",
            "chat_history": [],
            "documents": [],
            "relevance_scores": [],
            "needs_retrieval": False,
            "needs_clarification": False,
            "answer": "",
            "confidence": 0.0,
            "sources": []
        }

        result = query_analysis_node(state)
        # Vague questions should trigger clarification
        assert "needs_clarification" in result or state["needs_clarification"]


class TestRetrievalNode:
    """Tests for retrieval node."""

    @patch('app.graph.nodes.get_vector_store')
    def test_retrieval_returns_documents(self, mock_vector_store):
        """Test that retrieval returns relevant documents."""
        mock_doc = Mock()
        mock_doc.page_content = "LangGraph is a library for building stateful agents."
        mock_doc.metadata = {"source": "docs.md"}

        mock_store = Mock()
        mock_store.similarity_search_with_score.return_value = [(mock_doc, 0.85)]
        mock_vector_store.return_value = mock_store

        state: GraphState = {
            "question": "What is LangGraph?",
            "session_id": "test",
            "chat_history": [],
            "documents": [],
            "relevance_scores": [],
            "needs_retrieval": True,
            "needs_clarification": False,
            "answer": "",
            "confidence": 0.0,
            "sources": []
        }

        result = retrieval_node(state)

        assert "documents" in result
        assert len(result["documents"]) > 0

    @patch('app.graph.nodes.get_vector_store')
    def test_retrieval_empty_store(self, mock_vector_store):
        """Test retrieval with empty vector store."""
        mock_store = Mock()
        mock_store.similarity_search_with_score.return_value = []
        mock_vector_store.return_value = mock_store

        state: GraphState = {
            "question": "Unknown topic",
            "session_id": "test",
            "chat_history": [],
            "documents": [],
            "relevance_scores": [],
            "needs_retrieval": True,
            "needs_clarification": False,
            "answer": "",
            "confidence": 0.0,
            "sources": []
        }

        result = retrieval_node(state)
        assert result.get("documents", []) == []


class TestRelevanceCheckNode:
    """Tests for relevance check node."""

    def test_high_relevance_passes(self):
        """Test that high relevance scores pass the check."""
        mock_doc = Mock()
        mock_doc.page_content = "Relevant content"

        state: GraphState = {
            "question": "Test question",
            "session_id": "test",
            "chat_history": [],
            "documents": [mock_doc],
            "relevance_scores": [0.9],
            "needs_retrieval": True,
            "needs_clarification": False,
            "answer": "",
            "confidence": 0.0,
            "sources": []
        }

        result = relevance_check_node(state)
        # High relevance should keep documents
        assert len(result.get("documents", state["documents"])) > 0

    def test_low_relevance_filtered(self):
        """Test that low relevance documents are filtered."""
        mock_doc = Mock()
        mock_doc.page_content = "Irrelevant content"

        state: GraphState = {
            "question": "Test question",
            "session_id": "test",
            "chat_history": [],
            "documents": [mock_doc],
            "relevance_scores": [0.2],
            "needs_retrieval": True,
            "needs_clarification": False,
            "answer": "",
            "confidence": 0.0,
            "sources": []
        }

        result = relevance_check_node(state)
        # Low relevance documents should be filtered
        docs = result.get("documents", state["documents"])
        assert len(docs) == 0 or result.get("confidence", 0) < 0.5


class TestGenerationNode:
    """Tests for generation node."""

    @patch('app.graph.nodes.ChatOpenAI')
    def test_generation_with_context(self, mock_llm):
        """Test generation with retrieved context."""
        mock_response = Mock()
        mock_response.content = "LangGraph is a library for building agents."
        mock_llm.return_value.invoke.return_value = mock_response

        mock_doc = Mock()
        mock_doc.page_content = "LangGraph documentation content"
        mock_doc.metadata = {"source": "docs.md"}

        state: GraphState = {
            "question": "What is LangGraph?",
            "session_id": "test",
            "chat_history": [],
            "documents": [mock_doc],
            "relevance_scores": [0.9],
            "needs_retrieval": True,
            "needs_clarification": False,
            "answer": "",
            "confidence": 0.0,
            "sources": []
        }

        result = generation_node(state)

        assert "answer" in result
        assert len(result["answer"]) > 0


class TestWorkflowRouting:
    """Tests for workflow routing functions."""

    def test_should_retrieve_true(self):
        """Test should_retrieve returns correct route."""
        state: GraphState = {
            "question": "Test",
            "session_id": "test",
            "chat_history": [],
            "documents": [],
            "relevance_scores": [],
            "needs_retrieval": True,
            "needs_clarification": False,
            "answer": "",
            "confidence": 0.0,
            "sources": []
        }

        result = should_retrieve(state)
        assert result == "retrieval"

    def test_should_retrieve_clarification(self):
        """Test routing to clarification."""
        state: GraphState = {
            "question": "Test",
            "session_id": "test",
            "chat_history": [],
            "documents": [],
            "relevance_scores": [],
            "needs_retrieval": False,
            "needs_clarification": True,
            "answer": "",
            "confidence": 0.0,
            "sources": []
        }

        result = should_retrieve(state)
        assert result == "clarification"

    def test_should_generate_with_docs(self):
        """Test should_generate with documents."""
        mock_doc = Mock()

        state: GraphState = {
            "question": "Test",
            "session_id": "test",
            "chat_history": [],
            "documents": [mock_doc],
            "relevance_scores": [0.8],
            "needs_retrieval": True,
            "needs_clarification": False,
            "answer": "",
            "confidence": 0.0,
            "sources": []
        }

        result = should_generate(state)
        assert result == "generation"

    def test_should_generate_fallback(self):
        """Test should_generate routes to fallback without docs."""
        state: GraphState = {
            "question": "Test",
            "session_id": "test",
            "chat_history": [],
            "documents": [],
            "relevance_scores": [],
            "needs_retrieval": True,
            "needs_clarification": False,
            "answer": "",
            "confidence": 0.0,
            "sources": []
        }

        result = should_generate(state)
        assert result == "fallback"


class TestSourceAttributionNode:
    """Tests for source attribution node."""

    def test_source_extraction(self):
        """Test that sources are properly extracted."""
        mock_doc = Mock()
        mock_doc.metadata = {"source": "architecture.md", "title": "Architecture"}

        state: GraphState = {
            "question": "Test",
            "session_id": "test",
            "chat_history": [],
            "documents": [mock_doc],
            "relevance_scores": [0.85],
            "needs_retrieval": True,
            "needs_clarification": False,
            "answer": "The architecture uses microservices.",
            "confidence": 0.0,
            "sources": []
        }

        result = source_attribution_node(state)

        assert "sources" in result
        assert "confidence" in result


class TestFallbackNode:
    """Tests for fallback node."""

    def test_fallback_message(self):
        """Test fallback provides helpful message."""
        state: GraphState = {
            "question": "Unknown topic question",
            "session_id": "test",
            "chat_history": [],
            "documents": [],
            "relevance_scores": [],
            "needs_retrieval": True,
            "needs_clarification": False,
            "answer": "",
            "confidence": 0.0,
            "sources": []
        }

        result = fallback_node(state)

        assert "answer" in result
        assert len(result["answer"]) > 0
        # Fallback should have low confidence
        assert result.get("confidence", 1.0) < 0.5


class TestClarificationNode:
    """Tests for clarification node."""

    def test_clarification_response(self):
        """Test clarification provides helpful questions."""
        state: GraphState = {
            "question": "How do I do that?",
            "session_id": "test",
            "chat_history": [],
            "documents": [],
            "relevance_scores": [],
            "needs_retrieval": False,
            "needs_clarification": True,
            "answer": "",
            "confidence": 0.0,
            "sources": []
        }

        result = clarification_node(state)

        assert "answer" in result
        # Clarification should ask for more info
        assert "?" in result["answer"] or "clarif" in result["answer"].lower()


@pytest.fixture
def sample_state() -> GraphState:
    """Provide a sample state for testing."""
    return {
        "question": "What is LangGraph?",
        "session_id": "test-session-123",
        "chat_history": [],
        "documents": [],
        "relevance_scores": [],
        "needs_retrieval": True,
        "needs_clarification": False,
        "answer": "",
        "confidence": 0.0,
        "sources": []
    }
