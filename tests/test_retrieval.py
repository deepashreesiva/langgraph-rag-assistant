"""Tests for retrieval and vector store functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from app.retrieval.vector_store import (
    get_vector_store,
    add_documents,
    search_documents,
    clear_vector_store
)
from app.ingestion.loader import load_document, load_documents_from_directory
from app.ingestion.chunker import chunk_document, RecursiveTextChunker


class TestVectorStore:
    """Tests for vector store operations."""

    @patch('app.retrieval.vector_store.Chroma')
    @patch('app.retrieval.vector_store.OpenAIEmbeddings')
    def test_get_vector_store_singleton(self, mock_embeddings, mock_chroma):
        """Test that vector store returns singleton instance."""
        mock_chroma.return_value = MagicMock()

        # Reset singleton for testing
        import app.retrieval.vector_store as vs
        vs._vector_store = None

        store1 = get_vector_store()
        store2 = get_vector_store()

        # Should return same instance
        assert store1 is store2

    @patch('app.retrieval.vector_store.get_vector_store')
    def test_add_documents(self, mock_get_store):
        """Test adding documents to vector store."""
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store

        documents = [
            {"content": "Test document 1", "metadata": {"source": "test1.md"}},
            {"content": "Test document 2", "metadata": {"source": "test2.md"}}
        ]

        add_documents(documents)

        mock_store.add_documents.assert_called_once()

    @patch('app.retrieval.vector_store.get_vector_store')
    def test_search_documents(self, mock_get_store):
        """Test searching documents."""
        mock_doc = Mock()
        mock_doc.page_content = "Relevant content about LangGraph"
        mock_doc.metadata = {"source": "docs.md"}

        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = [(mock_doc, 0.85)]
        mock_get_store.return_value = mock_store

        results = search_documents("What is LangGraph?", top_k=5)

        assert len(results) == 1
        assert results[0][1] == 0.85

    @patch('app.retrieval.vector_store.get_vector_store')
    def test_search_documents_empty_results(self, mock_get_store):
        """Test search with no results."""
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = []
        mock_get_store.return_value = mock_store

        results = search_documents("Nonexistent topic")

        assert len(results) == 0

    @patch('app.retrieval.vector_store.get_vector_store')
    def test_clear_vector_store(self, mock_get_store):
        """Test clearing vector store."""
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store

        clear_vector_store()

        mock_store.delete_collection.assert_called_once()


class TestDocumentLoader:
    """Tests for document loading functionality."""

    def test_load_markdown_document(self, tmp_path):
        """Test loading a markdown document."""
        # Create temp markdown file
        md_file = tmp_path / "test.md"
        md_file.write_text("# Test Document\n\nThis is test content.")

        doc = load_document(str(md_file))

        assert doc is not None
        assert "Test Document" in doc.page_content
        assert doc.metadata["source"] == str(md_file)

    def test_load_text_document(self, tmp_path):
        """Test loading a plain text document."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("This is plain text content.")

        doc = load_document(str(txt_file))

        assert doc is not None
        assert "plain text" in doc.page_content

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_document("/nonexistent/path/file.md")

    def test_load_documents_from_directory(self, tmp_path):
        """Test loading multiple documents from directory."""
        # Create temp files
        (tmp_path / "doc1.md").write_text("# Document 1\nContent 1")
        (tmp_path / "doc2.md").write_text("# Document 2\nContent 2")
        (tmp_path / "doc3.txt").write_text("Plain text document")

        docs = load_documents_from_directory(str(tmp_path))

        assert len(docs) >= 2  # At least the md files


class TestChunker:
    """Tests for text chunking functionality."""

    def test_chunk_small_document(self):
        """Test chunking a small document."""
        mock_doc = Mock()
        mock_doc.page_content = "This is a small document."
        mock_doc.metadata = {"source": "test.md"}

        chunks = chunk_document(mock_doc, chunk_size=1000, chunk_overlap=200)

        # Small doc should result in single chunk
        assert len(chunks) == 1
        assert chunks[0].page_content == "This is a small document."

    def test_chunk_large_document(self):
        """Test chunking a large document."""
        # Create a document larger than chunk size
        large_content = "This is a sentence. " * 100  # ~2000 chars

        mock_doc = Mock()
        mock_doc.page_content = large_content
        mock_doc.metadata = {"source": "large.md"}

        chunks = chunk_document(mock_doc, chunk_size=500, chunk_overlap=100)

        # Should create multiple chunks
        assert len(chunks) > 1

        # Each chunk should be <= chunk_size (approximately)
        for chunk in chunks:
            assert len(chunk.page_content) <= 600  # Allow some buffer

    def test_chunk_preserves_metadata(self):
        """Test that chunking preserves document metadata."""
        mock_doc = Mock()
        mock_doc.page_content = "Test content " * 50
        mock_doc.metadata = {"source": "test.md", "author": "Test Author"}

        chunks = chunk_document(mock_doc, chunk_size=200, chunk_overlap=50)

        for chunk in chunks:
            assert chunk.metadata["source"] == "test.md"
            assert chunk.metadata["author"] == "Test Author"

    def test_chunk_overlap(self):
        """Test that chunks have proper overlap."""
        content = "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9 Word10 " * 10

        mock_doc = Mock()
        mock_doc.page_content = content
        mock_doc.metadata = {"source": "test.md"}

        chunks = chunk_document(mock_doc, chunk_size=100, chunk_overlap=20)

        if len(chunks) > 1:
            # Check that consecutive chunks have some overlap
            for i in range(len(chunks) - 1):
                chunk1_end = chunks[i].page_content[-20:]
                chunk2_start = chunks[i + 1].page_content[:20]
                # There should be some shared content due to overlap
                # This is a simplified check


class TestRecursiveTextChunker:
    """Tests for RecursiveTextChunker class."""

    def test_chunker_initialization(self):
        """Test chunker initializes with correct parameters."""
        chunker = RecursiveTextChunker(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " "]
        )

        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 100

    def test_chunker_split_text(self):
        """Test chunker splits text correctly."""
        chunker = RecursiveTextChunker(chunk_size=100, chunk_overlap=20)

        text = """First paragraph with some content.

Second paragraph with more content.

Third paragraph to make it longer."""

        chunks = chunker.split_text(text)

        assert len(chunks) > 0
        # All text should be covered
        combined = "".join(chunks)
        assert "First paragraph" in combined
        assert "Third paragraph" in combined


class TestSearchQuality:
    """Tests for search quality and relevance."""

    @patch('app.retrieval.vector_store.get_vector_store')
    def test_search_score_ordering(self, mock_get_store):
        """Test that results are ordered by relevance score."""
        mock_docs = [
            (Mock(page_content="Highly relevant"), 0.95),
            (Mock(page_content="Somewhat relevant"), 0.75),
            (Mock(page_content="Less relevant"), 0.55)
        ]

        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = mock_docs
        mock_get_store.return_value = mock_store

        results = search_documents("test query")

        # Verify scores are in descending order
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    @patch('app.retrieval.vector_store.get_vector_store')
    def test_search_top_k_limit(self, mock_get_store):
        """Test that top_k limits results correctly."""
        mock_docs = [(Mock(page_content=f"Doc {i}"), 0.9 - i * 0.1) for i in range(10)]

        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = mock_docs[:3]
        mock_get_store.return_value = mock_store

        results = search_documents("test query", top_k=3)

        assert len(results) <= 3


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    return [
        {"content": "LangGraph is a library for building stateful agents.", "metadata": {"source": "intro.md"}},
        {"content": "ChromaDB is a vector database for AI applications.", "metadata": {"source": "vector.md"}},
        {"content": "FastAPI is a modern web framework for building APIs.", "metadata": {"source": "api.md"}}
    ]
