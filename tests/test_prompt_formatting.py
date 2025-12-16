"""Tests for prompt formatting."""

import pytest

from qwen3_reranker.core.prompt import PromptFormatter, PromptTemplates

# Official Qwen3 templates from model card
OFFICIAL_PREFIX = """<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
"""

OFFICIAL_SUFFIX = """<|im_end|>
<|im_start|>assistant
<think>

</think>

"""

OFFICIAL_QUERY_TEMPLATE = (
    "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
)


@pytest.fixture
def templates() -> PromptTemplates:
    """Create templates matching official Qwen3 format."""
    return PromptTemplates(
        prefix=OFFICIAL_PREFIX,
        suffix=OFFICIAL_SUFFIX,
        query_template=OFFICIAL_QUERY_TEMPLATE,
    )


@pytest.fixture
def formatter(templates: PromptTemplates) -> PromptFormatter:
    """Create a formatter with official templates."""
    return PromptFormatter(
        templates=templates,
        default_instruction="Given a web search query, retrieve relevant passages that answer the query",
    )


class TestPromptTemplates:
    """Tests for PromptTemplates class."""

    def test_format_pair_basic(self, templates: PromptTemplates) -> None:
        """Test basic prompt formatting."""
        prompt = templates.format_pair(
            instruction="Find relevant documents",
            query="What is Python?",
            doc="Python is a programming language.",
        )

        # Check that all parts are present
        assert "<|im_start|>system" in prompt
        assert "<|im_start|>user" in prompt
        assert "<|im_start|>assistant" in prompt
        assert "<think>" in prompt
        assert "</think>" in prompt
        assert "<Instruct>: Find relevant documents" in prompt
        assert "<Query>: What is Python?" in prompt
        assert "<Document>: Python is a programming language." in prompt

    def test_format_pair_structure(self, templates: PromptTemplates) -> None:
        """Test that prompt has correct structure order."""
        prompt = templates.format_pair(
            instruction="test instruction",
            query="test query",
            doc="test doc",
        )

        # Check ordering
        system_pos = prompt.find("<|im_start|>system")
        user_pos = prompt.find("<|im_start|>user")
        assistant_pos = prompt.find("<|im_start|>assistant")

        assert system_pos < user_pos < assistant_pos

    def test_format_pair_ends_with_suffix(self, templates: PromptTemplates) -> None:
        """Test that prompt ends with the suffix."""
        prompt = templates.format_pair(
            instruction="test",
            query="test",
            doc="test",
        )

        # Should end with empty think tags and newlines
        assert prompt.endswith("</think>\n\n")


class TestPromptFormatter:
    """Tests for PromptFormatter class."""

    def test_format_single_uses_default_instruction(
        self, formatter: PromptFormatter
    ) -> None:
        """Test that format_single uses default instruction when none provided."""
        prompt = formatter.format_single(
            query="test query",
            doc="test document",
        )

        assert "Given a web search query, retrieve relevant passages" in prompt

    def test_format_single_with_custom_instruction(
        self, formatter: PromptFormatter
    ) -> None:
        """Test that format_single uses provided instruction."""
        prompt = formatter.format_single(
            query="test query",
            doc="test document",
            instruction="Custom instruction here",
        )

        assert "<Instruct>: Custom instruction here" in prompt
        assert "retrieve relevant passages" not in prompt

    def test_format_batch(self, formatter: PromptFormatter) -> None:
        """Test batch formatting."""
        docs = ["doc1", "doc2", "doc3"]
        prompts = formatter.format_batch(
            query="test query",
            documents=docs,
        )

        assert len(prompts) == 3
        assert "<Document>: doc1" in prompts[0]
        assert "<Document>: doc2" in prompts[1]
        assert "<Document>: doc3" in prompts[2]

        # All should have the same query
        for prompt in prompts:
            assert "<Query>: test query" in prompt

    def test_format_batch_with_custom_instruction(
        self, formatter: PromptFormatter
    ) -> None:
        """Test batch formatting with custom instruction."""
        docs = ["doc1", "doc2"]
        prompts = formatter.format_batch(
            query="query",
            documents=docs,
            instruction="Custom",
        )

        for prompt in prompts:
            assert "<Instruct>: Custom" in prompt

    def test_special_characters_in_content(self, formatter: PromptFormatter) -> None:
        """Test handling of special characters in content."""
        prompt = formatter.format_single(
            query='Query with "quotes" and <tags>',
            doc="Doc with {braces} and [brackets]",
        )

        # Content should be preserved as-is
        assert '"quotes"' in prompt
        assert "<tags>" in prompt  # Note: this is inside <Document>
        assert "{braces}" in prompt
        assert "[brackets]" in prompt

    def test_empty_document(self, formatter: PromptFormatter) -> None:
        """Test handling of empty document."""
        prompt = formatter.format_single(
            query="test query",
            doc="",
        )

        # Should still have document marker
        assert "<Document>: " in prompt

    def test_multiline_document(self, formatter: PromptFormatter) -> None:
        """Test handling of multiline document."""
        doc = """Line 1
Line 2
Line 3"""
        prompt = formatter.format_single(
            query="test",
            doc=doc,
        )

        assert "Line 1\nLine 2\nLine 3" in prompt
