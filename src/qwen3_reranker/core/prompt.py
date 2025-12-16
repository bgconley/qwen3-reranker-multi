"""Prompt template formatting for Qwen3 Reranker.

Implements the official Qwen3 reranker prompt format with
configurable templates from the profile configuration.
"""

from dataclasses import dataclass


@dataclass
class PromptTemplates:
    """Compiled prompt templates for efficient formatting."""

    prefix: str
    suffix: str
    query_template: str

    def format_pair(self, instruction: str, query: str, doc: str) -> str:
        """Format a single query-document pair into the full prompt.

        The format follows Qwen3's official template:
        - prefix: system message setting up yes/no judgment task
        - query_template: formats instruction, query, and document
        - suffix: closes user turn and triggers assistant response

        Args:
            instruction: Task instruction (e.g., "retrieve relevant passages")
            query: The search query
            doc: The document to evaluate

        Returns:
            Complete prompt string ready for tokenization
        """
        # Format the content using the query template
        content = self.query_template.format(
            instruction=instruction,
            query=query,
            doc=doc,
        )

        # Combine prefix + content + suffix
        return f"{self.prefix}{content}{self.suffix}"


def get_default_instruction() -> str:
    """Return the default instruction from Qwen3 model card."""
    return "Given a web search query, retrieve relevant passages that answer the query"


class PromptFormatter:
    """Formats query-document pairs for the Qwen3 reranker model.

    This class encapsulates the prompt formatting logic and provides
    efficient batch formatting for multiple documents.
    """

    def __init__(self, templates: PromptTemplates, default_instruction: str) -> None:
        """Initialize the formatter.

        Args:
            templates: Compiled prompt templates
            default_instruction: Default instruction to use when none provided
        """
        self.templates = templates
        self.default_instruction = default_instruction

    @classmethod
    def from_scoring_config(
        cls,
        prefix: str,
        suffix: str,
        query_template: str,
        default_instruction: str,
    ) -> "PromptFormatter":
        """Create a formatter from scoring configuration."""
        templates = PromptTemplates(
            prefix=prefix,
            suffix=suffix,
            query_template=query_template,
        )
        return cls(templates, default_instruction)

    def format_single(
        self,
        query: str,
        doc: str,
        instruction: str | None = None,
    ) -> str:
        """Format a single query-document pair.

        Args:
            query: The search query
            doc: The document to evaluate
            instruction: Optional instruction override

        Returns:
            Complete prompt string
        """
        instr = instruction if instruction is not None else self.default_instruction
        return self.templates.format_pair(instr, query, doc)

    def format_batch(
        self,
        query: str,
        documents: list[str],
        instruction: str | None = None,
    ) -> list[str]:
        """Format multiple documents for the same query.

        Args:
            query: The search query
            documents: List of documents to evaluate
            instruction: Optional instruction override

        Returns:
            List of complete prompt strings
        """
        instr = instruction if instruction is not None else self.default_instruction
        return [self.templates.format_pair(instr, query, doc) for doc in documents]

    def format_content_only(
        self,
        query: str,
        doc: str,
        instruction: str | None = None,
    ) -> str:
        """Format content only (without prefix/suffix) for tokenization.

        When using RerankerTokenizer, prefix/suffix are pre-tokenized,
        so we only need the content portion formatted.

        Args:
            query: The search query
            doc: The document to evaluate
            instruction: Optional instruction override

        Returns:
            Content string (without prefix/suffix)
        """
        instr = instruction if instruction is not None else self.default_instruction
        return self.templates.query_template.format(
            instruction=instr,
            query=query,
            doc=doc,
        )

    @property
    def prefix(self) -> str:
        """Get the prefix template."""
        return self.templates.prefix

    @property
    def suffix(self) -> str:
        """Get the suffix template."""
        return self.templates.suffix
