import os
from pydantic import BaseModel, Field
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig


class Configuration(BaseModel):
    """The configuration for the agent."""

    # Translation-specific models
    translation_model: str = Field(
        default="gemini-1.5-flash",
        description="The name of the language model to use for translation.",
    )

    memory_search_model: str = Field(
        default="gemini-1.5-flash",
        description="The name of the language model to use for memory search queries.",
    )

    memory_update_model: str = Field(
        default="gemini-1.5-flash",
        description="The name of the language model to use for memory update decisions.",
    )

    context_summary_model: str = Field(
        default="gemini-1.5-flash",
        description="The name of the language model to use for context summarization.",
    )

    max_translation_attempts: int = Field(
        default=3,
        description="Maximum number of retry attempts for failed translations.",
    )

    max_memory_context_items: int = Field(
        default=5, description="Maximum number of recent context items to maintain."
    )

    memory_search_limit: int = Field(
        default=5, description="Maximum number of memory nodes to retrieve per search."
    )

    # Legacy web research models (kept for backward compatibility)
    query_generator_model: str = Field(
        default="gemini-2.0-flash",
        description="The name of the language model to use for the agent's query generation.",
    )

    reflection_model: str = Field(
        default="gemini-2.5-flash",
        description="The name of the language model to use for the agent's reflection.",
    )

    # answer_model: str = Field(
    #     default="gemini-2.5-pro",
    #     description="The name of the language model to use for the agent's answer.",
    # )

    # number_of_initial_queries: int = Field(
    #     default=3, description="The number of initial search queries to generate."
    # )

    # max_research_loops: int = Field(
    #     default=2, description="The maximum number of research loops to perform."
    # )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)
