from __future__ import annotations

import json
import logging
from typing import Any, Callable, Protocol

from constants import MAX_ERROR_DATA_LENGTH, MAX_SAMPLE_DATA_LENGTH

logger = logging.getLogger(__name__)


class DataParser(Protocol):
    """
    A protocol for classes that can parse raw tool output into a format
    that is more suitable for an LLM.
    """

    def parse(self, data: Any) -> str | dict[str, Any]:
        """Parses the raw data and returns a string or a dictionary."""
        ...


class ElasticsearchJSONParser:
    """
    An example parser for Elasticsearch-like JSON responses.
    """

    def parse(self, data: str | dict[str, Any]) -> str:
        """
        Parses a JSON response that looks like an Elasticsearch result,
        summarizing the number of hits and showing the source of the first hit.

        Args:
            data: Either a JSON string or already-parsed dictionary

        Returns:
            A human-readable summary of the Elasticsearch results
        """
        logger.debug("Parsing Elasticsearch JSON response")

        if isinstance(data, str):
            try:
                # The initial payload is often a string representation of a list of content blocks
                # We need to handle this structure. Let's assume the first block is the JSON.
                if data.startswith("["):
                    parsed_list = json.loads(data)
                    if parsed_list:
                        data = parsed_list[0]
                        logger.debug("Extracted first element from list")
                else:
                    data = json.loads(data)
                logger.debug("Successfully parsed JSON string")
            except (json.JSONDecodeError, IndexError) as e:
                # If it's not JSON or not a list, return the original data
                logger.error("Failed to parse data as JSON: %s", e)
                error_sample = str(data)[:MAX_ERROR_DATA_LENGTH]
                if len(str(data)) > MAX_ERROR_DATA_LENGTH:
                    error_sample += "..."
                return f"Could not parse data as JSON: {error_sample}"

        if not isinstance(data, dict):
            logger.error("Data is not a dictionary: %s", type(data))
            error_sample = str(data)[:MAX_ERROR_DATA_LENGTH]
            if len(str(data)) > MAX_ERROR_DATA_LENGTH:
                error_sample += "..."
            return f"Data is not in the expected dictionary format: {error_sample}"

        hits_data = data.get("hits", {})
        total_hits = hits_data.get("total", {}).get("value", 0)
        hits = hits_data.get("hits", [])

        logger.info("Parsed Elasticsearch response: %d total hits", total_hits)

        summary = f"Found {total_hits} results."

        if not hits:
            logger.debug("No hits in response")
            return f"{summary} No results to display."

        first_hit_source = hits[0].get("_source", {})
        summary += "\n--- First Hit Source (Summary) ---\n"
        summary += json.dumps(first_hit_source, indent=2)

        logger.debug("Generated summary for %d hit(s)", len(hits))
        return summary


# --- Parser Registry ---

ParserFunc = Callable[[Any], str | dict[str, Any]]

# Instantiate our parsers
elasticsearch_parser = ElasticsearchJSONParser()

# A registry mapping tool name prefixes to parser functions.
# This allows us to catch various elastic tools like "elastic_search_logs", "elastic_metrics", etc.
PARSER_REGISTRY: dict[str, ParserFunc] = {
    "elastic": elasticsearch_parser.parse,
}


def get_parser(tool_name: str) -> ParserFunc | None:
    """
    Gets the parser for a given tool name by checking if the tool name
    starts with a registered prefix.

    Args:
        tool_name: Name of the tool to find a parser for

    Returns:
        Parser function if found, None otherwise
    """
    for prefix, parser_func in PARSER_REGISTRY.items():
        if tool_name.startswith(prefix):
            logger.debug("Found parser for tool '%s' using prefix '%s'", tool_name, prefix)
            return parser_func
    logger.debug("No parser found for tool '%s'", tool_name)
    return None
