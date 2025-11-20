from __future__ import annotations

import json
import logging
from typing import AsyncIterator

from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage

from client_streamable_vertex_ai import MCPChatClient
from utils import ai_content_to_str
from exceptions import AgentError, ParsingError

logger = logging.getLogger(__name__)


class Agent:
    """Base class for a domain-specific agent that can execute complex workflows."""

    def __init__(self, client: MCPChatClient):
        """
        Initialize the agent with an MCP client.

        Args:
            client: An initialized MCPChatClient instance

        Raises:
            AgentError: If client is not initialized
        """
        self._client = client
        if not self._client.is_initialized:
            raise AgentError(
                agent_name=self.__class__.__name__,
                reason="MCPChatClient must be initialized before being used by an agent"
            )
        self._llm = self._client.bound_llm
        logger.info("Initialized agent: %s", self.__class__.__name__)

    async def run(self, query: str) -> AsyncIterator[str]:
        """
        Runs the agent's workflow. This should be implemented by subclasses.

        Args:
            query: The user's query

        Yields:
            Chunks of the agent's response

        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        yield f"Agent received query: {query}"
        raise NotImplementedError(f"{self.__class__.__name__}.run() must be implemented by subclass")


class LogAnalysisAgent(Agent):
    """An agent specialized in analyzing logs from Elasticsearch."""

    async def run(self, query: str) -> AsyncIterator[str]:
        """
        Executes a multi-step workflow to analyze logs.
        1. Clarify query into a structured search query.
        2. Execute the search.
        3. Summarize the results.

        Args:
            query: The user's log analysis query

        Yields:
            Chunks of the agent's response

        Raises:
            AgentError: If the workflow fails at any step
        """
        logger.info("Starting log analysis workflow for query: %s", query[:100])
        messages: list[HumanMessage | BaseMessage | ToolMessage] = [HumanMessage(content=query)]

        # === Step 1: Generate structured query ===
        yield ">>> [Agent Step 1/3] Generating search query from your request...\n"
        logger.debug("Step 1: Generating structured search query")
        clarify_prompt = f"""
Given the user's request, what is the most appropriate JSON object to use as arguments for an 'elastic_search' tool?
The tool is used for searching logs in Elasticsearch.
The tool expects arguments like: {{"query": "some lucene query", "timeframe": "24h", "index": "logs-*"}}.
Base the arguments on the user's request.

User request: "{query}"

Return ONLY the JSON object for the arguments. Do not add any other text or explanation.
"""
        messages.append(HumanMessage(content=clarify_prompt))

        try:
            response: BaseMessage = await self._client.ainvoke_llm(messages)
            messages.append(response)
        except Exception as e:
            logger.error("Failed to generate search query: %s", e)
            error_msg = f">>> [Agent Step 1/3] Failed to generate search query: {str(e)}\n"
            yield error_msg
            raise AgentError(
                agent_name="LogAnalysisAgent",
                step="Step 1: Generate search query",
                reason=str(e)
            ) from e

        # Extract the JSON from the response
        try:
            # The response might have ```json ... ``` markdown, so we clean it.
            json_str = response.content.strip().removeprefix("```json").removesuffix("```").strip()
            search_args = json.loads(json_str)
            logger.debug("Extracted search args: %s", search_args)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse search query JSON: %s", e)
            error_msg = f">>> [Agent Step 1/3] Failed to parse search query. LLM response was not valid JSON.\n"
            yield error_msg
            raise AgentError(
                agent_name="LogAnalysisAgent",
                step="Step 1: Parse search query",
                reason=f"Invalid JSON in LLM response: {json_str[:200]}"
            ) from e

        yield f">>> [Agent Step 1/3] Generated search query: {json.dumps(search_args, indent=2)}\n"
        logger.info("Step 1 complete: Generated search args with %d parameter(s)", len(search_args))

        # === Step 2: Execute the search ===
        yield ">>> [Agent Step 2/3] Executing search via 'elastic_search' tool...\n"
        logger.debug("Step 2: Executing elastic_search tool")

        try:
            # The agent needs to know the tool name. We assume "elastic_search" is available.
            tool_result_message = await self._client.execute_tool("elastic_search", search_args)
            messages.append(tool_result_message)
            logger.info("Step 2 complete: Tool executed successfully, returned %d characters", len(tool_result_message.content))
        except Exception as e:
            logger.error("Failed to execute tool: %s", e)
            error_msg = f">>> [Agent Step 2/3] Failed to execute tool: {str(e)}\n"
            yield error_msg
            raise AgentError(
                agent_name="LogAnalysisAgent",
                step="Step 2: Execute search tool",
                reason=str(e)
            ) from e

        yield f">>> [Agent Step 2/3] Search complete. Result preview:\n{tool_result_message.content[:500]}\n...\n"

        # === Step 3: Summarize the results ===
        yield ">>> [Agent Step 3/3] Summarizing results...\n"
        logger.debug("Step 3: Generating summary")

        summarize_prompt = """
Based on the preceding conversation, which includes the original query and the results from the tool,
provide a comprehensive, human-readable summary of the findings.
If you found errors, highlight them. If you found no results, state that clearly.
"""
        messages.append(HumanMessage(content=summarize_prompt))

        # Stream the final response
        try:
            async for chunk in self._client.astream_llm(messages):
                content = ai_content_to_str(chunk)
                if content:
                    yield content
            logger.info("Step 3 complete: Summary generated successfully")
            logger.info("Log analysis workflow completed successfully")
        except Exception as e:
            logger.error("Failed to generate summary: %s", e)
            error_msg = f"\n>>> [Agent Step 3/3] Failed to generate summary: {str(e)}\n"
            yield error_msg
            raise AgentError(
                agent_name="LogAnalysisAgent",
                step="Step 3: Generate summary",
                reason=str(e)
            ) from e
