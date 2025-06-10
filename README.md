# Deep-Search-AI-Agent

This project implements an AI research agent agent using the LangChain framework, OpenAI GPT-4o, and Pydantic for structured output parsing. The agent is designed to interactively handle user queries by leveraging external tools such as web search and Wikipedia, and then store the output in a text file.

The agent is prompt-driven via LangChain’s ChatPromptTemplate, and responses are validated and structured using a Pydantic model called ResearchResponse, which includes:

- topic: The main subject of the research

- summary: A concise summary of findings

- sources: A list of sources consulted or referenced

- tools_used: A list of tools the agent invoked during the process
