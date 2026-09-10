# Agentic-Framework-BT-Generation
An agentic framework to generate Behavior Tree with correct custom node usage.

## Setup
Host machine software dependencies:
1. Docker
2. Vscode (For devcontainer setup)
3. Nvidia-container-toolkit (For nvidia gpu access in docker container)

## File structure
1. agentic_framework/dataset: The folder where the train_set and test_set is stored.
2. agentic_framework/src: The major algorithm for agentic framework.
3. agentic_framework/src/mcp_tools: The mcp server for the agent in this research. There are two type of servers -- 2-tools configuration server and 3-tools configuration server.
4. agentic_framework/src/script: The major script of (1) create the agent (2) collect the result and store in a json file.
5. agentic_framework/src/utils: Utilities functions required for this research.
