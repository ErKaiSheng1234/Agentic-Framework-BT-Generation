from datetime import datetime
import pytz
from mcp.server.fastmcp import FastMCP
from typing import Any, Dict, Optional
import logging
import os

import asyncio
import websockets
import json
import textwrap

# Initialize FastMCP server
mcp = FastMCP("behavior_tree_node_info")

# Define path using resources to get the path relative to current package (duty_agent)
from importlib import resources
# SCHEMA_PATH = resources.files('agentic_framework').joinpath('src/mcp_tools/info/bt_node_schema_w_example.json')
SCHEMA_PATH = '/workspaces/agentic_framework/src/mcp_tools/info/bt_node_schema_w_example.json'

def load_bt_schema():
    try:
        with open(SCHEMA_PATH, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Failed to load BT schema: {e}")
        return {}

# Load once at startup
node_schema = load_bt_schema()

@mcp.tool()
async def get_nodes_name() -> dict:
    """
    Returns a mapping of available Behavior Tree node names to their execution types.
    Example: {"NavToPose": "Action", "Fallback": "Control"}
    """
    if not node_schema:
        return {"error": "Node schema is empty or could not be loaded."}

    logging.info("Fetching BT node names.")
    
    # Using a dictionary comprehension for cleaner syntax
    return {name: data.get("type", "Unknown") for name, data in node_schema.items()}

@mcp.tool()
async def get_nodes_info(node_list: list[str]) -> dict:
    """
    Returns the schema of provided nodes name for Behavior Tree construction in dictionary type.
    Information includes port info (input/output, name, type), description, and XML examples.
    
    Args:
        node_list: A list of node type names (e.g., ["SetBlackboard", "NavToPose"]) 
                   to retrieve information for.
    """
    if not node_schema:
        return {"error": "Node schema is empty or could not be loaded."}
    
    logging.info(f"Fetching schema for nodes: {node_list}")

    result = {}
    missing = []

    for node in node_list:
        if node in node_schema:
            result[node] = node_schema[node]
        else:
            missing.append(node)

    # logging.info(f"Fetched schema: {result}")

    if missing:
        logging.warning(f"Requested nodes not found in schema: {missing}")
        result["metadata"] = {"nodes not found": missing}

    return result


# @mcp.tool()
# async def get_navtopose_node_xml(targets: Dict[str, str]) -> Dict[str, str]:
#     """
#     Generates Behavior Tree XML snippets for navigation goals using the NavToPose node.

#     For goals with coordinates, it automatically generates a preceding <SetBlackboard> 
#     node to store the coordinate string into the specified goal key before executing <NavToPose>.

#     Args:
#         targets: A mapping of goal names to their coordinate string values.
#                  - Key (str): The goal name (e.g., "kitchen", "charging_station").
#                  - Value (str): The coordinate string (e.g., "1.2;3.4;0.0"). 
#                                 If coordinates are unknown or managed elsewhere, pass an empty string "".

#     Returns:
#         Dict[str, str]: A dictionary mapping each goal name to its formatted XML string snippet.
    
#     Examples of expected input:
#         targets = {
#             "kitchen": "1.5;2.0;0.0",
#             "hallway": ""
#         }
#     """
#     logging.info(f"Getting XML snippet for: {targets}")

#     if not isinstance(targets, dict):
#         return {"error": "Input 'targets' must be a dictionary mapping goal names (str) to coordinate strings (str)."}

#     all_xml = {}
#     for goal, coord in targets.items():
#         goal_clean = str(goal).strip()
#         coord_clean = str(coord).strip() if coord else ""

#         if not coord_clean:
#             xml_output = f'<NavToPose goal="{{{goal_clean}}}" />'
#         else:
#             xml_output = textwrap.dedent(f"""\
#                 <SetBlackboard output_key="{goal_clean}" value="{coord_clean}" />
#                 <NavToPose goal="{{{goal_clean}}}" />""").strip()

#         all_xml[goal_clean] = xml_output

#     logging.info(f"XML snippet: {all_xml}")
#     return all_xml

def main():
    # Run MCP server over stdio
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
