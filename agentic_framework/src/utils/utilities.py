from langchain.messages import AIMessage
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Tuple
import json

def extract_final_answer(result: dict) -> str:
    """
    Extract only the final natural-language answer from a LangChain agent result.
    
    The function:
    - scans result["messages"]
    - picks the last AIMessage that has no tool_calls
    - returns its content
    """
    msgs = result.get("messages", [])

    # Traverse from the end (final messages first)
    for m in reversed(msgs):
        if isinstance(m, AIMessage) and not m.tool_calls:
            return m.content.strip()

    # Fallback: if no match, return empty string
    return ""

def remove_think_blocks(text: str) -> str:
    """
    Remove any <think>...</think> blocks from a string.
    Returns the cleaned text.
    """
    # Remove all <think>...</think> blocks, including newlines
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    
    # Strip leading/trailing whitespace
    return cleaned.strip()

def read_file(file_path: str) -> str:
    """Utility function to read a text file and return its content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return str(content)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def fix_xml_line(s: str) -> str:
    stripped = s.rstrip()
    if not stripped.endswith(">"):
        return stripped + ">\n"
    return s

def categorize_plugins(plugin_json_string: str):
    plugins = json.loads(plugin_json_string)

    return (
        [k for k, v in plugins.items() if v == "Action"],
        [k for k, v in plugins.items() if v == "Decorator"],
        [k for k, v in plugins.items() if v == "Control"],
    )

def json_to_xml(json_bt: dict, plugins: list[str]) -> str:
    """
    Convert a JSON behavior tree (dict) into BT.CPP XML format.
    
    :param json_bt: JSON dict describing the behavior tree
    :param plugins: list of valid leaf node plugin names
    :return: XML string
    """

    def iterate(node: dict) -> str:
        # Root node
        if "root" in node:
            return (
                '<root BTCPP_format="3">\n'
                '<BehaviorTree ID="MultiGoalNavigate">\n'
                + iterate(node["root"])
                + '\n</BehaviorTree>\n</root>'
            )

        # Control node (has children)
        if "children" in node:
            node_type = node["type"]
            node_name = node.get("name", "")

            xml = f'<{node_type}'
            if node_name:
                xml += f' name="{node_name}"'
            xml += ">\n"

            for child in node["children"]:
                xml += iterate(child)

            xml += f'</{node_type}>\n'
            return xml

        # Leaf/action node
        if "type" in node and "name" in node and node["type"] in plugins:
            key_list = [key for key in node.keys() if key not in ["type", "name"]]
            value_list = [node[key] for key in key_list]
            add_params = " ".join(f'{k}="{{{v}}}"' for k, v in zip(key_list, value_list))
            xml = f'<{node["type"]} name="{node["name"]}" {add_params} />\n'
            return xml
        # for key, value in node.items():
        #     if key in plugins:
        #         xml = f'<{key} '
        #         for param_key, param_value in value.items():
        #             xml += f'{param_key}="{param_value}" '
        #         xml = xml.strip() + " />\n"
        #         return xml

        # If nothing matched
        raise ValueError(f"Invalid node format: {node}")

    # Add XML header
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + iterate(json_bt)    

def extract_and_parse_xml(raw_llm_output: str) -> tuple[bool, str]:
    # Use a greedy match to get everything from the first '<' to the last '>'
    # The [^<]* allows for potential leading text/whitespace
    match = re.search(r'(<.*>).*', raw_llm_output, re.DOTALL)
    
    if not match:
        err_msg = "No XML-like structure found in the LLM output."
        return False, err_msg
    
    xml_content = match.group(1).strip()

    try:
        ET.fromstring(xml_content)
        return True, xml_content
    except ET.ParseError as e:
        # Cast 'e' to str to avoid TypeError
        err_msg = f"Unstructured XML generated: {str(e)}"
        print(err_msg)
        return False, err_msg

def check_xml_completeness(xml_string: str) -> tuple[bool, str]:
    if not xml_string.strip():
        err_msg = "Empty XML string generated."
        return False, err_msg
    
    return extract_and_parse_xml(xml_string)
    
def escape_curly_braces(s: str, number: int = 2) -> str:
    """Utility function to replace curly braces with escaped versions for XML attributes."""
    return s.replace("{", "{" * number).replace("}", "}" * number)


import logging

def logging_format():
    
    class CustomFormatter(logging.Formatter):
        """Subclassing logging.Formatter to add color based on level."""
        grey = "\x1b[38;20m"
        yellow = "\x1b[33;20m"
        red = "\x1b[31;20m"
        bold_red = "\x1b[31;1m"
        reset = "\x1b[0m"
        fmt = "[%(levelname)s] - %(message)s"

        FORMATS = {
            logging.DEBUG: grey + fmt + reset,
            logging.INFO: grey + fmt + reset,
            logging.WARNING: yellow + fmt + reset,
            logging.ERROR: red + fmt + reset,
            logging.CRITICAL: bold_red + fmt + reset
        }

        def format(self, record):
            log_fmt = self.FORMATS.get(record.levelno)
            formatter = logging.Formatter(log_fmt)
            return formatter.format(record)

    # Usage
    logger = logging.getLogger()
    h = logging.StreamHandler()
    h.setFormatter(CustomFormatter())
    logger.addHandler(h)
    
    return logger