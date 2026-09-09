import json
import copy
import time

from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

from importlib import resources

#=============================================================
# Tasks for simple pose navigation
#=============================================================
# 1. Load the original JSON data
# TEST_FILE = "/workspaces/DuTY_agent/duty_agent/dataset/eval_mod_dataset.json"
TEST_FILE = "/workspaces/agentic_framework/dataset/test_set.json"

with open(TEST_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# System prompt — improved to prevent tool-calling loops on small models (4B–9B)
def sys_prompt():
    instruction_xml = (
        "You are a Behavior Tree XML generator. Complete these steps IN ORDER. "
        "Do exactly one step at a time. Do NOT skip steps. Do NOT repeat steps.\n\n"

        "STEP 1 — Discover nodes: Call get_nodes_name() to see all available behavior tree nodes.\n\n"

        "STEP 2 — Learn syntax: Call get_nodes_info() with the node names you need for this task. "
        "You MUST include at least: Sequence, NavToPose, SetBlackboard, "
        "plus any control nodes mentioned in the task (RetryUntilSuccessful, Fallback, etc.).\n\n"

        "STEP 3 — Compose and output: Wrap everything in the root structure below. Output the final XML and STOP.\n\n"

        "ROOT STRUCTURE (use EXACTLY):\n"
        '<root main_tree_to_execute="MainTree">\n'
        '    <BehaviorTree ID="MainTree">\n'
        "        ...your composed tree...\n"
        "    </BehaviorTree>\n"
        "</root>\n\n"

        "CRITICAL STOP RULES:\n"
        "- After step 3, output your answer and STOP. Do NOT call any more tools.\n"
        "- Never call the same tool twice with the same parameters.\n"
        "- Once you have the nodes information, you are DONE. Generate the output.\n"
        "- Output ONLY valid raw XML. No markdown fences (no ```xml), no explanations, no extra text.\n"
        "- Limit yourself to exactly 4 tool calls maximum per task.\n"
    )
    return instruction_xml

# Mcp tools config in dictionary format
mcp_tool_config = {
        "behavior_tree_node_info" : {
            "transport": "stdio",
            "command": "python3",
            "args": ["/workspaces/agentic_framework/src/mcp_tools/btnode_info_server_2tools.py"]
        },
    }

async def init_agent(
        model_name: str, 
        temp: float,
        reasoning: bool,
        tools_config
        ):

    # Setup model api
    llm = ChatOllama(model=model_name,
                    temperature=temp,
                    num_ctx=16384,     # Increase the context size, default is 4096
                    reasoning=reasoning   # Use thinking mode
                    )

    # Setup mcp tools
    client = MultiServerMCPClient(tools_config)
    tools = await client.get_tools()

    # Create agent with current llm and tools
    agent = create_agent(
                llm, 
                tools=tools,
                system_prompt=sys_prompt()
            )

    return agent


# Task specific navigation
async def bt_generation(
        agent,
        output_file_path
        ):

    results_list = []
    for idx, task in enumerate(data, start=1):
        generation_result = {}

        question = task["input"]
        generation_result["input"] = question

        # Invoke the model with single input
        start_time = time.perf_counter()
        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": question}]}
            )
        except Exception as e:
            result = f"Error generating Behavior Tree: {e}"
        else:
            result = result["messages"][-1].content     # Manually extract the output since no chattemplate is provided.
        elapsed_time = time.perf_counter() - start_time

        generation_result["elapsed_time"] = f"{elapsed_time:.4f}"    # Record the total time taken to generate an output
        
        # Output processing
        result = remove_think_blocks(result)
        status, xml_output = check_xml_completeness(result)

        # Check if the syntax is correct.
        if status == True:
            print("Corrected BT format: ", xml_output)
            generation_result["output"] = xml_output
        else:
            print("Failed to parse bt in correct xml.")
            generation_result["output"] = result

        results_list.append(generation_result)

    # 3. Save any remaining items (e.g., if total count is not a multiple of 100)
    with open(output_file_path, "w", encoding="utf-8") as file:
        json.dump(results_list, file, indent=4)
    
    print("Generation complete and file saved successfully.")


if __name__ == "__main__":
    import asyncio
    from agentic_framework.src.utils.utilities import extract_final_answer, remove_think_blocks, check_xml_completeness

    import itertools

    # Uncomment the models to test
    # "reasoning" flag is redundant, please ignore it.
    test_specs = [
        {
            "model": "granite4:3b",
            "temp": [0.0, 0.2, 0.4, 0.6, 0.8],
            "reasoning": [False]
        },
        # {
        #     "model": "gemma4:12b",
        #     "temp": [0.0, 0.2, 0.4, 0.6, 0.8],
        #     "reasoning": [False]
        # },
        # {
        #     "model": "glm4-instruct:latest",
        #     "temp": [0.0, 0.2, 0.4, 0.6, 0.8],
        #     "reasoning": [False]
        # },
        # {
        #     "model": "qwen3.5:4b",
        #     "temp": [0.0, 0.2, 0.4, 0.6, 0.8],
        #     "reasoning": [False]
        # }
    ]

    # Create a lock (semaphore) for each model so it only handles 1 task at a time
    model_locks = {spec["model"]: asyncio.Semaphore(1) for spec in test_specs}

    async def bt_gen_wrapper(model, temp, reasoning):
        """
        Acquires the lock for the specific model before running the task.
        This guarantees Qwen and Gemma run at the same time, but never 
        two Qwens or two Gemmas simultaneously.
        """
        async with model_locks[model]:
            print(f"[START] {model} | Temp: {temp} | Reasoning: {reasoning}")
            
            agent = await init_agent(
                model_name=model,
                temp=temp,
                reasoning=reasoning,
                tools_config=mcp_tool_config
            )

            think = "thinking" if reasoning else ""
            await bt_generation(
                agent=agent,
                output_file_path=f"/workspaces/result/result-2tools/{model}_{temp}_{think}.json"
            )
            
            print(f"[FINISH] {model} | Temp: {temp} | Reasoning: {reasoning}")

    async def main():
        tasks = []
        
        # Build all combinations
        for spec in test_specs:
            model = spec["model"]
            for temp, reasoning in itertools.product(spec["temp"], spec["reasoning"]):
                # Pass to our lock-managed wrapper
                tasks.append(bt_gen_wrapper(model, temp, reasoning))
                
        # Fire everything off in parallel!
        await asyncio.gather(*tasks)

    asyncio.run(main=main())