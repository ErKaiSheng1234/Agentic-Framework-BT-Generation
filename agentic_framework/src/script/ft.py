import subprocess
import time
import requests
import signal
import sys
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import json

# For image input
import base64


model_name = 'granite4-3b-ft-bt-q4:latest'
test_set_path = '/workspaces/agentic_framework/dataset/test_set.json'
test_temp = [0.0, 0.2, 0.4, 0.6, 0.8]

def unload_ollama_model(model_name=model_name):
    try:
        subprocess.run(["ollama", "stop", model_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to stop model {model_name}: {e}")

def signal_handler(sig, frame):
    print("\nKeyboardInterrupt detected. Attempting to unload model...")
    unload_ollama_model()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Ensure model is pulled
def check_model(model_name: str):
    # Manually start ollama server
    subprocess.Popen(['ollama', 'serve'])
    
    # Give it a moment to start
    time.sleep(2)
    
    try:
        subprocess.run(["ollama", "pull", model_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to pull model {model_name}: {e}")

def input_prompt(llm, text_input: str):
    # human message without img input
    human_content = [{"type": "text", "text": text_input}]

    messages = [
        SystemMessage(
            content=[
                {"type": "text", 
                 "text": """\nYou are a helpful assistant that can assist with creating behavior trees.\nYour task:\n- Convert the provided summary of a behavior into an XML-formatted behavior tree.\n- Ensure the behavior tree matches the description in the summary.\n- The behavior tree must be compatible with the BehaviorTree.CPP library.\n- Only use the actions and parameters provided in the action list below the summary.\n\nOutput Requirements:\n- Output only the XML representation of the behavior tree. Do not include explanations, comments, or any additional text.\n- Ensure all actions and parameters strictly match the provided list.\n- If possible limit the use of SubTrees.\n\nPlease generate the behavior tree based on the summary and action list provided.\n"""
                }
            ]
        ),
        HumanMessage(
            human_content
        )
    ]
    
    ai_msg = llm.invoke(messages)
    
    return ai_msg.content
    
if __name__ == '__main__':

    with open(test_set_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for temp in test_temp:

        llm = ChatOllama(model=model_name, 
                        num_ctx=2048,     # No tool calling
                        temperature=temp,
                        reasoning=False   # No reasoning since it is fine-tuned without cot.
                        )

        print(f"[START] {model_name} | Temp: {temp} | Reasoning: False")
        results = []

        for item in data:
            single_result = {}

            question = item["input"]

            start_time = time.perf_counter()
            llm_output = input_prompt(llm, str(question))
            elapsed_time = time.perf_counter() - start_time

            single_result["elapsed_time"] = f"{elapsed_time:.4f}"

            single_result["input"] = question
            single_result["output"] = llm_output

            results.append(single_result)

            print("----------------------------------------------")
            print(f"{llm_output}")

        with open(f"/workspaces/result/{model_name}_{temp}-ft.json", "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4)