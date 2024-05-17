#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: cbk914

import os
import time
import logging
from collections import deque
from typing import Dict, List
import importlib
import openai
import chromadb
import tiktoken as tiktoken
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
import re
from dotenv import load_dotenv

# Load default environment variables (.env)
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Default opt-out of ChromaDB telemetry
from chromadb.config import Settings
client = chromadb.Client(Settings(anonymized_telemetry=False))

# Engine configuration
LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("OPENAI_API_MODEL", "gpt-3.5-turbo")).lower()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not (LLM_MODEL.startswith("llama") or LLM_MODEL.startswith("human")):
    assert OPENAI_API_KEY, "OPENAI_API_KEY environment variable is missing from .env"

# Table config
RESULTS_STORE_NAME = os.getenv("RESULTS_STORE_NAME", os.getenv("TABLE_NAME", ""))
assert RESULTS_STORE_NAME, "RESULTS_STORE_NAME environment variable is missing from .env"

# Run configuration
INSTANCE_NAME = os.getenv("INSTANCE_NAME", os.getenv("BABY_NAME", "BabyAGI"))
COOPERATIVE_MODE = "none"
JOIN_EXISTING_OBJECTIVE = False

# Goal configuration
OBJECTIVE = os.getenv("OBJECTIVE", "")
INITIAL_TASK = os.getenv("INITIAL_TASK", os.getenv("FIRST_TASK", ""))

# Model configuration
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.0))

def can_import(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

DOTENV_EXTENSIONS = os.getenv("DOTENV_EXTENSIONS", "").split(" ")

# Command line arguments extension
ENABLE_COMMAND_LINE_ARGS = os.getenv("ENABLE_COMMAND_LINE_ARGS", "false").lower() == "true"
if ENABLE_COMMAND_LINE_ARGS:
    if can_import("extensions.argparseext"):
        from extensions.argparseext import parse_arguments
        OBJECTIVE, INITIAL_TASK, LLM_MODEL, DOTENV_EXTENSIONS, INSTANCE_NAME, COOPERATIVE_MODE, JOIN_EXISTING_OBJECTIVE = parse_arguments()

# Human mode extension
if LLM_MODEL.startswith("human"):
    if can_import("extensions.human_mode"):
        from extensions.human_mode import user_input_await

# Load additional environment variables for enabled extensions
if DOTENV_EXTENSIONS:
    if can_import("extensions.dotenvext"):
        from extensions.dotenvext import load_dotenv_extensions
        load_dotenv_extensions(DOTENV_EXTENSIONS)

print(f"\n*****CONFIGURATION*****\nName  : {INSTANCE_NAME}\nMode  : {'alone' if COOPERATIVE_MODE in ['n', 'none'] else 'local' if COOPERATIVE_MODE in ['l', 'local'] else 'distributed' if COOPERATIVE_MODE in ['d', 'distributed'] else 'undefined'}\nLLM   : {LLM_MODEL}")

assert OBJECTIVE, "OBJECTIVE environment variable is missing from .env"
assert INITIAL_TASK, "INITIAL_TASK environment variable is missing from .env"

LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "models/llama-13B/ggml-model.bin")
if LLM_MODEL.startswith("llama"):
    if can_import("llama_cpp"):
        from llama_cpp import Llama

        assert os.path.exists(LLAMA_MODEL_PATH), "Model can't be found."

        CTX_MAX = 1024
        LLAMA_THREADS_NUM = int(os.getenv("LLAMA_THREADS_NUM", 8))

        llm = Llama(model_path=LLAMA_MODEL_PATH, n_ctx=CTX_MAX, n_threads=LLAMA_THREADS_NUM, n_batch=512, use_mlock=False)
        llm_embed = Llama(model_path=LLAMA_MODEL_PATH, n_ctx=CTX_MAX, n_threads=LLAMA_THREADS_NUM, n_batch=512, embedding=True, use_mlock=False)
        print("\n*****USING LLAMA.CPP. POTENTIALLY SLOW.*****")
    else:
        print("\nLlama LLM requires package llama-cpp. Falling back to GPT-3.5-turbo.")
        LLM_MODEL = "gpt-3.5-turbo"

if LLM_MODEL.startswith("gpt-4"):
    print("\n*****USING GPT-4. POTENTIALLY EXPENSIVE. MONITOR YOUR COSTS*****")

if LLM_MODEL.startswith("human"):
    print("\n*****USING HUMAN INPUT*****")

print(f"\n*****OBJECTIVE*****\n{OBJECTIVE}")

if not JOIN_EXISTING_OBJECTIVE:
    print(f"\nInitial task: {INITIAL_TASK}")
else:
    print(f"\nJoining to help the objective")

openai.api_key = OPENAI_API_KEY

class LlamaEmbeddingFunction(EmbeddingFunction):
    def __call__(self, texts: Documents) -> Embeddings:
        return [llm_embed.embed(t) for t in texts]

class DefaultResultsStorage:
    def __init__(self):
        logging.getLogger('chromadb').setLevel(logging.ERROR)
        chroma_persist_dir = "chroma"
        chroma_client = chromadb.PersistentClient(
            settings=chromadb.config.Settings(persist_directory=chroma_persist_dir)
        )
        metric = "cosine"
        if LLM_MODEL.startswith("llama"):
            embedding_function = LlamaEmbeddingFunction()
        else:
            embedding_function = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY)
        self.collection = chroma_client.get_or_create_collection(
            name=RESULTS_STORE_NAME,
            metadata={"hnsw:space": metric},
            embedding_function=embedding_function,
        )

    def add(self, task: Dict, result: str, result_id: str):
        if LLM_MODEL.startswith("human"):
            return

        embeddings = llm_embed.embed(result) if LLM_MODEL.startswith("llama") else None
        if len(self.collection.get(ids=[result_id], include=[])["ids"]) > 0:
            self.collection.update(
                ids=result_id,
                embeddings=embeddings,
                documents=result,
                metadatas={"task": task["task_name"], "result": result},
            )
        else:
            self.collection.add(
                ids=result_id,
                embeddings=embeddings,
                documents=result,
                metadatas={"task": task["task_name"], "result": result},
            )

    def query(self, query: str, top_results_num: int) -> List[dict]:
        count: int = self.collection.count()
        if count == 0:
            return []
        results = self.collection.query(
            query_texts=query,
            n_results=min(top_results_num, count),
            include=["metadatas"]
        )
        return [item["task"] for item in results["metadatas"][0]]

def try_weaviate():
    WEAVIATE_URL = os.getenv("WEAVIATE_URL", "")
    WEAVIATE_USE_EMBEDDED = os.getenv("WEAVIATE_USE_EMBEDDED", "False").lower() == "true"
    if (WEAVIATE_URL or WEAVIATE_USE_EMBEDDED) and can_import("extensions.weaviate_storage"):
        WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")
        from extensions.weaviate_storage import WeaviateResultsStorage
        print("\nUsing results storage: Weaviate")
        return WeaviateResultsStorage(OPENAI_API_KEY, WEAVIATE_URL, WEAVIATE_API_KEY, WEAVIATE_USE_EMBEDDED, LLM_MODEL, LLAMA_MODEL_PATH, RESULTS_STORE_NAME, OBJECTIVE)
    return None

def try_pinecone():
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
    if PINECONE_API_KEY and can_import("extensions.pinecone_storage"):
        PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "")
        assert PINECONE_ENVIRONMENT, "PINECONE_ENVIRONMENT environment variable is missing from .env"
        from extensions.pinecone_storage import PineconeResultsStorage
        print("\nUsing results storage: Pinecone")
        return PineconeResultsStorage(OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, LLM_MODEL, LLAMA_MODEL_PATH, RESULTS_STORE_NAME, OBJECTIVE)
    return None

def use_chroma():
    print("\nUsing results storage: Chroma (Default)")
    return DefaultResultsStorage()

results_storage = try_weaviate() or try_pinecone() or use_chroma()

class SingleTaskListStorage:
    def __init__(self):
        self.tasks = deque([])
        self.task_id_counter = 0

    def append(self, task: Dict):
        self.tasks.append(task)

    def replace(self, tasks: List[Dict]):
        self.tasks = deque(tasks)

    def popleft(self):
        return self.tasks.popleft()

    def is_empty(self):
        return False if self.tasks else True

    def next_task_id(self):
        self.task_id_counter += 1
        return self.task_id_counter

    def get_task_names(self):
        return [t["task_name"] for t in self.tasks]

tasks_storage = SingleTaskListStorage()
if COOPERATIVE_MODE in ['l', 'local']:
    if can_import("extensions.ray_tasks"):
        import sys
        from pathlib import Path

        sys.path.append(str(Path(__file__).resolve().parent))
        from extensions.ray_tasks import CooperativeTaskListStorage

        tasks_storage = CooperativeTaskListStorage(OBJECTIVE)
        print("\nReplacing tasks storage: Ray")
elif COOPERATIVE_MODE in ['d', 'distributed']:
    pass

def limit_tokens_from_string(string: str, model: str, limit: int) -> str:
    """Limits the string to a number of tokens (estimated)."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except:
        encoding = tiktoken.encoding_for_model('gpt2')
    encoded = encoding.encode(string)
    return encoding.decode(encoded[:limit])

def openai_call(prompt: str, model: str = LLM_MODEL, temperature: float = OPENAI_TEMPERATURE, max_tokens: int = 100):
    while True:
        try:
            if model.lower().startswith("llama"):
                result = llm(prompt[:CTX_MAX], stop=["### Human"], echo=False, temperature=0.2, top_k=40, top_p=0.95, repeat_penalty=1.05, max_tokens=200)
                return result['choices'][0]['text'].strip()
            elif model.lower().startswith("human"):
                return user_input_await(prompt)
            elif not model.lower().startswith("gpt-"):
                response = openai.Completion.create(
                    engine=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                return response.choices[0].text.strip()
            else:
                trimmed_prompt = limit_tokens_from_string(prompt, model, 4000 - max_tokens)
                messages = [{"role": "system", "content": trimmed_prompt}]
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=1,
                    stop=None,
                )
                return response.choices[0].message.content.strip()
        except openai.error.RateLimitError:
            logging.warning("The OpenAI API rate limit has been exceeded. Waiting 10 seconds and trying again.")
            time.sleep(10)
        except openai.error.Timeout:
            logging.warning("OpenAI API timeout occurred. Waiting 10 seconds and trying again.")
            time.sleep(10)
        except openai.error.APIError:
            logging.warning("OpenAI API error occurred. Waiting 10 seconds and trying again.")
            time.sleep(10)
        except openai.error.APIConnectionError:
            logging.warning("OpenAI API connection error occurred. Check your network settings, proxy configuration, SSL certificates, or firewall rules. Waiting 10 seconds and trying again.")
            time.sleep(10)
        except openai.error.InvalidRequestError:
            logging.warning("OpenAI API invalid request. Check the documentation for the specific API method you are calling and make sure you are sending valid and complete parameters. Waiting 10 seconds and trying again.")
            time.sleep(10)
        except openai.error.ServiceUnavailableError:
            logging.warning("OpenAI API service unavailable. Waiting 10 seconds and trying again.")
            time.sleep(10)
        else:
            break

def task_creation_agent(objective: str, result: Dict, task_description: str, task_list: List[str]):
    prompt = f"""
You are to use the result from an execution agent to create new tasks with the following objective: {objective}.
The last completed task has the result: \n{result["data"]}
This result was based on this task description: {task_description}.\n"""

    if task_list:
        prompt += f"These are incomplete tasks: {', '.join(task_list)}\n"
    prompt += "Based on the result, return a list of tasks to be completed in order to meet the objective. "
    if task_list:
        prompt += "These new tasks must not overlap with incomplete tasks. "

    prompt += """
Return one task per line in your response. The result must be a numbered list in the format:

#. First task
#. Second task

The number of each entry must be followed by a period. If your list is empty, write "There are no tasks to add at this time."
Unless your list is empty, do not include any headers before your numbered list or follow your numbered list with any other output."""

    logging.info(f'TASK CREATION AGENT PROMPT: {prompt}')
    response = openai_call(prompt, max_tokens=2000)
    logging.info(f'TASK CREATION AGENT RESPONSE: {response}')
    new_tasks = response.split('\n')
    new_tasks_list = []
    for task_string in new_tasks:
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = ''.join(s for s in task_parts[0] if s.isnumeric())
            task_name = re.sub(r'[^\w\s_]+', '', task_parts[1]).strip()
            if task_name.strip() and task_id.isnumeric():
                new_tasks_list.append(task_name)

    return [{"task_name": task_name} for task_name in new_tasks_list]

def prioritization_agent():
    task_names = tasks_storage.get_task_names()
    bullet_string = '\n'

    prompt = f"""
You are tasked with prioritizing the following tasks: {bullet_string + bullet_string.join(task_names)}
Consider the ultimate objective of your team: {OBJECTIVE}.
Tasks should be sorted from highest to lowest priority, where higher-priority tasks are those that act as pre-requisites or are more essential for meeting the objective.
Do not remove any tasks. Return the ranked tasks as a numbered list in the format:

#. First task
#. Second task

The entries must be consecutively numbered, starting with 1. The number of each entry must be followed by a period.
Do not include any headers before your ranked list or follow your list with any other output."""

    logging.info(f'TASK PRIORITIZATION AGENT PROMPT: {prompt}')
    response = openai_call(prompt, max_tokens=2000)
    logging.info(f'TASK PRIORITIZATION AGENT RESPONSE: {response}')
    if not response:
        logging.warning('Received empty response from prioritization agent. Keeping task list unchanged.')
        return
    new_tasks = response.split("\n") if "\n" in response else [response]
    new_tasks_list = []
    for task_string in new_tasks:
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = ''.join(s for s in task_parts[0] if s.isnumeric())
            task_name = re.sub(r'[^\w\s_]+', '', task_parts[1]).strip()
            if task_name.strip():
                new_tasks_list.append({"task_id": task_id, "task_name": task_name})

    return new_tasks_list

def execution_agent(objective: str, task: str) -> str:
    context = context_agent(query=objective, top_results_num=5)
    prompt = f'Perform one task based on the following objective: {objective}.\n'
    if context:
        prompt += 'Take into account these previously completed tasks:' + '\n'.join(context)
    prompt += f'\nYour task: {task}\nResponse:'
    return openai_call(prompt, max_tokens=2000)

def context_agent(query: str, top_results_num: int):
    results = results_storage.query(query=query, top_results_num=top_results_num)
    return results

if not JOIN_EXISTING_OBJECTIVE:
    initial_task = {"task_id": tasks_storage.next_task_id(), "task_name": INITIAL_TASK}
    tasks_storage.append(initial_task)

def main():
    loop = True
    while loop:
        if not tasks_storage.is_empty():
            logging.info("\n*****TASK LIST*****")
            for t in tasks_storage.get_task_names():
                logging.info(f" • {t}")

            task = tasks_storage.popleft()
            logging.info("\n*****NEXT TASK*****")
            logging.info(task["task_name"])

            result = execution_agent(OBJECTIVE, task["task_name"])
            logging.info("\n*****TASK RESULT*****")
            logging.info(result)

            enriched_result = {"data": result}
            result_id = f"result_{task['task_id']}"

            results_storage.add(task, result, result_id)

            new_tasks = task_creation_agent(OBJECTIVE, enriched_result, task["task_name"], tasks_storage.get_task_names())
            logging.info('Adding new tasks to task_storage')
            for new_task in new_tasks:
                new_task.update({"task_id": tasks_storage.next_task_id()})
                logging.info(new_task)
                tasks_storage.append(new_task)

            if not JOIN_EXISTING_OBJECTIVE:
                prioritized_tasks = prioritization_agent()
                if prioritized_tasks:
                    tasks_storage.replace(prioritized_tasks)

            time.sleep(5)
        else:
            logging.info('Done.')
            loop = False

if __name__ == "__main__":
    main()
