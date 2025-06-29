import os
import json
from openai import OpenAI
from tavily import TavilyClient
import re
from mem0 import MemoryClient as Memory
from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.id import ID


# --- INITIALIZE API CLIENTS ---
try:
    appwrite_client = Client()
    (appwrite_client
        .set_endpoint('https://cloud.appwrite.io/v1')
        .set_project(os.environ['APPWRITE_PROJECT_ID'])
        .set_key(os.environ['APPWRITE_API_KEY'])
    )
    databases = Databases(appwrite_client)
    APPWRITE_DATABASE_ID = os.environ['APPWRITE_DATABASE_ID']
    APPWRITE_COLLECTION_ID = os.environ['APPWRITE_COLLECTION_ID_RQRAG']

    openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    tavily_client = TavilyClient(api_key=os.environ['TAVILY_API_KEY'])
    mem0_client = Memory()
except KeyError as e:
    raise RuntimeError(f"Missing environment variable: {e}. Please set it in your Appwrite function settings.")

AGENT_ID = "fact-checker-agent-v3-rq-rag" # Updated agent ID for new memory scope

# --- HELPER FUNCTIONS ---

def call_llm(prompt, system_message="You are a helpful assistant."):
    """Generic function to call the OpenAI API."""
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        response_format={"type": "json_object"}, # Enforce JSON output where needed
    )
    return response.choices[0].message.content

def extract_json_from_llm_response(llm_response_str: str) -> str | None:
    """Extracts a JSON string from an LLM response string."""
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', llm_response_str)
    if match:
        return match.group(1).strip()
    # Fallback for non-markdown responses
    first_brace = llm_response_str.find('{')
    last_brace = llm_response_str.rfind('}')
    if first_brace != -1 and last_brace != -1:
        return llm_response_str[first_brace:last_brace+1]
    return None

# --- CORE EXECUTION & SYNTHESIS LOGIC (MODIFIED WITH MEM0) ---

def execute_and_synthesize(original_prompt, plan, context):
    """
    Executes a given plan of questions, caches results in memory, and synthesizes a final answer.
    """
    context.log(f"Executing plan: {plan}")
    verified_facts = []

    for sub_question in plan:
        context.log(f"Executing step: '{sub_question}'")
        try:
            memories = mem0_client.search(query=sub_question, agent_id=AGENT_ID, limit=1)
            if memories and memories[0].get('score', 0) > 0.7:
                context.log(f"Found high-confidence answer in memory for: '{sub_question}'")
                memory_data = memories[0].get('metadata', {})
                answer = memory_data.get('answer', 'Answer found in memory.')
                source = memory_data.get('source', 'Source from memory.')
                
                verified_facts.append({"question": sub_question, "answer": answer, "source": source})
                continue # Skip the web search and move to the next question
        except Exception as e:
            context.error(f"Error searching memory for '{sub_question}': {e}")

        # If not in memory, proceed with web search
        try:
            search_result = tavily_client.search(query=sub_question, search_depth="basic", include_answer=True)
            if search_result and search_result.get('answer'):
                answer = search_result['answer']
                source = search_result['results'][0]['url'] if search_result.get('results') else "N/A"
                verified_facts.append({"question": sub_question, "answer": answer, "source": source})
                context.log(f"Found answer via web search: '{answer}'")
                
                # Add to memory synchronously
                try:
                    context.log(f"Adding fact to memory for '{sub_question}'")
                    mem0_client.add(
                        messages=[{"role": "assistant", "content": sub_question}],
                        agent_id=AGENT_ID,
                        metadata={'answer': answer, 'source': source}
                    )
                    context.log(f"Successfully added fact for '{sub_question}'")
                except Exception as e:
                    context.error(f"Error adding to memory for '{sub_question}': {e}")

            else:
                verified_facts.append({"question": sub_question, "answer": "Could not find a definitive answer.", "source": "N/A"})
        except Exception as e:
            context.error(f"Error executing step '{sub_question}': {e}")
            verified_facts.append({"question": sub_question, "answer": f"Error during search: {e}", "source": "N/A"})

    context.log("Synthesizing the final response...")
    synthesis_system_message = "You are a helpful AI assistant. Synthesize a final, comprehensive answer to the user's original query based *only* on the provided list of verified facts. For each piece of information you use, you MUST cite its source URL in the format [Source: http://...]."
    facts_summary = "\n".join([
        f"- Question: {fact['question']}\n  Verified Answer: {fact['answer']}\n  Source: {fact['source']}"
        for fact in verified_facts
    ])
    synthesis_prompt = f"""Original User Query: "{original_prompt}"
                    Please construct a final answer using only the following verified facts:
                    ---
                    {facts_summary}
                    ---
                    """
    synthesis_response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": synthesis_system_message},
            {"role": "user", "content": synthesis_prompt},
        ],
        temperature=0.7,
    )
    return synthesis_response.choices[0].message.content


# --- RQ-RAG STRATEGY IMPLEMENTATIONS ---

def plan_and_execute_agent(user_prompt, context):
    """
    Implements the DECOMPOSITION strategy from RQ-RAG[cite: 12, 26].
    This function now focuses only on creating the plan.
    """
    context.log("Creating a decomposition plan...")
    planning_system_message = """You are a world-class planning agent. Your job is to break down a complex user query into a series of simple, verifiable, and independent questions.
    Respond ONLY with a JSON object containing a single key 'plan' which is a list of strings.
    Example:
    User Query: "Compare the height of Mount Everest with K2."
    Your Response: {"plan": ["What is the height of Mount Everest?", "What is the height of K2?"]}
    """
    planning_prompt = f"Decompose the following user query into a plan: '{user_prompt}'"
    
    try:
        llm_response_str = call_llm(planning_prompt, planning_system_message)
        extracted_json_str = extract_json_from_llm_response(llm_response_str)
        plan_data = json.loads(extracted_json_str)
        plan = plan_data.get('plan', [])
        if not plan or not isinstance(plan, list):
            plan = [user_prompt]
    except Exception as e:
        context.error(f"Error during planning phase: {e}. Reverting to simple search.")
        plan = [user_prompt]

    return execute_and_synthesize(user_prompt, plan, context)

def disambiguate_and_execute_agent(user_prompt, context):
    """
    Implements the DISAMBIGUATION strategy from RQ-RAG[cite: 12, 41, 64].
    Handles ambiguous queries by generating and answering multiple interpretations.
    """
    context.log(f"Disambiguating prompt: '{user_prompt}'")
    disambiguation_system_message = """You are a helpful assistant. The user's query is ambiguous.
    Your task is to rewrite it into a JSON list of clear, specific, and distinct questions that cover the most likely interpretations.
    Example:
    User Query: "Most total goals in a premier league season?"
    Your Response: {"plan": ["Most total goals in a premier league season by a team?", "Most total goals in a premier league season by a single player?"]}
    """
    disambiguation_prompt = f"The following query is ambiguous. Generate a JSON list of specific questions to clarify it: '{user_prompt}'"
    
    try:
        llm_response_str = call_llm(disambiguation_prompt, disambiguation_system_message)
        extracted_json_str = extract_json_from_llm_response(llm_response_str)
        plan_data = json.loads(extracted_json_str)
        plan = plan_data.get('plan', [])
        if not plan or not isinstance(plan, list):
            plan = [user_prompt]
    except Exception as e:
        context.error(f"Error during disambiguation phase: {e}. Reverting to simple search.")
        plan = [user_prompt]

    return execute_and_synthesize(user_prompt, plan, context)

# --- RQ-RAG MASTER AGENT (NEW) ---

def select_refinement_strategy(user_prompt, context):
    """
    Uses an LLM to analyze the user_prompt and select the best refinement strategy, as proposed in RQ-RAG[cite: 12, 67].
    """
    context.log("Selecting RQ-RAG strategy...")
    system_message = """You are a query analysis expert. Your task is to classify a user's query and determine the best strategy to find the answer. The strategies are:
    1. 'simple': The query is direct, clear, and can be answered with a single search.
    2. 'decompose': The query is complex, compares multiple things, or requires a sequence of steps to answer.
    3. 'disambiguate': The query is ambiguous and could have multiple valid interpretations.
    Respond ONLY with a JSON object with a single key 'strategy'.
    """
    prompt = f"""Analyze the following user query and choose the best strategy: '{user_prompt}'

    Examples:
    - "When were the 2024 NBA playoffs?" -> {{"strategy": "simple"}}
    - "What kind of university is the school where Rey Ramsey was educated an instance of?" -> {{"strategy": "decompose"}}
    - "Most total goals in a premier league season?" -> {{"strategy": "disambiguate"}}
    """
    try:
        llm_response_str = call_llm(prompt, system_message)
        extracted_json_str = extract_json_from_llm_response(llm_response_str)
        if extracted_json_str:
            strategy = json.loads(extracted_json_str).get('strategy', 'simple')
            context.log(f"LLM selected strategy: '{strategy}'")
            return strategy
    except Exception as e:
        context.error(f"Error selecting strategy: {e}. Defaulting to 'simple'.")
    return 'simple'

def rq_rag_master_agent(user_prompt, context):
    """
    The main agent that implements the full RQ-RAG framework.
    It first selects a refinement strategy and then routes the query to the appropriate handler.
    """
    strategy = select_refinement_strategy(user_prompt, context)

    if strategy == 'decompose':
        return plan_and_execute_agent(user_prompt, context)
    elif strategy == 'disambiguate':
        return disambiguate_and_execute_agent(user_prompt, context)
    else: # 'simple' or fallback
        # For a simple query, the "plan" is just the query itself.
        return execute_and_synthesize(user_prompt, [user_prompt], context)


# --- APPWRITE FUNCTION ENTRYPOINT ---

def main(context):
    """
    Main function executed by Appwrite. It now uses the RQ-RAG master agent.
    """
    try:
        body = json.loads(context.req.body)
        user_prompt = body.get('prompt')
        if not user_prompt:
            return context.res.json({'error': 'Prompt is missing.'}, 400)
    except Exception as e:
        return context.res.json({'error': f'Error parsing request: {e}'}, 400)

    context.log(f"Received prompt: '{user_prompt}'")
    
    # 1. Generate the naive LLM response (direct, unverified baseline)
    context.log("Generating naive LLM response...")
    try:
        # Direct call to OpenAI for a simple text response, not JSON formatted.
        naive_response_obj = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )
        naive_llm_response = naive_response_obj.choices[0].message.content
        context.log("Successfully generated naive LLM response.")
    except Exception as e:
        context.error(f"Error generating naive LLM response: {e}")
        naive_llm_response = f"Error: {e}"

    # 2. Generate the advanced RQ-RAG response
    try:
        rq_rag_response = rq_rag_master_agent(user_prompt, context)
        context.log("RQ-RAG agent finished.")
    except Exception as e:
        context.error(f"An error occurred in the RQ-RAG master agent: {e}")
        rq_rag_response = "Sorry, an error occurred while processing your request."

    # 3. Save all responses to Appwrite Database
    try:
        document_id = ID.unique()
        databases.create_document(
            database_id=APPWRITE_DATABASE_ID,
            collection_id=APPWRITE_COLLECTION_ID,
            document_id=document_id,
            data={'prompt': user_prompt, 'naive_llm_response': naive_llm_response, 'rq_rag_agent_response': rq_rag_response}
        )
        context.log(f"Response saved to Appwrite Database with ID: {document_id}")
    except Exception as e:
        context.error(f"Error saving to Appwrite Database: {e}")

    # # 4. Return the primary (RQ-RAG) response to the user
    # return context.res.json({'response': rq_rag_response})

    # 4. Combine results and return.
    final_output = {
        'naive_llm_response': naive_llm_response,
        'rq_rag_response': rq_rag_response
    }

    return context.res.json(final_output)
