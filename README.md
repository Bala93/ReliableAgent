# RQ-RAG Fact-Checker Agent

This project is an Appwrite Function that implements an advanced Retrieval-Augmented Generation (RAG) system. It uses the **RQ-RAG (Refined Query RAG)** framework to improve the accuracy and reliability of AI-generated answers by refining user queries, verifying facts, and citing sources.

The agent is designed to be deployed as a serverless function on the Appwrite platform.

## Features

- **Intelligent Query Refinement**: Automatically analyzes user prompts and refines them using one of three strategies:
    - **Decomposition**: Breaks down complex queries into simple, verifiable questions.
    - **Disambiguation**: Clarifies ambiguous queries by exploring multiple interpretations.
    - **Simple**: Handles straightforward, direct queries.
- **Fact Verification**: Uses a web search API (Tavily) to find and verify information for each sub-question.
- **Memory-Based Caching**: Leverages the Mem0 platform to store and retrieve previously verified facts, reducing API calls and improving response time for recurring queries.
- **Source Citation**: Synthesizes a final answer and provides source URLs for the information used.
- **Appwrite Integration**:
    - Designed to run as an Appwrite Cloud Function.
    - Stores both the naive LLM response and the advanced RQ-RAG response in an Appwrite Database for logging and analysis.

## Prerequisites

To deploy and run this function, you need to set up the following environment variables in your Appwrite Function settings:

- `OPENAI_API_KEY`: Your API key for the OpenAI platform.
- `TAVILY_API_KEY`: Your API key for the Tavily web search API.
- `APPWRITE_PROJECT_ID`: Your Appwrite project ID.
- `APPWRITE_API_KEY`: Your Appwrite API key with `databases.write` permissions.
- `APPWRITE_DATABASE_ID`: The ID of the database you want to use for storage.
- `APPWRITE_COLLECTION_ID_RQRAG`: The ID of the collection within your database where responses will be saved.

You also need to set up a collection in your Appwrite Database with the following attributes to store the results:

| Attribute Name         | Type     | Required |
|------------------------|----------|----------|
| `prompt`               | String   | Yes      |
| `naive_llm_response`   | String   | Yes      |
| `rq_rag_agent_response`| String   | Yes      |

## Installation and Deployment

1.  **Clone the repository** or download the function code.
2.  **Create a new Appwrite Function** in your Appwrite project.
3.  **Select Python 3.11** (or a compatible version) as the runtime.
4.  **Upload the `main.py` file** to the function.
5.  **Add the required dependencies** by creating a `requirements.txt` file in the same directory as your code with the following content:

    ```text
    openai
    tavily-python
    mem0
    appwrite
    ```

6.  **Deploy the function** by clicking the "Deploy" button in the Appwrite console.
7.  **Set the environment variables** listed in the Prerequisites section within your Appwrite Function's settings.
8.  **Configure the function's execution permissions** to allow it to be triggered by your application.

## Usage

This function is designed to be triggered via an HTTP request (e.g., from a web application or another service). It expects a JSON payload in the request body with a `prompt` key.

**Example Request Body:**

```json
{
    "prompt": "What kind of university is the school where Rey Ramsey was educated an instance of?"
}
