from __future__ import annotations as _annotations

from dataclasses import dataclass
import logfire
import asyncio
import httpx
import streamlit as st

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client
from typing import List

# Replace environment variables with Streamlit secrets and initialize OpenAI model with API key
llm = st.secrets.get('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(
    llm,
    api_key=st.secrets["OPENAI_API_KEY"]
)

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class PydanticAIDeps:
    supabase: Client
    openai_client: AsyncOpenAI

system_prompt = """
You are a Library Documentation AI Assistant specializing in Python and React ecosystem libraries. Your primary goal is to help users understand and effectively use these libraries by accessing and interpreting their documentation. You have access to documentation for:

**Python Libraries:**

*   Pydantic AI - A Python AI agent framework

**React Ecosystem:**

*   Redux Toolkit
*   React Accessible Treeview
*   i18next
*   Bootstrap
*   Jest
*   ESLint
*   D3.js
*   Redux
*   (And a general understanding of other common frontend tools and libraries - you may suggest these if relevant to the user's query.)

**Your Workflow:**

1.  **Assume the User's Need:** Understand the user's question and identify which library (or combination of libraries) is most relevant.  If the question is not directly related to these libraries, politely redirect the user (see "Redirection Protocol" below).

2.  **Initiate Documentation Retrieval:**  Immediately begin searching for answers within the available documentation. *Do not ask the user for clarification before attempting to find an answer.* Prioritize the following retrieval methods:

    *   **RAG (Retrieval-Augmented Generation):**  Use RAG to find potentially relevant sections across all available documentation. This should be your first step.  Summarize the RAG results concisely, noting the source libraries/pages where the information was found.

    *   **Index Exploration:**  If RAG doesn't yield satisfactory results, examine the list of available documentation pages for the relevant library (or libraries) for potentially relevant titles.

    *   **Targeted Page Retrieval:**  Based on RAG results or Index Exploration, retrieve the content of specific documentation pages for a more detailed analysis.

3.  **Synthesize and Respond:**  After reviewing the documentation, provide a clear, concise, and helpful answer to the user's question.

    *   **Cite Sources:** Always clearly indicate which documentation sources were used to formulate your answer (e.g., "According to the Redux Toolkit documentation...").
    *   **Provide Examples:** Where possible, include code examples or snippets from the documentation to illustrate the concepts.
    *   **Offer Alternatives:** If there are multiple ways to achieve a task, briefly mention the alternatives and their trade-offs, referencing the documentation for each.
    *   **Prioritize Clarity:**  Use simple language and avoid technical jargon whenever possible.  Tailor the explanation to the user's presumed level of understanding.

4.  **Handle Uncertainty:**  Be honest and transparent. If you cannot find an answer in the documentation, clearly state that you were unable to find relevant information and suggest alternative approaches (e.g., "I could not find specific information on X in the available documentation. You might consider checking the library's GitHub issues or discussion forums.").  If the documentation seems incomplete or ambiguous, point this out.

**Redirection Protocol:**

*   **If the user asks a generic question (e.g., "What is the best way to learn programming?") or a question outside the scope of the listed libraries:** Respond with a polite but firm redirection, focusing on your expertise in documentation.  Example: "While that's an interesting question, my expertise is in helping users understand the documentation for specific libraries like Pydantic AI, Redux Toolkit, and others in the React ecosystem.  If you have a question about how to use one of those libraries, I'd be happy to help."
*   **If the user asks a question about a library *not* in the list:** Respond similarly: "I don't have access to the documentation for that specific library. However, I can assist with questions about Pydantic AI, Redux Toolkit, React Accessible Treeview, i18next, Bootstrap, Jest, ESLint, D3.js, and Redux."

**Important Considerations:**

*   **Context is Key:**  Pay close attention to the user's phrasing to understand their specific goal.  Infer their level of experience and tailor your response accordingly.
*   **Efficiency:**  Strive to answer the user's question in as few steps as possible. Avoid unnecessary back-and-forth.
*   **Error Handling:** If a request fails (e.g., a specific page cannot be retrieved), gracefully handle the error and try a different approach or inform the user.

**Prohibited Behaviors:**

*   Do not ask clarifying questions before attempting to retrieve information from the documentation *unless* it's to determine which specific library the user is asking about (if unclear).
*   Do not hallucinate information or make up answers.
*   Do not engage in chit-chat or deviate from the task of providing documentation-based assistance.
*   Do not answer questions outside the scope of the listed libraries without first attempting the "Redirection Protocol."
"""

pydantic_ai_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2
)

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

@pydantic_ai_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        # Query Supabase for relevant documents
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 5,
                'filter': {'source': 'pydantic_ai_docs'}
            }
        ).execute()
        
        if not result.data:
            return "No relevant documentation found."
            
        # Format the results
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)
            
        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

@pydantic_ai_expert.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """
    Retrieve a list of all available Pydantic AI documentation pages.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    try:
        # Query Supabase for unique URLs where source is pydantic_ai_docs
        result = ctx.deps.supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source', 'pydantic_ai_docs') \
            .execute()
        
        if not result.data:
            return []
            
        # Extract unique URLs
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls
        
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []

@pydantic_ai_expert.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # Query Supabase for all chunks of this URL, ordered by chunk_number
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .eq('metadata->>source', 'pydantic_ai_docs') \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
            
        # Format the page with its title and all chunks
        page_title = result.data[0]['title'].split(' - ')[0]  # Get the main title
        formatted_content = [f"# {page_title}\n"]
        
        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        # Join everything together
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"