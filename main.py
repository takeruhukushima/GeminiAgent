# @title S2 Paper Search Agent (Terminal Interface)
# Import necessary libraries
import os
import asyncio
import requests
import json
import datetime
import time
import pathlib
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm # For multi-model support
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types # For creating message Content/Parts

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.ERROR)

print("Libraries imported.")

# Load environment variables from .env file
load_dotenv()

# Access API keys from environment variables
s2_api_key = os.environ.get("S2_API_KEY")
google_api_key = os.environ.get("GOOGLE_API_KEY")

# --- Verify Keys (Optional Check) ---
print("API Keys Set:")
print(f"Semantic Scholar API Key set: {'Yes' if s2_api_key else 'No (MISSING S2_API_KEY in .env)'}")
print(f"Google API Key set: {'Yes' if google_api_key else 'No (MISSING GOOGLE_API_KEY in .env)'}")

MODEL_GEMINI_2_0_FLASH = "gemini-2.0-flash"

print("\nEnvironment configured.")

# @title Define the search_papers Tool
def search_papers(query: str) -> dict:
    """Searches for academic papers on Semantic Scholar based on the provided query.

    Args:
        query (str): The search query (e.g., "machine learning", "climate change").

    Returns:
        dict: A dictionary containing the search results.
              Includes 'total' (total number of results),
              'papers' (list of top 20 papers with title and abstract),
              and 'status' ('success' or 'error').
    """
    print(f"--- Tool: search_papers called for query: {query} ---") # Log tool execution
    
    # S2 API endpoint for paper search
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    # Parameters for the API request
    params = {
        "query": query,
        "limit": 20,  # Get top 20 papers
        "fields": "title,abstract,year,citationCount,authors,url"
    }
    
    # Headers including the API key
    headers = {
        "x-api-key": s2_api_key
    }
    
    try:
        # Make the API request
        response = requests.get(url, params=params, headers=headers)
        
        # Check if request was successful
        if response.status_code == 200:
            data = response.json()
            
            # Extract relevant information
            total = data.get('total', 0)
            papers = data.get('data', [])
            
            # Format each paper to include only needed information
            formatted_papers = []
            for paper in papers:
                authors = paper.get('authors', [])
                author_names = [author.get('name', '') for author in authors]
                
                formatted_paper = {
                    'title': paper.get('title', 'No title available'),
                    'abstract': paper.get('abstract', 'No abstract available'),
                    'year': paper.get('year'),
                    'authors': ', '.join(author_names),
                    'citationCount': paper.get('citationCount', 0),
                    'url': paper.get('url', '')
                }
                formatted_papers.append(formatted_paper)
            
            return {
                'status': 'success',
                'total': total,
                'papers': formatted_papers
            }
        else:
            return {
                'status': 'error',
                'error_message': f"API request failed with status code {response.status_code}: {response.text}"
            }
    except Exception as e:
        return {
            'status': 'error',
            'error_message': f"An error occurred: {str(e)}"
        }

# @title Define the Paper Search Agent
AGENT_MODEL = MODEL_GEMINI_2_0_FLASH

paper_search_agent = Agent(
    name="paper_search_agent_v1",
    model=AGENT_MODEL,
    description="Searches and analyzes academic papers from Semantic Scholar in japanese.",
    instruction="""You are a helpful academic research assistant.
                When the user asks for papers on a specific topic, use the 'search_papers' tool to find relevant information.
                After getting the results, provide the following information:
                1. Total number of papers found for the query
                2. Details of the top 20 papers (title and abstract)
                3. A brief analysis of trends or patterns in these top papers
                
                If the tool returns an error, inform the user politely.
                Always format your response in a clear, structured way that helps academics understand the research landscape.
                use japanese for response.
                """,
    tools=[search_papers],
)

print(f"Agent '{paper_search_agent.name}' created using model '{AGENT_MODEL}'.")

# @title Setup Session Service and Runner

# --- Session Management ---
# Key Concept: SessionService stores conversation history & state.
# InMemorySessionService is simple, non-persistent storage for this tutorial.
session_service = InMemorySessionService()

# Define constants for identifying the interaction context
APP_NAME = "paper_search_app"
USER_ID = "user_1"
SESSION_ID = "session_001" # Using a fixed ID for simplicity

# Create the specific session where the conversation will happen
session = session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID
)
print(f"Session created: App='{APP_NAME}', User='{USER_ID}', Session='{SESSION_ID}'")

# --- Runner ---
# Key Concept: Runner orchestrates the agent execution loop.
runner = Runner(
    agent=paper_search_agent, # The agent we want to run
    app_name=APP_NAME,   # Associates runs with our app
    session_service=session_service # Uses our session manager
)
print(f"Runner created for agent '{runner.agent.name}'.")

# @title Define Agent Interaction Function

from google.genai import types # For creating message Content/Parts

# Create a class to handle the chat session and logging
class ChatSession:
    def __init__(self, runner, user_id, session_id):
        self.runner = runner
        self.user_id = user_id
        self.session_id = session_id
        self.conversation_log = []
        
        # Create output directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = pathlib.Path(f"out/{timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / "conversation.md"
        
        # Initialize the output file with a header
        with open(self.output_file, "w") as f:
            f.write(f"# Semantic Scholar Paper Search Conversation\n\n")
            f.write(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"---\n\n")
    
    async def process_query(self, query):
        """Process a user query and get the agent's response"""
        # Add the user query to the log
        self.conversation_log.append({"role": "user", "content": query})
        
        # Format for the markdown file
        with open(self.output_file, "a") as f:
            f.write(f"## User Query\n\n")
            f.write(f"{query}\n\n")
        
        print(f"\n\033[1;32mProcessing your query...\033[0m")
        
        # Prepare the user's message in ADK format
        content = types.Content(role='user', parts=[types.Part(text=query)])
        
        final_response_text = "Agent did not produce a final response." # Default
        
        # Run the agent
        async for event in self.runner.run_async(user_id=self.user_id, session_id=self.session_id, new_message=content):
            if event.is_final_response():
                if event.content and event.content.parts:
                    # Assuming text response in the first part
                    final_response_text = event.content.parts[0].text
                elif event.actions and event.actions.escalate:
                    final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
                break
        
        # Add the response to the log
        self.conversation_log.append({"role": "assistant", "content": final_response_text})
        
        # Format for the markdown file
        with open(self.output_file, "a") as f:
            f.write(f"## Agent Response\n\n")
            f.write(f"{final_response_text}\n\n")
            f.write(f"---\n\n")
        
        # Print the response
        print(f"\n\033[1;34mAgent Response:\033[0m\n{final_response_text}\n")
        return final_response_text

# @title Run the Interactive Terminal Chat

async def run_interactive_chat():
    """Run an interactive chat session with the agent"""
    # Create a chat session
    chat = ChatSession(runner=runner, user_id=USER_ID, session_id=SESSION_ID)
    
    print("\n\033[1;36m=== S2 Paper Search Agent Terminal Chat ===\033[0m")
    print("Type your research queries and get paper summaries and analysis.")
    print("Type 'exit', 'quit', or 'q' to end the conversation.")
    print(f"Conversation log will be saved to: \033[1;33m{chat.output_file}\033[0m")
    print("\033[1;36m=============================================\033[0m\n")
    
    # Welcome message
    print("\033[1;34mS2 Paper Search Agent:\033[0m I'm ready to help you search for academic papers! What topic would you like to explore?\n")
    
    # Main conversation loop
    while True:
        # Get user input
        user_input = input("\033[1;32mYou:\033[0m ")
        
        # Check for exit commands
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("\n\033[1;34mS2 Paper Search Agent:\033[0m Thank you for using the Paper Search Agent. Goodbye!")
            print(f"\nConversation saved to: \033[1;33m{chat.output_file}\033[0m\n")
            break
        
        # Process the query
        await chat.process_query(user_input)

# Use this section for standard Python scripts
if __name__ == "__main__":
    try:
        asyncio.run(run_interactive_chat())
    except KeyboardInterrupt:
        print("\n\nChat session terminated by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")