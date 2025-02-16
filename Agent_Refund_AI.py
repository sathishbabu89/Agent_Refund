# Agent_Refund_Policy.py

# Warning control
import warnings
warnings.filterwarnings('ignore')

# Import libraries, APIs, and LLM
from crewai import Agent, Task, Crew, Process
import os
from crewai_tools import ScrapeWebsiteTool, SerperDevTool, YoutubeVideoSearchTool, PDFSearchTool
from langchain_huggingface import HuggingFaceEndpoint  # Change to HuggingFace endpoint import
import streamlit as st

# Set API tokens
HUGGINGFACE_API_TOKEN = 'your_hf_token'
SERPER_API_KEY = 'your_serper_token'

# Set environment variables for tokens
os.environ["HUGGINGFACE_API_TOKEN"] = HUGGINGFACE_API_TOKEN  # Set environment variable for Hugging Face API token
os.environ["SERPER_API_KEY"] = SERPER_API_KEY  # Set environment variable for Serper API key

# Initialize tools
search_tool = SerperDevTool(api_key=SERPER_API_KEY)  # Use the Serper API key here
scrape_tool = ScrapeWebsiteTool()  # Scraping tool for web data
youtube_tool = YoutubeVideoSearchTool()  # Tool to search for related YouTube videos
pdf_tool = PDFSearchTool()  # Tool to search and extract data from PDFs

# LLM initialization for repo summary using Hugging Face model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",  # The model you want to use from Hugging Face
    task="text-generation", 
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
    huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,  # Hugging Face API token
)


# Test LLM connection before creating agents
try:
    test_response = llm.invoke("Test connection. Reply with 'OK' if working.", task="text-generation")
    print(f"LLM Test Response: {test_response}")
except Exception as e:
    raise Exception(f"Failed to connect to HuggingFace LLM: {str(e)}")

# Creating Agents
data_analyst_agent = Agent(
    role="United Airlines Reservation Agent",
    goal="Refund a customer ({question}) according to the ({url})",
    backstory="Specializing in refunding, this agent uses the ({url}) United which can be searched online to provide crucial insights on refunding for customers",
    verbose=True,
    allow_delegation=False,
    tools=[scrape_tool, search_tool, youtube_tool, pdf_tool],  # Added tools here
    llm=llm  # Explicitly pass the Hugging Face LLM to the agent
)

# Creating Tasks
data_analysis_task = Task(
    description=(
        "Read and understand the United Refund Page. When user asks ({question}), provide them an answer promptly according to the provided url ({url})"
    ),
    expected_output=(
        "answer from user({question}) "
    ),
    agent=data_analyst_agent,
)

# Creating the Crew
refund_crew = Crew(
    agents=[data_analyst_agent],
    tasks=[data_analysis_task],
    manager_llm=llm,  # Use HuggingFace LLM instead of OpenAI's
    process=Process.hierarchical,
    verbose=True
)

# Streamlit UI
def run_refund_agent():
    st.title('United Airlines Refund Agent')
    st.write("Welcome! Please provide your details below to check if you're eligible for a refund.")
    
    # Input form for user to submit their question
    question = st.text_input('Ask your refund question:', 'I cancelled my flight 2 hours prior to departure time, can I still get a refund?')
    url = st.text_input('Provide the URL for the refund policy page:', 'https://www.united.com/en/us/')

    if st.button('Get Refund Info'):
        # Execute the crew
        refund_crew_inputs = {
            'question': question,
            'url': url
        }
        
        result = refund_crew.kickoff(inputs=refund_crew_inputs)

        # Display the result
        st.subheader('Refund Information:')
        st.write(result)

# Run Streamlit app
if __name__ == "__main__":
    run_refund_agent()
