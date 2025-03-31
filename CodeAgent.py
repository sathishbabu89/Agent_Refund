import logging
import streamlit as st
import torch
import os
from crewai import Agent, Task, Crew, Process
from langchain_huggingface import HuggingFaceEndpoint
from crewai_tools import ScrapeWebsiteTool

# Force PyTorch to use CPU
device = torch.device("cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set API Tokens
HUGGINGFACE_API_TOKEN = "API_TOKEN"

os.environ["HUGGINGFACE_API_TOKEN"] = HUGGINGFACE_API_TOKEN

# Initialize LLM
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation", 
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
    huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
)

# Initialize Tools
scrape_tool = ScrapeWebsiteTool()

# Define Agents
code_doc_agent = Agent(
    role="Code Documentation Specialist",
    goal="Analyze the provided code and generate structured documentation with explanations.",
    backstory="Expert in understanding and documenting various programming languages, focusing on clarity and completeness.",
    verbose=True,
    allow_delegation=False,
    tools=[scrape_tool],  # Removed search_tool
    llm=llm
)

code_chat_agent = Agent(
    role="Code Analysis Chat Assistant",
    goal="Answer user queries about the uploaded code with detailed insights.",
    backstory="An AI assistant skilled in providing technical insights and explanations for various programming languages.",
    verbose=True,
    allow_delegation=False,
    tools=[],  # Removed search_tool
    llm=llm
)

# Define Tasks
documentation_task = Task(
    description="Generate documentation for the uploaded code, including functions, classes, and business logic.",
    expected_output="Structured documentation in markdown format.",
    agent=code_doc_agent,
)

chat_task = Task(
    description="Provide detailed responses to user questions about the uploaded code.",
    expected_output="A clear and concise technical explanation.",
    agent=code_chat_agent,
)

# Define Crew
code_analysis_crew = Crew(
    agents=[code_doc_agent, code_chat_agent],
    tasks=[documentation_task, chat_task],
    manager_llm=llm,
    process=Process.hierarchical,
    verbose=True
)

# Streamlit UI
st.set_page_config(page_title="🚀 AI-Powered Code Analyzer", page_icon="🤖")
st.title("🚀 AI-Powered Code Analyzer")

st.sidebar.header("Upload Your Code File")
uploaded_file = st.sidebar.file_uploader("Upload a code file (Java, Python, JS, etc.)", type=["java", "py", "js", "cpp", "c"])

if uploaded_file:
    file_contents = uploaded_file.read().decode("utf-8")
    st.sidebar.success("File uploaded successfully!")
    
    # Generate Documentation
    if st.button("Generate Documentation"):
        with st.spinner("Generating documentation... ✍️"):
            doc_result = code_analysis_crew.kickoff(inputs={"code": file_contents}, task=documentation_task)
            st.subheader("📄 Generated Documentation")
            st.write(doc_result)
    
    # AI Chat Feature
    user_question = st.text_input("Ask a question about the uploaded code:")
    if user_question:
        with st.spinner("Thinking... 🤖"):
            chat_result = code_analysis_crew.kickoff(inputs={"code": file_contents, "question": user_question}, task=chat_task)
            st.subheader("🤖 AI Response")
            st.write(chat_result)

# Sidebar Info
st.sidebar.markdown("### About")
st.sidebar.write("This AI tool provides automated documentation and chat-based insights for various programming languages.")
