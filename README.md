# Agent Refund Policy

This repository contains a Python application that uses AI agents to provide refunds information based on a provided URL of a company's refund policy. The system specifically focuses on United Airlines' refund policy but can be adapted to other services.

## Features

- **Automated refund agent:** Uses a specialized agent to handle refund-related queries based on a company's refund policy.
- **Web scraping and data extraction tools:** Allows scraping of web pages, searching YouTube, and extracting information from PDFs for refund details.
- **Integration with Hugging Face LLM:** Leverages a powerful text generation model to summarize and provide accurate answers.
- **Streamlit web interface:** A simple user interface to interact with the refund agent.

## Requirements

- Python 3.7+
- Hugging Face API token
- Serper API key
- Streamlit
- Other dependencies listed below

## Setup

### 1. Clone this repository

```bash
git clone https://github.com/yourusername/Agent_Refund_Policy.git
cd Agent_Refund_Policy
```

### 2. Install required libraries

Install dependencies using `pip`:

```bash
pip install -r requirements.txt
```

### 3. Set up API tokens

Create a `.env` file in the root of your project and add your API tokens:

```bash
HUGGINGFACE_API_TOKEN=your_hf_token
SERPER_API_KEY=your_serper_token
```

Make sure to replace `your_hf_token` and `your_serper_token` with your actual API tokens from Hugging Face and Serper, respectively.

### 4. Run the Streamlit app

To start the Streamlit web application, run the following command:

```bash
streamlit run Agent_Refund_Policy.py
```

The web interface will be available at `http://localhost:8501` by default.

## How It Works

### Tools and Libraries

- **CrewAI:** Utilized for orchestrating agents, tasks, and processes in the system.
- **ScrapeWebsiteTool:** A tool to scrape relevant web pages and gather refund-related information.
- **SerperDevTool:** Allows you to perform searches on the web to gather data for refund questions.
- **YoutubeVideoSearchTool:** Used for searching YouTube for relevant videos related to refunds.
- **PDFSearchTool:** Extracts data from PDFs if relevant refund policies are stored in that format.
- **Hugging Face Endpoint:** A custom model endpoint used to generate responses for refund queries.

### Agent Setup

- **Agent:** The refund agent is specialized to understand and respond to refund-related questions. It uses a backstory focused on United Airlines' refund policy and employs multiple tools to gather data for answering user queries.
  
### Streamlit Web Interface

- Users can interact with the system by entering a refund-related question and providing the URL to the company's refund policy page.
- Once the user clicks "Get Refund Info," the agent processes the request, uses the relevant tools to extract information, and provides a response on the Streamlit page.

## Example Usage

1. Open the web interface and enter a refund-related question, such as:
   - "I canceled my flight 2 hours prior to departure time, can I still get a refund?"
2. Provide the URL of the refund policy page, such as:
   - `https://www.united.com/en/us/`
3. Click "Get Refund Info" to get the relevant information extracted by the agent.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

