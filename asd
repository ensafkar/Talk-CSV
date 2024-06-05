import os
import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent
import tempfile

# Ensure tabulate is imported
try:
    import tabulate
except ImportError:
    st.error("The 'tabulate' package is not installed. Please install it using 'pip install tabulate'.")

# Set your OpenAI API key here
os.environ["OPENAI_API_KEY"] = "sk-proj-hA1wcGCpBbw7wy4NWWPUT3BlbkFJGjVqjo8sez695xRSwZoI"

# Initialize the ChatOpenAI model
llm = ChatOpenAI( temperature=0)

# Streamlit UI
st.title("AI-Powered CSV Data Analysis")
st.write("This application uses OpenAI's language model to analyze CSV data.")

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            temp_file_path = tmp_file.name
            df.to_csv(temp_file_path, index=False)

        # Create the CSV agent
        agent_executor = create_csv_agent(llm, temp_file_path, verbose=True)

        # User input for querying
        query = st.text_input("Enter your query:", "Group by vendor code and count messages for each code")

        # Use the agent to process the query
        if st.button("Run Query"):
            with st.spinner("Running query..."):
                try:
                    # Use the agent for query execution
                    result = agent_executor.invoke({"input": query})
                    st.subheader("Query Result")
                    st.write(result['output'])
                except Exception as e:
                    st.error(f"An error occurred during the query execution: {e}")

    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty. Please upload a valid CSV file.")
    except ImportError as e:
        st.error(f"Missing dependency: {e}. Please ensure all dependencies are installed.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a CSV file to proceed.")











































# Display interaction history
st.sidebar.subheader("Interaction History")
for entry in langchain.memory:
    st.sidebar.write(f"Q: {entry['question']}")
    st.sidebar.write(f"A: {entry['answer']}")

# Display Data Sources
if st.sidebar.checkbox("Show Data Sources"):
    st.subheader("Public Data Sources")
    for source in langchain.public_data_sources:
        st.write(source.fetch_data())
        st.subheader("Private Data Sources")
        for source in langchain.private_data_sources:
            st.write(source.fetch_data())
        st.subheader("RAS Data Sources")
        for source in langchain.ras_data_sources:
            st.write(source.fetch_data())

    # Process data
    if st.button("Process Data"):
        langchain.process()









































import streamlit as st
import openai
import pandas as pd
import tempfile
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_community.document_loaders import PyPDFLoader

# Initialize OpenAI
openai.api_key = 'your-openai-api-key'

class LangchainSystem:
    def __init__(self):
        self.public_data_sources = []
        self.private_data_sources = []
        self.ras_data_sources = []
        self.tools = []
        self.flowise_langflow = None
        self.agent = None
        self.ui_html = None
        self.chatbot = None

    def add_public_data_source(self, source):
        self.public_data_sources.append(source)

    def add_private_data_source(self, source):
        self.private_data_sources.append(source)

    def add_ras_data_source(self, source):
        self.ras_data_sources.append(source)

    def add_tool(self, tool):
        self.tools.append(tool)

    def set_flowise_langflow(self, flow):
        self.flowise_langflow = flow

    def set_agent(self, agent):
        self.agent = agent

    def set_ui_html(self, ui):
        self.ui_html = ui

    def set_chatbot(self, chatbot):
        self.chatbot = chatbot

    def process(self):
        st.write("Processing data from all sources with the tools and agents.")
        # Implement the processing logic here

class DataSource:
    def __init__(self, name, data=None):
        self.name = name
        self.data = data

    def fetch_data(self):
        st.write(f"Fetching data from {self.name}")
        # Implement data fetching logic here
        return self.data

class Tool:
    def __init__(self, name):
        self.name = name

    def use_tool(self):
        st.write(f"Using tool: {self.name}")
        # Implement tool usage logic here

class FlowiseLangflow:
    def __init__(self, config):
        self.config = config

    def execute_flow(self):
        st.write("Executing flow with Flowise/Langflow.")
        # Implement flow execution logic here

class Agent:
    def __init__(self, name):
        self.name = name

    def perform_task(self):
        st.write(f"Agent {self.name} performing task.")
        # Implement agent tasks here

class UIHTML:
    def __init__(self):
        self.elements = []

    def render(self):
        st.write("Rendering UI/HTML.")
        # Implement UI rendering logic here

class Chatbot:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = api_key

    def interact(self, prompt):
        response = openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()

# Initialize Langchain system
langchain = LangchainSystem()

# Streamlit UI
st.title("Langchain System")
st.sidebar.title("Configuration")

# File upload for CSV
uploaded_csv = st.sidebar.file_uploader("Upload CSV", type=["csv"])
csv_agent = None
if uploaded_csv:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_csv:
        temp_csv.write(uploaded_csv.getvalue())
        temp_csv_path = temp_csv.name
    csv_data = pd.read_csv(temp_csv_path)
    langchain.add_public_data_source(DataSource("Uploaded CSV", data=csv_data))
    csv_agent = create_csv_agent(ChatOpenAI(api_key=openai.api_key), temp_csv_path)

# File upload for PDF
uploaded_pdf = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
pdf_text = ""
if uploaded_pdf:
    loader = PyPDFLoader(uploaded_pdf)
    pages = loader.load_and_split()
    for page in pages:
        pdf_text += page.page_content
    langchain.add_ras_data_source(DataSource("Uploaded PDF", data=pdf_text))

# Adding public data sources
langchain.add_public_data_source(DataSource("OpenAI"))
langchain.add_public_data_source(DataSource("CSV"))

# Adding private data sources
langchain.add_private_data_source(DataSource("Lane 3"))
langchain.add_private_data_source(DataSource("Thr-3"))

# Adding RAS data sources
langchain.add_ras_data_source(DataSource("PDF"))

# Adding tools
langchain.add_tool(Tool("Web Scrape"))
langchain.add_tool(Tool("Pandas AI"))
langchain.add_tool(Tool("Unstructured.io"))

# Setting up Flowise/Langflow
langchain.set_flowise_langflow(FlowiseLangflow(config="config"))

# Setting up an agent
langchain.set_agent(Agent("Agent 1"))

# Setting up UI/HTML
langchain.set_ui_html(UIHTML())

# Setting up Chatbot
chatbot = Chatbot(api_key=openai.api_key)
langchain.set_chatbot(chatbot)

# Input for Chatbot
prompt = st.text_input("Ask the chatbot:")
if prompt:
    if uploaded_csv:
        response = csv_agent.run(prompt)
        st.write(f"Chatbot response for CSV: {response}")
    elif uploaded_pdf:
        response = chatbot.interact(prompt)
        st.write(f"Chatbot response for PDF: {response}")
    else:
        response = chatbot.interact(prompt)
        st.write(f"Chatbot response: {response}")

# Display Data Sources
if st.sidebar.checkbox("Show Data Sources"):
    st.subheader("Public Data Sources")
    for source in langchain.public_data_sources:
        st.write(source.fetch_data())
    st.subheader("Private Data Sources")
    for source in langchain.private_data_sources:
        st.write(source.fetch_data())
    st.subheader("RAS Data Sources")
    for source in langchain.ras_data_sources:
        st.write(source.fetch_data())

# Process data
if st.button("Process Data"):
    langchain.process()






######################################################################################################
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import AgentType
import pandas as pd
import config  # config.py dosyasını import ediyoruz

# Initialize the LangChain components
llm = ChatOpenAI(api_key=config.OPENAI_API_KEY,
                 temperature=0,
                 max_tokens=None,
                 timeout=None,
                 max_retries=2,
                 model="gpt-4",
)


# Function to create a Pandas DataFrame agent
def create_dataframe_agent(df):
    try:
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Doğru AgentType enum değeri
        )
        return agent
    except Exception as e:
        st.error(f"Error creating DataFrame agent: {e}")
        return None


# Function to analyze CSV using Pandas
def analyze_csv(file):
    try:
        # Read file content to debug
        file_content = file.getvalue().decode("utf-8")

        # Check if the file content is empty
        if not file_content.strip():
            st.error("The uploaded CSV file is empty.")
            return None, None

        # Load CSV into DataFrame
        df = pd.read_csv(file)
        unique_values = df['VENDORCODE'].value_counts().to_dict()
        return df, unique_values
    except Exception as e:
        st.error(f"Error analyzing CSV file: {e}")
        return None, None


# Function to generate user-friendly text
def generate_user_friendly_text(response):
    try:
        prompt = [
            {"role": "system", "content": "You are a helpful assistant that formats responses in a user-friendly way."},
            {"role": "user", "content": response}
        ]
        chat_response = llm(prompt)
        return chat_response.choices[0].message['content']
    except Exception as e:
        st.error(f"Error generating user-friendly text: {e}")
        return response



# Streamlit UI
st.title("Chat with your CSV files")
st.sidebar.header("Upload your files")
uploaded_files = st.sidebar.file_uploader("Choose a CSV file", accept_multiple_files=True, type=["csv"])

agents = {}
csv_analysis = {}
dataframe_agents = {}

if uploaded_files:
    for file in uploaded_files:
        if file.type == "text/csv":
            df, unique_values = analyze_csv(file)
            if df is not None:
                agent = create_dataframe_agent(df)
                if agent:
                    agents[file.name] = agent
                    st.sidebar.success(f"CSV file {file.name} indexed")
                dataframe_agents[file.name] = df
                csv_analysis[file.name] = unique_values

query = st.text_input("Ask a question about your files")
if query:
    if agents:
        for file_name, agent in agents.items():
            try:
                response = agent.invoke(query)
                st.write(f"Response from {file_name}:")
                st.write(response['output'])  # Yanıtları daha düzgün göstermek
            except Exception as e:
                st.error(f"Error querying agent for {file_name}: {e}")
    else:
        st.write("Please upload a CSV file to enable querying.")

if csv_analysis:
    st.sidebar.header("CSV Analysis")
    for file_name, analysis in csv_analysis.items():
        st.sidebar.subheader(f"Analysis of {file_name}")
        st.sidebar.write(analysis)


# Function to use Pandas AI for DataFrame analysis
def use_pandas_ai(df, query):
    try:
        agent = create_dataframe_agent(df)
        if agent:
            response = agent.invoke(query)
            return response
        else:
            st.error("Failed to create Pandas DataFrame agent.")
            return None
    except Exception as e:
        st.error(f"Error using Pandas AI: {e}")
        return None


if st.sidebar.button("Analyze with Pandas AI"):
    if dataframe_agents:
        for file_name, df in dataframe_agents.items():
            response = use_pandas_ai(df, query)
            if response:
                st.write(f"Pandas AI Response from {file_name}: {response['output']}")
    else:
        st.write("Please upload a CSV file and enter a query to use Pandas AI.")
#######################################################################################################################################################










