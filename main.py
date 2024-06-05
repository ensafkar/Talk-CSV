import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import config  # config.py dosyasını import ediyoruz

# Initialize the LangChain components
llm = ChatOpenAI(api_key=config.OPENAI_API_KEY,
                 temperature=0,
                 max_tokens=512,
                 timeout=None,
                 max_retries=2,
                 model="gpt-4")

# Function to analyze CSV using Pandas
def analyze_csv(file):
    try:
        file_content = file.getvalue().decode("utf-8")
        if not file_content.strip():
            st.error("The uploaded CSV file is empty.")
            return None
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error analyzing CSV file: {e}")
        return None

# Function to calculate the average GPSGROUNDSPEED for each VENDORCODE
def calculate_average_speed(df):
    try:
        avg_speed = df.groupby('VENDORCODE')['GPSGROUNDSPEED'].mean().reset_index()
        return avg_speed
    except Exception as e:
        st.error(f"Error calculating average speed: {e}")
        return None

# Define the chaining prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "<s> [INST] You are an assistant specialized in question-answering tasks, especially those involving CSV files. Utilize the following pieces of retrieved context to answer the question. If you don't know the answer, simply state that you don't know. [/INST] </s>"),
        ("human", "[INST] Question: {question} \n\nContext: {context} \n\nAnswer: [/INST]")
    ]
)

# Function to send a question to LLM and get the response
def query_llm(question, context):
    try:
        chain = prompt_template | llm
        response = chain.invoke({
            "context": context,
            "question": question
        })
        return response.content
    except Exception as e:
        st.error(f"Error querying LLM: {e}")
        return str(e)

# Streamlit UI
st.title("Chat with your CSV files")
st.sidebar.header("Upload your files")
uploaded_files = st.sidebar.file_uploader("Choose a CSV file", accept_multiple_files=True, type=["csv"])

dataframe_agents = {}

if uploaded_files:
    for file in uploaded_files:
        if file.type == "text/csv":
            df = analyze_csv(file)
            if df is not None:
                dataframe_agents[file.name] = df
                st.sidebar.success(f"CSV file {file.name} uploaded successfully")

query = st.text_input("Ask a question about your files")
if query:
    for file_name, df in dataframe_agents.items():
        if "average GPSGROUNDSPEED for each VENDORCODE" in query:
            avg_speed_df = calculate_average_speed(df)
            if avg_speed_df is not None:
                st.write(f"Response from {file_name}:")
                st.write(avg_speed_df)
                st.write("Table format:")
                st.dataframe(avg_speed_df)
        else:
            csv_content = df.to_csv(index=False)
            response = query_llm(query, csv_content)
            st.write(f"Response from {file_name}:")
            st.write(response)

        if "show the table" in query.lower():
            try:
                st.write("Table format:")
                st.dataframe(df)
            except Exception as e:
                st.error(f"Error displaying table: {e}")
