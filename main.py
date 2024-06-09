from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import config  # config.py dosyasını import ediyoruz

# Initialize the LangChain components
llm = ChatOpenAI(api_key=config.OPENAI_API_KEY,
                 temperature=0,
                 max_tokens=512,
                 timeout=None,
                 max_retries=2,
                 model="gpt-4")

# Function to analyze and merge CSV files using Pandas
def analyze_and_merge_csv(files):
    dataframes = []
    for file in files:
        try:
            file_content = file.getvalue().decode("utf-8")
            if not file_content.strip():
                st.error(f"The uploaded CSV file {file.name} is empty.")
                continue
            df = pd.read_csv(file)
            dataframes.append(df)
        except Exception as e:
            st.error(f"Error analyzing CSV file {file.name}: {e}")
            continue
    if dataframes:
        merged_df = pd.concat(dataframes, ignore_index=True)
        merged_df.columns = merged_df.columns.str.lower()  # Convert columns to lowercase
        return merged_df
    else:
        st.error("No valid CSV files uploaded.")
        return None

# Function to create a Pandas DataFrame agent
def create_agent(df):
    try:
        agent = create_pandas_dataframe_agent(llm, df, agent_type="openai-tools", verbose=True)
        return agent
    except Exception as e:
        st.error(f"Error creating DataFrame agent: {e}")
        return None

# Function to generate a histogram
def generate_histogram(df, column):
    try:
        fig, ax = plt.subplots()
        ax.hist(df[column], bins=50)
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Histogram of {column}')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating histogram: {e}")

# Function to generate a bar chart
def generate_bar_chart(df, column):
    try:
        fig, ax = plt.subplots()
        df[column].value_counts().plot(kind='bar', ax=ax)
        ax.set_xlabel(column)
        ax.set_ylabel('Count')
        ax.set_title(f'Bar Chart of {column}')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating bar chart: {e}")

# Function to generate a line chart
def generate_line_chart(df, column):
    try:
        fig, ax = plt.subplots()
        df[column].plot(kind='line', ax=ax)
        ax.set_xlabel('Index')
        ax.set_ylabel(column)
        ax.set_title(f'Line Chart of {column}')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating line chart: {e}")

# Function to handle prompts and LLM responses
def handle_prompt(prompt_template, params):
    chain = prompt_template | llm
    response = chain.invoke(params)
    return response

# Streamlit UI
st.title("Telematics AIoT Assistant")
st.sidebar.header("Upload your files")
uploaded_files = st.sidebar.file_uploader("Choose CSV files", accept_multiple_files=True, type=["csv"])

agent = None
dataframe = None

if uploaded_files:
    dataframe = analyze_and_merge_csv(uploaded_files)
    if dataframe is not None:
        agent = create_agent(dataframe)
        st.sidebar.success("CSV files uploaded and merged successfully")

query = st.text_input("Ask a question about your files")
follow_up_query = st.text_input("Ask a follow-up question")

prompt_templates = {

    "overview": ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that analyzes CSV files and provides detailed information."),
            ("human", "The user has uploaded a CSV file with the following data:\n{csv_data}\nPlease provide an overview of the CSV file, including the number of rows, columns, and the column names."),
        ]
    ),
    "stat_summary": ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that analyzes CSV files and provides detailed information."),
            ("human", "The user has uploaded a CSV file with the following data:\n{csv_data}\nPlease provide a statistical summary of the '{column_name}' column, including mean, median, standard deviation, min, and max values."),
        ]
    ),
    "histogram": ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that analyzes CSV files and provides detailed information."),
            ("human", "The user has uploaded a CSV file with the following data:\n{csv_data}\nPlease create a histogram for the '{column_name}' column."),
        ]
    ),
    "unique_count": ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that analyzes CSV files and provides detailed information."),
            ("human", "The user has uploaded a CSV file with the following data:\n{csv_data}\nPlease count the unique values in the '{column_name}' column."),
        ]
    ),
    "correlation": ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that analyzes CSV files and provides detailed information."),
            ("human", "The user has uploaded a CSV file with the following data:\n{csv_data}\nPlease calculate the correlation between the '{column1}' and '{column2}' columns."),
        ]
    ),
    "general": ChatPromptTemplate.from_messages(
        [
            ("system", """
You are a highly skilled data analysis assistant capable of understanding and analyzing CSV files. 
You will receive a CSV file and perform various data analysis tasks based on the user's requests. 
These tasks may include providing overviews, statistical summaries, visualizations, and generating insights.
When the user asks a question, analyze the question to determine the appropriate task and perform it on the given data.
Here are the tasks you can perform:

1. Overview: Provide a summary of the CSV file including the number of rows, columns, and column names.
2. Statistical Summary: Provide a statistical summary of a specified column, including mean, median, standard deviation, min, and max values.
3. Histogram: Create a histogram for a specified column.
4. Bar Chart: Create a bar chart for a specified column.
5. Line Chart: Create a line chart for a specified column.
6. Unique Count: Count the unique values in a specified column.
7. Correlation: Calculate the correlation between two specified columns.
8. Identify Outliers: Identify any outliers in a specified column.
9. Fastest Cars: Identify the top 5 fastest cars based on their speed.
10. Generate Insights: Generate meaningful insights from the dataset.
"""),
            ("human", "The user has uploaded a CSV file with the following data:\n{csv_data}\nPlease analyze the user's question and perform the appropriate task based on the given data:\nUser's question: {query}"),
        ]
    ),
}

if query or follow_up_query:
    try:
        if query and agent:
            if "overview" in query.lower():
                response = handle_prompt(prompt_templates["overview"], {"csv_data": dataframe.to_csv(index=False)})
                st.write("Response:")
                st.write(response)

            elif "statistical summary" in query.lower():
                # Extract the column name for the statistical summary
                column = query.lower().split("statistical summary of")[-1].strip().split()[0]
                if column in dataframe.columns:
                    response = handle_prompt(prompt_templates["stat_summary"], {"csv_data": dataframe.to_csv(index=False), "column_name": column})
                    st.write("Response:")
                    st.write(response)
                else:
                    st.error(f"Column '{column}' not found in the DataFrame.")

            elif "histogram" in query.lower():
                # Extract the column name for the histogram
                column = query.lower().split("histogram of")[-1].strip().split()[0]
                if column in dataframe.columns:
                    generate_histogram(dataframe, column)
                else:
                    st.error(f"Column '{column}' not found in the DataFrame.")

            elif "bar chart" in query.lower():
                # Extract the column name for the bar chart
                column = query.lower().split("bar chart of")[-1].strip().split()[0]
                if column in dataframe.columns:
                    generate_bar_chart(dataframe, column)
                else:
                    st.error(f"Column '{column}' not found in the DataFrame.")

            elif "line chart" in query.lower():
                # Extract the column name for the line chart
                column = query.lower().split("line chart of")[-1].strip().split()[0]
                if column in dataframe.columns:
                    generate_line_chart(dataframe, column)
                else:
                    st.error(f"Column '{column}' not found in the DataFrame.")

            elif "unique count" in query.lower():
                # Extract the column name for the unique count
                column = query.lower().split("unique count of")[-1].strip().split()[0]
                if column in dataframe.columns:
                    response = handle_prompt(prompt_templates["unique_count"], {"csv_data": dataframe.to_csv(index=False), "column_name": column})
                    st.write("Response:")
                    st.write(response)
                else:
                    st.error(f"Column '{column}' not found in the DataFrame.")

            elif "correlation" in query.lower():
                # Extract the column names for the correlation
                parts = query.lower().split("correlation between")[-1].strip().split("and")
                column1 = parts[0].strip()
                column2 = parts[1].strip() if len(parts) > 1 else None
                if column1 in dataframe.columns and column2 in dataframe.columns:
                    response = handle_prompt(prompt_templates["correlation"], {"csv_data": dataframe.to_csv(index=False), "column1": column1, "column2": column2})
                    st.write("Response:")
                    st.write(response)
                else:
                    st.error(f"One or both columns '{column1}', '{column2}' not found in the DataFrame.")
            else:
                response = agent.invoke({
                    "input": query
                })
                st.write("Response:")
                st.write(response['output'])

                if "show the table" in query.lower():
                    try:
                        st.write("Table format:")
                        st.dataframe(dataframe)
                    except Exception as e:
                        st.error(f"Error displaying table: {e}")

        if follow_up_query and agent:
            follow_up_response = agent.invoke({
                "input": follow_up_query
            })
            st.write(f"Follow-up Question: {follow_up_query}")
            st.write(f"Follow-up Response: {follow_up_response['output']}")

    except Exception as e:
        st.error(f"Error querying agent: {e}")
