import streamlit as st
import pandas as pd
import plotly.express as px
import requests  # For Groq API or other API interactions
import os

# Groq API endpoint (replace with your actual Groq API endpoint)
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com "  # Replace this with the actual Groq API URL

# Function to load the dataset (CSV or Excel)
def load_data(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
            return None
        return df
    else:
        return None

# Function to query the dataset using Groq's API
def query_to_analysis(query, df):
    # Prepare the payload with the dataset column names and the query
    payload = {
        "query": query,
        "columns": list(df.columns)
    }
    
    # Headers for the API request
    headers = {
        "Authorization": f"Bearer {GROQ_API_URL}",  # Groq API key passed in headers for authentication
        "Content-Type": "application/json"
    }
    
    # Make the request to Groq API
    try:
        response = requests.post(GROQ_API_URL, json=payload, headers=headers)
        response.raise_for_status()  # Raise an error if the request failed
        result = response.json()
        return result.get("insights", "No insights available.")
    except requests.exceptions.RequestException as e:
        st.error(f"Error querying Groq API: {e}")
        return None

# Function to generate visualization
def generate_visualization(query, df):
    if 'trend' in query.lower():
        fig = px.line(df, x=df.columns[0], y=df.columns[1:])
    elif 'distribution' in query.lower():
        fig = px.box(df, x=df.columns[0], y=df.columns[1:])
    else:
        fig = px.scatter(df, x=df.columns[0], y=df.columns[1:])
    return fig

# Initialize session state for chat history and data
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'df' not in st.session_state:
    st.session_state.df = None

# Sidebar for file upload
st.sidebar.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    st.session_state.df = load_data(uploaded_file)
    st.session_state.chat_history.append(f"Uploaded file: {uploaded_file.name}")
    st.sidebar.success("File uploaded successfully!")

# Display the dataset preview
if st.session_state.df is not None:
    st.write("Dataset Preview:")
    st.dataframe(st.session_state.df.head())

# Sidebar for selecting the action (Analysis or Visualization)
st.sidebar.subheader("Choose Action")
action = st.sidebar.selectbox("What do you want to do?", ["Data Analysis", "Data Visualization"])

# Handle user queries for Data Analysis
if action == "Data Analysis":
    query = st.text_input("Ask a question about your data for analysis:")

    if query:
        # Add query to chat history
        st.session_state.chat_history.append(f"User: {query}")
        
        # Get analysis from Groq API (or your custom model)
        analysis = query_to_analysis(query, st.session_state.df)
        if analysis:
            st.session_state.chat_history.append(f"AI: {analysis}")
            st.subheader("AI's Insights:")
            st.write(analysis)

# Handle user queries for Data Visualization
elif action == "Data Visualization":
    query = st.text_input("Ask a question about your data for visualization:")

    if query:
        # Add query to chat history
        st.session_state.chat_history.append(f"User: {query}")
        
        # Generate and display visualization
        fig = generate_visualization(query, st.session_state.df)
        st.session_state.chat_history.append(f"AI: Visualization generated.")
        st.subheader("Visualization:")
        st.plotly_chart(fig)

# Display the chat history
st.sidebar.subheader("Chat History")
for message in st.session_state.chat_history:
    st.sidebar.write(message)

# Sandbox-like environment for continuous interaction
st.sidebar.markdown("### Chat with Your Data")
st.sidebar.markdown("Ask questions about your data in natural language.")

# Allow users to clear chat history (optional)
if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.sidebar.success("Chat history cleared!")
