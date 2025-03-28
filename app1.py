import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class AgenticAIAnalyzer:
    def __init__(self, data):
        self.data = data
        self.cleaned_data = None
        self.logger = logging.getLogger(__name__)

    def clean_data(self):
        """Clean the data by handling missing values, duplicates, and outliers."""
        if self.data is None:
            self.logger.error("No data to clean. Load data first.")
            return

        # Drop duplicates
        self.data.drop_duplicates(inplace=True)

        # Handle missing values
        imputer = SimpleImputer(strategy='mean')  # Can be changed to 'median' or 'most_frequent'
        self.data = pd.DataFrame(imputer.fit_transform(self.data), columns=self.data.columns)

        # Remove outliers using Z-score
        scaler = StandardScaler()
        z_scores = np.abs(scaler.fit_transform(self.data))
        self.data = self.data[(z_scores < 3).all(axis=1)]  # Keep data within 3 standard deviations

        self.cleaned_data = self.data
        self.logger.info("Data cleaned successfully.")

    def analyze_data(self):
        """Perform exploratory data analysis (EDA)."""
        if self.cleaned_data is None:
            self.logger.error("No cleaned data to analyze. Clean data first.")
            return

        # Basic statistics
        st.write("### Basic Statistics")
        st.write(self.cleaned_data.describe())

        # Correlation matrix
        st.write("### Correlation Matrix")
        st.write(self.cleaned_data.corr())

        # Principal Component Analysis (PCA) for dimensionality reduction
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(self.cleaned_data)
        self.cleaned_data['PCA1'] = pca_result[:, 0]
        self.cleaned_data['PCA2'] = pca_result[:, 1]

        # Clustering using KMeans
        kmeans = KMeans(n_clusters=3)
        self.cleaned_data['Cluster'] = kmeans.fit_predict(self.cleaned_data)

    def visualize_data(self):
        """Visualize the data using Matplotlib."""
        if self.cleaned_data is None:
            self.logger.error("No cleaned data to visualize. Clean data first.")
            return

        # Pairplot for relationships
        st.write("### Pairplot of Data with Clusters")
        fig, ax = plt.subplots()
        sns.pairplot(self.cleaned_data, hue='Cluster', ax=ax)
        st.pyplot(fig)

        # PCA Visualization
        st.write("### PCA Visualization with Clusters")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=self.cleaned_data, palette='viridis', ax=ax)
        plt.title("PCA Visualization with Clusters")
        st.pyplot(fig)

# Streamlit App
def main():
    st.title("Agentic AI Data Analysis and Visualization")
    st.sidebar.title("Settings")

    # Initialize session state for analyzer and chat history
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar for uploading dataset
    uploaded_file = st.sidebar.file_uploader("Upload a dataset (CSV or Excel)", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                data = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload a CSV or Excel file.")
                return

            st.write("### Raw Data")
            st.write(data)

            # Initialize the analyzer if not already done
            if st.session_state.analyzer is None:
                st.session_state.analyzer = AgenticAIAnalyzer(data)

            # Clean data
            if st.sidebar.button("Clean Data"):
                st.session_state.analyzer.clean_data()
                st.write("### Cleaned Data")
                st.write(st.session_state.analyzer.cleaned_data)

            # Analyze data
            if st.sidebar.button("Analyze Data"):
                st.session_state.analyzer.analyze_data()

            # Visualize data
            if st.sidebar.button("Visualize Data"):
                st.session_state.analyzer.visualize_data()

        except Exception as e:
            st.error(f"Error processing file: {e}")

    # Chat history
    st.sidebar.write("### Chat History")
    for message in st.session_state.chat_history:
        st.sidebar.write(message)

    # Chat input
    user_input = st.text_input("Ask a question or provide feedback:")
    if user_input:
        st.session_state.chat_history.append(f"User: {user_input}")
        if "clean" in user_input.lower():
            if st.session_state.analyzer is not None:
                st.session_state.analyzer.clean_data()
                st.session_state.chat_history.append("Agent: Data cleaned successfully.")
            else:
                st.session_state.chat_history.append("Agent: No data loaded. Please upload a dataset first.")
        elif "analyze" in user_input.lower():
            if st.session_state.analyzer is not None and st.session_state.analyzer.cleaned_data is not None:
                st.session_state.analyzer.analyze_data()
                st.session_state.chat_history.append("Agent: Data analyzed successfully.")
            else:
                st.session_state.chat_history.append("Agent: No cleaned data available. Please clean the data first.")
        elif "visualize" in user_input.lower():
            if st.session_state.analyzer is not None and st.session_state.analyzer.cleaned_data is not None:
                st.session_state.analyzer.visualize_data()
                st.session_state.chat_history.append("Agent: Data visualized successfully.")
            else:
                st.session_state.chat_history.append("Agent: No cleaned data available. Please clean the data first.")
        else:
            st.session_state.chat_history.append("Agent: I'll process your request.")
        st.experimental_rerun()

if __name__ == "__main__":
    main()