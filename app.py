import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from groq import Groq
from dotenv import load_dotenv
import os

# Load API Key
load_dotenv()

client = Groq(
    api_key=os.getenv("your_api_key")
)

st.title("📊 AI Data Visualization Assistant")

# Upload CSV
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file:

    # Read CSV
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df)

    st.subheader("📈 Basic Dataset Info")

    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

    st.write("Column Names:")
    st.write(df.columns.tolist())

    # Select columns
    numeric_columns = df.select_dtypes(
        include=['number']
    ).columns.tolist()

    if len(numeric_columns) >= 2:

        x_axis = st.selectbox(
            "Select X-axis",
            numeric_columns
        )

        y_axis = st.selectbox(
            "Select Y-axis",
            numeric_columns
        )

        chart_type = st.selectbox(
            "Select Chart Type",
            ["Line", "Bar", "Scatter"]
        )

        fig, ax = plt.subplots()

        if chart_type == "Line":
            ax.plot(df[x_axis], df[y_axis])

        elif chart_type == "Bar":
            ax.bar(df[x_axis], df[y_axis])

        elif chart_type == "Scatter":
            ax.scatter(df[x_axis], df[y_axis])

        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)

        st.pyplot(fig)

    # AI Insights
    st.subheader("🤖 AI Data Insights")

    prompt = f"""
    Analyze this dataset briefly:

    Columns:
    {df.columns.tolist()}

    First 5 rows:
    {df.head().to_string()}

    Give simple insights.
    """

    if st.button("Generate AI Insights"):

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        insights = response.choices[0].message.content

        st.write(insights)
