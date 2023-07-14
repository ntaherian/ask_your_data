from langchain.agents import create_csv_agent
from render import user_msg_container_html_template, bot_msg_container_html_template
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
import glob
from langchain.agents.agent_types import AgentType
import tempfile
from pandasai import PandasAI
from pandasai.llm.open_assistant import OpenAssistant
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import matplotlib
import plotly.graph_objects as go

def submit():
    st.session_state.input = st.session_state.widget
    st.session_state.widget = ''

def main():
    load_dotenv()
    
    # Load the OpenAI API key from the environment variable
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")
 
    st.set_page_config(page_title="Ask your CSVs",initial_sidebar_state="expanded")
    st.header("Ask your CSVs 📈")

    uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True)

    if uploaded_files:
    
        if 'input' not in st.session_state:
            st.session_state.input = ''
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
    
        # Create a temporary directory to store the uploaded files
        temp_dir = tempfile.mkdtemp()
        temp_path = temp_dir

        # Save the uploaded files to the temporary directory
        for i, uploaded_file in enumerate(uploaded_files):
            file_path = os.path.join(temp_path, f"file_{i}.csv")
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

        # Get all CSV files in the temporary directory
        csv_files = glob.glob(os.path.join(temp_path, "*.csv"))

        # Process the CSV files as desired
        for file_name in csv_files:
            file_path = os.path.join(temp_path, file_name)
            print(file_path)

        # Create an empty dictionary to store the DataFrames
        dataframes = {}

        # Iterate over the CSV files
        for file in csv_files:
            # Extract the file name without extension
            file_name = file.split('/')[-1].split('.')[0]
            
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file)
            
            # Store the DataFrame in the dictionary
            dataframes[file_name] = df
        

        agent = create_csv_agent(
            ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"), csv_files,     agent_type=AgentType.OPENAI_FUNCTIONS,verbose=True)
            
        df = pd.read_csv(uploaded_file)
            
        llm = OpenAI()

        # create PandasAI object, passing the LLM
        pandas_ai = PandasAI(llm, conversational=False, verbose=True)
        pandas_ai.clear_cache()
        
        user_question = st.text_input("Ask a question about your CSV: ",key='widget', on_change=submit)
        
        if st.session_state.input is not None and st.session_state.input != "":
            with st.spinner(text="In progress..."):
                if any(word in st.session_state.input for word in ["plot","chart","Plot","Chart"]):
                    st.session_state.input = st.session_state.input + ' ' + 'using seaborn'
                #st.write(agent.run(st.session_state.input))
                x = pandas_ai.run(list(dataframes.values()), st.session_state.input)
                fig = plt.gcf()
                #buffer = io.BytesIO()
                if fig.get_axes():
                    #buffer.seek(0)
                    # Display the image in Streamlit
                    #st.pyplot(fig)
                    st.session_state.chat_history.append({"message": st.session_state.input, "response": fig, "is_fig": True})
                    
                else:
                    st.write(x)
                    st.session_state.chat_history.append({"message": st.session_state.input, "response": x, "is_fig": False})
        
        for message in st.session_state.chat_history[::-1]:
            if message['is_fig']:
                st.write(user_msg_container_html_template.replace("$MSG", message['message']), unsafe_allow_html=True)
                st.pyplot(message['response'])
            else:
                st.write(user_msg_container_html_template.replace("$MSG", message['message']), unsafe_allow_html=True)
                st.write(bot_msg_container_html_template.replace("$MSG", str(message['response'])), unsafe_allow_html=True)

            
    else:
        st.write("No files uploaded.")
    


if __name__ == "__main__":
    main()
