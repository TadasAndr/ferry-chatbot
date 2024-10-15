import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Function to check if we're running on Streamlit Cloud
def is_streamlit_cloud():
    return os.environ.get('STREAMLIT_RUNTIME') == 'streamlit_cloud'

# Load from .env file if running locally
if not is_streamlit_cloud():
    load_dotenv()

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    PINECONE_API_KEY: str
    RETRIEVER_K: int = 5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @classmethod
    def load(cls):
        if is_streamlit_cloud():
            # We're on Streamlit Cloud, use st.secrets
            import streamlit as st
            env_vars = {}
            for field in cls.__fields__:
                value = st.secrets.get(field)
                if value is not None:
                    env_vars[field] = value
                else:
                    print(f"Warning: {field} not found in Streamlit secrets")
        else:
            # We're running locally, use environment variables
            env_vars = {}
            for field in cls.__fields__:
                value = os.getenv(field)
                if value is not None:
                    env_vars[field] = value
                else:
                    print(f"Warning: {field} not found in environment variables")
        
        # Create and return an instance of Settings
        return cls(**env_vars)

# Create the config object
try:
    config = Settings.load()
except Exception as e:
    error_msg = f"Error loading configuration: {str(e)}\n"
    if is_streamlit_cloud():
        error_msg += "Make sure you have set up your secrets in the Streamlit Cloud dashboard."
    else:
        error_msg += "Make sure you have set up your .env file or environment variables correctly."
    raise Exception(error_msg)

# Validate that required fields are present
missing_fields = [field for field in ['OPENAI_API_KEY', 'PINECONE_API_KEY'] if getattr(config, field, None) is None]
if missing_fields:
    raise Exception(f"Missing required configuration fields: {', '.join(missing_fields)}")