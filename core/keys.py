import os
import openai
from dotenv import load_dotenv


load_dotenv()


class InvalidApiKeyError(Exception):
    pass


def set_env_vars(file_path=".env", variables_dict={}):
    """
    Set environment variables programmatically in a .env file.

    Args:
        file_path (str): The path to the .env file.
        variables_dict (dict): A dictionary where keys are variable names and values are their corresponding values.
    """
    for variable, value in variables_dict.items():
        set_key(file_path, variable, value)


def load_api_key(env_var_name):
    """
    Load an API key from an environment variable.

    Parameters:
        env_var_name (str): The name of the environment variable containing the API key.

    Returns:
        str: The API key.
    """
    api_key = os.getenv(env_var_name)
    if not api_key:
        raise ValueError(f"{env_var_name} not found in environment variables.")
    return api_key


def init_openai_key():
    """
    Initialize the OpenAI API key from the environment variable.
    """
    openai_api_key = load_api_key("OPENAI_API_KEY")
    openai.api_key = openai_api_key


def init_groq_key():
    groq_api_key = load_api_key("GROQ_API_KEY")
    set_env_vars({"groq_api_key": groq_api_key})


def init_safeguards_key(key=""):
    """
    Initialize the Guardrail API key from the environment variable.
    """
    if key == "":
        safeguards_api_key = load_api_key("SAFEGUARDS_API_KEY")
        if safeguards_api_key == "":
            raise InvalidApiKeyError("Please enter a valid Guardrail API Key.")
    else:
        safeguards_api_key = key
    set_env_vars({"SAFEGUARDS_API_KEY": safeguards_api_key})


def init_perspective_key():
    """
    Initialize the Perspective API key from the environment variable.
    """
    perspective_api_key = load_api_key("PERSPECTIVE_API_KEY")

    set_env_vars({"PERSPECTIVE_API_KEY": perspective_api_key})
