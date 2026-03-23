# Import Libraries
import os
from dotenv import load_dotenv

load_dotenv()

def get_env_var(
    name: str,
    default: str | None = None
) -> str:
    """
    Retrieve the value of an environment variable.

    This function loads environment variables from a `.env` file (if present)
    and returns the value of the requested variable. If the variable is not set
    and no default value is provided, a runtime error is raised.

    Parameters
    ----------
    name: str
        Name of the environment variable to retrieve.
    default: str or None, optional
        Default value to return if the environment variable is not set.
        If None and the variable is missing, an error is raised.

    Returns
    -------
    str
        Value of the environment variable.

    Raises
    ------
    RuntimeError
        If the environment variable is not set and no default is provided.
    """
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"Required environment variable '{name}' is not set")
    return value