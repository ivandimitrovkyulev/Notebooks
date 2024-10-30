import dotenv
import tomllib
from pathlib import Path


# Load all env variables
dotenv.load_dotenv()

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Read the TOML file
with open(f"{BASE_DIR}/pyproject.toml", "rb") as file:
    __toml_data__ = tomllib.load(file)
    __version__ = str(__toml_data__["tool"]["poetry"]["version"])
