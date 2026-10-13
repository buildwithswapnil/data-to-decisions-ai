import os
from dotenv import load_dotenv

# WHY: This calculates the absolute path to your main project root directory.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# WHY: Explicitly pointing load_dotenv to the root .env file ensures 
# your API keys load perfectly no matter which subfolder you run your terminal from.
dotenv_path = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path)

# WHY: We fetch the key from the environment.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# WHY: Centralized database path calculation to prevent file-not-found errors.
DB_PATH = os.path.join(BASE_DIR, 'db', 'healthcare_data.db')

if __name__ == "__main__":
    print("--- Configuration Testing ---")
    print(f"Project Base Directory: {BASE_DIR}")
    print(f"Database Resolution Path: {DB_PATH}")
    if OPENAI_API_KEY:
        print("OpenAI API Key: Loaded Successfully (Hidden for security)")
    else:
        print("WARNING: OpenAI API Key NOT found! Check your .env file placement.")