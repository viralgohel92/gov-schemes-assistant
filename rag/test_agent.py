from dotenv import load_dotenv
import warnings
import logging

# Load environment variables
load_dotenv()

warnings.filterwarnings("ignore")

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from agent import ask_agent


while True:

    q = input("\nAsk a question: ")

    if q.lower() == "exit":
        break

    answer = ask_agent(q)

    print("\nAI:", answer)