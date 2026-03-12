from dotenv import load_dotenv
import warnings
import logging

load_dotenv()
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from agent import ask_agent

LABELS = {
    "scheme_name":        "Scheme Name",
    "description":        "Description",
    "category":           "Category",
    "benefits":           "Benefits",
    "eligibility":        "Eligibility",
    "documents_required": "Documents Required",
    "application_process":"Application Process",
    "state":              "State",
    "official_link":      "Official Link",
}

def print_scheme(data: dict, index: int):
    print(f"\n{'=' * 60}")
    print(f"  SCHEME {index}")
    print(f"{'=' * 60}")
    for key, label in LABELS.items():
        value = data.get(key, "Not Available")
        print(f"\n📌 {label}:\n   {value}")
    print(f"\n{'=' * 60}")


print("\n🇮🇳  AI Government Scheme Assistant")
print("Type 'exit' to stop\n")

while True:

    q = input("\nAsk a question: ").strip()

    if not q:
        continue

    if q.lower() == "exit":
        break

    schemes = ask_agent(q)

    print(f"\n✅ Found {len(schemes)} scheme(s)\n")

    for i, scheme in enumerate(schemes, start=1):
        print_scheme(scheme, i)