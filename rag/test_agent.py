from dotenv import load_dotenv
import warnings
import logging

load_dotenv()
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from agent import ask_agent

LABELS = {
    "scheme_name":         "Scheme Name",
    "description":         "Description",
    "category":            "Category",
    "benefits":            "Benefits",
    "eligibility":         "Eligibility",
    "documents_required":  "Documents Required",
    "application_process": "Application Process",
    "state":               "State",
    "official_link":       "Official Link",
}

def print_full_scheme(data: dict, index: int):
    print(f"\n{'─' * 55}")
    print(f"  SCHEME {index}: {data.get('scheme_name', '')}")
    print(f"{'─' * 55}")
    for key, label in LABELS.items():
        if key == "scheme_name":
            continue
        value = data.get(key, "Not Available")
        if value and value != "Not Available":
            print(f"\n📌 {label}:\n   {value}")
    print(f"\n{'─' * 55}")

def print_response(result: dict):
    rtype = result.get("type")

    if rtype == "conversational":
        print(f"\n🤖 {result['reply']}\n")

    elif rtype == "names_only":
        print(f"\n🤖 Here are the schemes:\n\n{result['reply']}\n")

    elif rtype == "specific_field":
        field = result.get("field", "").replace("_", " ").title()
        print(f"\n🤖 {field}:\n\n{result['reply']}\n")

    elif rtype == "full_detail":
        schemes = result.get("schemes", [])
        print(f"\n🤖 Found {len(schemes)} scheme(s):")
        for i, scheme in enumerate(schemes, 1):
            print_full_scheme(scheme, i)


print("\n🇮🇳  AI Government Scheme Assistant")
print("─" * 40)
print("Ask anything about Indian government schemes.")
print("Type 'exit' to quit.\n")

while True:
    q = input("You: ").strip()

    if not q:
        continue
    if q.lower() == "exit":
        print("Goodbye! 👋")
        break

    result = ask_agent(q)
    print_response(result)