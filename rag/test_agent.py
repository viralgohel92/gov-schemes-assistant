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

PROFILE_LABELS = {
    "age":            "Age",
    "income":         "Annual Income",
    "occupation":     "Occupation",
    "state":          "State",
    "gender":         "Gender",
    "caste_category": "Caste / Category",   # ← clear label
    "extra":          "Other Info",
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

def print_profile_summary(profile: dict):
    filled = {k: v for k, v in profile.items() if v}
    if not filled:
        return
    print("\n  👤 Your Profile:")
    for k, v in filled.items():
        label = PROFILE_LABELS.get(k, k.replace("_", " ").title())
        print(f"     • {label}: {v}")

def print_eligible_scheme(data: dict, index: int):
    print(f"\n  {'─' * 50}")
    print(f"  {index}. {data.get('scheme_name', 'Unknown')}")
    print(f"  {'─' * 50}")
    if data.get("why_eligible"):
        print(f"  ✅ Why eligible    : {data['why_eligible']}")
    if data.get("category"):
        print(f"  📂 Category        : {data['category']}")
    if data.get("state"):
        print(f"  📍 State           : {data['state']}")
    if data.get("official_link"):
        print(f"  🔗 Link            : {data['official_link']}")

def print_eligibility_check_result(data: dict, index: int):
    is_eligible = data.get("is_eligible", False)
    icon = "✅" if is_eligible else "❌"
    print(f"\n  {icon} {index}. {data.get('scheme_name', 'Unknown')}")
    print(f"       {data.get('reason', '')}")
    if is_eligible and data.get("official_link"):
        print(f"       🔗 {data['official_link']}")

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

    elif rtype == "eligibility_result":
        profile = result.get("profile", {})
        schemes = result.get("schemes", [])
        print(f"\n{'═' * 55}")
        print(f"  🎯 ELIGIBILITY RESULTS")
        print(f"{'═' * 55}")
        print_profile_summary(profile)
        print(f"\n  ✅ {len(schemes)} Eligible Scheme(s) Found:\n")
        for i, scheme in enumerate(schemes, 1):
            print_eligible_scheme(scheme, i)
        print(f"\n{'═' * 55}")
        print(f"  💡 Ask for full details of any scheme above.")
        print(f"{'═' * 55}\n")

    elif rtype == "eligibility_for_shown":
        profile = result.get("profile", {})
        schemes = result.get("schemes", [])
        eligible = [s for s in schemes if s.get("is_eligible")]
        not_eligible = [s for s in schemes if not s.get("is_eligible")]

        print(f"\n{'═' * 55}")
        print(f"  🎯 ELIGIBILITY CHECK FOR SHOWN SCHEMES")
        print(f"{'═' * 55}")
        print_profile_summary(profile)
        print(f"\n  Results ({len(schemes)} schemes checked):\n")
        for i, scheme in enumerate(schemes, 1):
            print_eligibility_check_result(scheme, i)
        print(f"\n  📊 Summary: ✅ {len(eligible)} eligible  |  ❌ {len(not_eligible)} not eligible")
        print(f"{'═' * 55}\n")


print("\n🇮🇳  AI Government Scheme Assistant")
print("─" * 55)
print("Commands:")
print('  • Find schemes      : "give me loan schemes for students"')
print('  • Check eligibility : "which scheme am I eligible for?"')
print('  • Your profile      : "age: 22, income: 1.5L, student, Gujarat, caste: OBC"')
print("─" * 55 + "\n")

while True:
    q = input("You: ").strip()
    if not q:
        continue
    if q.lower() == "exit":
        print("Goodbye! 👋")
        break

    result = ask_agent(q)
    print_response(result)