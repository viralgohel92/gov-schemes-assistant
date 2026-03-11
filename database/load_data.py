import pandas as pd
from db import SessionLocal
from models import Scheme

df = pd.read_csv("data/processed/scraped_schemes.csv")

db = SessionLocal()

for _, row in df.iterrows():

    scheme = Scheme(
        category=row["category"],
        scheme_name=row["scheme_name"],
        application_link=row["scheme_link"],
        description=row["details"],
        benefits=row["benefits"],
        eligibility=row["eligibility"],
        documents_required=row["documents_required"],
        application_process = row["application_process"],
        state = row["state"]
    )

    db.add(scheme)

db.commit()

print("Data inserted")