"""
Test script to check relational 'schemes' table content.
"""
import os
from dotenv import load_dotenv
load_dotenv(".env")

try:
    from database.db import SessionLocal
    from database.models import Scheme
    
    session = SessionLocal()
    count = session.query(Scheme).count()
    print(f"  Total schemes in relational DB: {count}")
    
    if count > 0:
        print("\n  Sample schemes:")
        samples = session.query(Scheme).limit(5).all()
        for s in samples:
            print(f"   - {s.scheme_name} (Category: {s.category})")
            
        # Test specific search
        search = "education"
        pattern = f"%{search}%"
        from sqlalchemy import or_
        results = session.query(Scheme).filter(or_(Scheme.scheme_name.ilike(pattern), Scheme.category.ilike(pattern))).all()
        print(f"\n  Search for '{search}' returned {len(results)} results")
        for r in results[:3]:
            print(f"   - {r.scheme_name}")
            
except Exception as e:
    print(f"Error: {e}")
