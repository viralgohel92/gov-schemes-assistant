import os
import sys
import datetime
from sqlalchemy import text

# Ensure project root is in path for imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from database.db import SessionLocal
from database.models import Scheme

def generate_sitemap():
    """Generates a sitemap.xml for Yojna AI based on static routes and database schemes."""
    base_url = "https://yojna-ai-seven.vercel.app"
    sitemap_path = os.path.join(REPO_ROOT, "frontend", "static", "sitemap.xml")
    
    # Static routes
    static_routes = [
        {"url": "/", "priority": "1.0", "changefreq": "daily"},
        {"url": "/app", "priority": "0.9", "changefreq": "weekly"},
    ]
    
    db = SessionLocal()
    try:
        # Note: Currently we don't have individual scheme pages yet, 
        # but once implemented, they would be added here.
        # For now, we just index the main functional pages.
        
        xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n'
        xml_content += '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        
        # Add static routes
        for route in static_routes:
            xml_content += f'  <url>\n'
            xml_content += f'    <loc>{base_url}{route["url"]}</loc>\n'
            xml_content += f'    <lastmod>{datetime.datetime.now().strftime("%Y-%m-%d")}</lastmod>\n'
            xml_content += f'    <changefreq>{route["changefreq"]}</changefreq>\n'
            xml_content += f'    <priority>{route["priority"]}</priority>\n'
            xml_content += f'  </url>\n'
            
        xml_content += '</urlset>'
        
        with open(sitemap_path, "w", encoding="utf-8") as f:
            f.write(xml_content)
            
        print(f"  Success: Sitemap generated at {sitemap_path}")
        
    except Exception as e:
        print(f"  Error generating sitemap: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    generate_sitemap()
