# src/data_loader.py
"""
Load caterer data from CSV files into LangChain Documents.
"""

import re
import pandas as pd
from pathlib import Path
from typing import List
from langchain_core.documents import Document


def parse_rating(rating_val) -> float:
    """Extract float rating like 4.8 from strings like '4.8/5 (Estimated)'."""
    if pd.isna(rating_val):
        return 0.0
    match = re.search(r"(\d+(\.\d+)?)", str(rating_val))
    return float(match.group(1)) if match else 0.0


def load_all_documents(data_dir: str) -> List[Document]:
    """
    Load all CSV files from data_dir and convert each row to a Document.

    Column name mapping (actual CSV headers → metadata keys):
      'Caterer Name'   → caterer_name
      'State/Area'     → state_area     ← NOTE: no spaces around /
      'Region/Zone'    → region_zone    ← NOTE: no spaces around /
      'Contact Number' → contact_phone
      'Email'          → contact_email
      'Website URL'    → website
      'Specialization' → specialization
      'Budget Tier'    → budget_tier
      'Guest Capacity' → guest_capacity
      'Status'         → status
      'Rating'         → rating_raw / rating_score
    """

    data_path = Path(data_dir).resolve()
    csv_files = list(data_path.glob("**/*.csv"))
    documents = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        # Strip whitespace from all string cells
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

        for _, row in df.iterrows():

            def get_val(col: str) -> str:
                val = row.get(col)
                return "" if pd.isna(val) or str(val).strip() == "" else str(val).strip()

            # ── Core fields ────────────────────────────────────────
            name        = get_val("Caterer Name") or "Unknown Caterer"
            # Actual CSV columns use no spaces around slash
            state_area  = get_val("State/Area")
            region_zone = get_val("Region/Zone")
            specialty   = get_val("Specialization")
            budget      = get_val("Budget Tier")
            capacity    = get_val("Guest Capacity")
            rating_str  = get_val("Rating")
            status      = get_val("Status")
            phone       = get_val("Contact Number")
            email       = get_val("Email")
            website     = get_val("Website URL")

            # ── Semantic text ──────────────────────────────────────
            # Structure with explicit domain anchors so the embedding
            # model correctly maps location queries to caterer records.
            location_text = ", ".join(
                filter(None, [state_area, region_zone])
            ) or "Location not specified"

            semantic_text = (
                f"Business Name: {name}. "
                f"Operating Location: {location_text}. "
                f"Culinary Specialization: {specialty or 'Not specified'}. "
                f"Pricing & Scale: Budget Tier is {budget or 'unspecified'}, "
                f"Capacity is {capacity or 'unspecified'}. "
                f"Customer Rating: {rating_str or 'Unrated'}. "
                f"Status: {status or 'Active'}."
            )

            # ── Structured metadata ────────────────────────────────
            metadata = {
                "caterer_name":  name,
                "state_area":    state_area.lower(),
                "region_zone":   region_zone.lower(),
                "address":       location_text,          # combined for display
                "specialization": specialty,
                "budget_tier":   budget,
                "guest_capacity": capacity,
                "rating_raw":    rating_str,
                "rating_score":  parse_rating(rating_str),
                "status":        status,
                "contact_phone": phone,
                "contact_email": email,
                "website":       website,
                "source":        csv_file.name,
            }

            documents.append(
                Document(page_content=semantic_text, metadata=metadata)
            )

    print(
        f"[INFO] Loaded {len(documents)} documents "
        f"from {len(csv_files)} CSV file(s) in {data_path}"
    )
    return documents