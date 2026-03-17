import json
import re
from typing import Dict, Any

def extract_headings(md_text: str) -> list:
    """Extract all markdown headings."""
    return re.findall(r'^(#{1,6})\s+(.*)$', md_text, re.MULTILINE)

def extract_entities(md_text: str) -> dict:
    """Extract common entities using regex (heuristic confidence)."""
    entities = {
        "emails": list(set(re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', md_text))),
        "dates": list(set(re.findall(r'\b\d{4}[-/]\d{2}[-/]\d{2}\b|\b\d{2}[-/]\d{2}[-/]\d{4}\b', md_text))),
        "amounts": list(set(re.findall(r'[$€£¥]\s?\d+(?:,\d{3})*(?:\.\d{2})?', md_text)))
    }
    return entities

def has_tables(md_text: str) -> bool:
    """Detect presence of markdown tables."""
    return bool(re.search(r'\|.*\|.*\|', md_text))

def generate_metadata(md_path: str) -> Dict[str, Any]:
    """Generates metadata for the given markdown file."""
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            md_text = f.read()
    except Exception:
        return {"error": "Could not read markdown file"}

    headings = extract_headings(md_text)
    entities = extract_entities(md_text)
    tables_present = has_tables(md_text)

    # v2 Contract: Document the confidence source
    meta = {
        "schema_version": "1.0",
        "confidence_source": "ocr-native+heuristic",
        "document_stats": {
            "char_count": len(md_text),
            "word_count": len(md_text.split()),
            "headings_count": len(headings),
            "has_tables": tables_present
        },
        "extracted_entities": {
            "emails": entities["emails"],
            "dates": entities["dates"],
            "amounts": entities["amounts"]
        },
        "confidence_sources": {
            "entities": "heuristic",
            "text": "ocr-native",
            "tables": "heuristic" # or ocr-native depending on marker-pdf's exact output
        }
    }
    
    return meta
