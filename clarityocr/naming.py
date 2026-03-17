import os
import re
import hashlib
from typing import Optional, Tuple, Dict
from pathlib import Path

# Provide a lightweight romanize/slugify to avoid huge dependencies if possible
# Since we just want safe basic slugs:
def slugify(text: str) -> str:
    # A robust enough slugifier that handles basic cases
    text = text.lower()
    # Very basic romanization for Cyrillic as an example
    chars = {
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'e', 'ж': 'zh',
        'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n', 'о': 'o',
        'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u', 'ф': 'f', 'х': 'h', 'ц': 'ts',
        'ч': 'ch', 'ш': 'sh', 'щ': 'sch', 'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu',
        'я': 'ya'
    }
    romanized = ''.join(chars.get(c, c) for c in text)
    # Remove non-alphanumeric, replace spaces with hyphens
    safe = re.sub(r'[^a-z0-9\s-]', '', romanized)
    return re.sub(r'[\s-]+', '-', safe).strip('-')

BLACKLIST_TOKENS = {"scan", "img", "new", "final", "untitled", "document", "file"}

def infer_title_from_text(md_text: str) -> Optional[str]:
    """Look for headers or prominent text to act as a title."""
    # Look for first H1
    h1_match = re.search(r'^#\s+(.*?)$', md_text, re.MULTILINE)
    if h1_match:
        return h1_match.group(1).strip()
    return None

def infer_title_from_filename(filename: str) -> str:
    """Fallback to original filename if no title is found."""
    name = Path(filename).stem
    # Replace underscores/hyphens with spaces
    name = re.sub(r'[\-_]', ' ', name)
    return name.strip()

def generate_naming(input_path: str, md_text: Optional[str] = None, output_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Generate deterministic naming, applying blacklist and collision policies.
    """
    title_human = None
    reason_short = "fallback_filename"
    
    # 1. Try to get title from Markdown text if available
    if md_text:
        title_human = infer_title_from_text(md_text)
        if title_human:
            reason_short = "extracted_heading"
            
    # 2. Fallback to filename
    if not title_human:
        title_human = infer_title_from_filename(input_path)
        
    # 3. Create Slug
    slug = slugify(title_human)
    
    # 4. Filter against blacklist (if slug is entirely made of blacklisted words, hash it)
    slug_parts = slug.split('-')
    valid_parts = [p for p in slug_parts if p not in BLACKLIST_TOKENS]
    
    if not valid_parts:
        slug = f"doc-{hashlib.md5(title_human.encode()).hexdigest()[:6]}"
        reason_short += "_blacklisted"
    else:
        slug = "-".join(valid_parts)
        
    # 5. Collision Policy
    if output_dir:
        base_slug = slug
        counter = 1
        while os.path.exists(os.path.join(output_dir, f"{slug}.md")):
            slug = f"{base_slug}-{counter}"
            counter += 1
            if counter > 1000: # Sanity check
                slug = f"{base_slug}-{hashlib.md5(str(counter).encode()).hexdigest()[:4]}"
                break

    return {
        "title_human": title_human,
        "filename_slug": slug,
        "reason_short": reason_short
    }
