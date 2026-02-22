import os
import hashlib
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Provide a robust image merge and PDF assembly using pypdfium2 and PIL
# PIL comes with marker-pdf which is in requirements
try:
    from PIL import Image, ExifTags
except ImportError:
    Image = None

import pypdfium2 as pdfium

def fix_exif_orientation(img_path: str) -> None:
    """Auto-rotate image based on EXIF and save it back."""
    if not Image:
        return
    try:
        with Image.open(img_path) as img:
            if not hasattr(img, '_getexif'):
                return
            exif = img._getexif()
            if not exif:
                return
                
            orientation_key = 274 # EXIF Orientation tag
            if orientation_key in exif:
                orientation = exif[orientation_key]
                if orientation == 3:
                    img = img.rotate(180, expand=True)
                elif orientation == 6:
                    img = img.rotate(270, expand=True)
                elif orientation == 8:
                    img = img.rotate(90, expand=True)
                else:
                    return # No rotation needed
                    
                img.save(img_path) # Overwrite with fixed orientation
    except Exception:
        pass # Corrupt image or unsupported format

def get_image_hash(img_path: str) -> str:
    """
    Get a hash of the image content for deduplication.
    In a full production impl, this could be a perceptual hash.
    For this MVP, we use sha256 of the raw file + pixel dimensions.
    """
    file_hash = hashlib.sha256()
    with open(img_path, 'rb') as f:
        file_hash.update(f.read())
    base_hash = file_hash.hexdigest()
    
    if Image:
        try:
            with Image.open(img_path) as img:
                return f"{base_hash}_{img.width}x{img.height}"
        except Exception:
            pass
    return base_hash

def get_file_creation_time(file_path: str) -> float:
    """Get modification/creation time. Tie-breaker."""
    try:
        return os.path.getmtime(file_path)
    except Exception:
        return 0.0

def sort_inputs(inputs: List[str], order_by: str = "filename") -> List[str]:
    """Sort inputs based on manual, filename, or exif/time."""
    if order_by == "manual":
        return inputs # Trust pre-sorted array
        
    def _sort_key(p: str) -> Tuple:
        path = Path(p)
        name_key = path.name.lower()
        if order_by == "exif_time":
            time_key = get_file_creation_time(p)
            # Find EXIF DateTimeOriginal if possible
            if Image and path.suffix.lower() in ['.jpg', '.jpeg']:
                try:
                    with Image.open(p) as img:
                        exif = img._getexif()
                        if exif and 36867 in exif: # DateTimeOriginal
                            # Simple string matching sort is fine for datetime format
                            time_key = exif[36867] 
                except Exception:
                    pass
            return (time_key, name_key)
        return (name_key,)

    return sorted(inputs, key=_sort_key)

def build_merged_pdf(sorted_inputs: List[str], output_pdf_path: str) -> Dict[str, Any]:
    """
    Merge a list of image/pdf paths into a single PDF.
    Implements deduplication and corrupt file skipping.
    """
    report = {
        "processed_files": 0,
        "skipped_corrupt": 0,
        "skipped_duplicate": 0,
        "warnings": [],
        "merged_pdf_path": output_pdf_path
    }
    
    seen_hashes = set()
    merged_pdf = pdfium.PdfDocument.new()
    
    for ipath in sorted_inputs:
        path_obj = Path(ipath)
        if not path_obj.exists():
            report["warnings"].append({"event": "skipping_missing_file", "file": ipath})
            report["skipped_corrupt"] += 1
            continue
            
        ext = path_obj.suffix.lower()
        
        # 1. Handle PDFs
        if ext == ".pdf":
            try:
                src_pdf = pdfium.PdfDocument(ipath)
                # Deduplicate based on PDF hash
                f_hash = hashlib.sha256(open(ipath, 'rb').read()).hexdigest()
                if f_hash in seen_hashes:
                    report["warnings"].append({"event": "duplicate_detected", "file": ipath})
                    report["skipped_duplicate"] += 1
                    continue
                    
                seen_hashes.add(f_hash)
                merged_pdf.import_pages(src_pdf)
                report["processed_files"] += 1
            except Exception as e:
                report["warnings"].append({"event": "skipping_corrupt_file", "file": ipath, "error": str(e)})
                report["skipped_corrupt"] += 1
                
        # 2. Handle Images
        elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
            try:
                fix_exif_orientation(ipath)
                img_hash = get_image_hash(ipath)
                
                if img_hash in seen_hashes:
                    report["warnings"].append({"event": "duplicate_detected", "file": ipath})
                    report["skipped_duplicate"] += 1
                    continue
                    
                seen_hashes.add(img_hash)
                
                # Convert image to PDF page
                if Image:
                    # pypdfium doesn't natively import images to blank pages easily without PIL to PDF conversion
                    # A robust way is to save image as PDF then import
                    temp_pdf = str(path_obj.with_suffix('.temp.pdf'))
                    with Image.open(ipath) as img:
                        if img.mode in ("RGBA", "P"):
                            img = img.convert("RGB")
                        img.save(temp_pdf, "PDF", resolution=100.0)
                    
                    src_pdf = pdfium.PdfDocument(temp_pdf)
                    merged_pdf.import_pages(src_pdf)
                    src_pdf.close()
                    os.remove(temp_pdf)
                else:
                    report["warnings"].append({"event": "skipping_corrupt_file", "file": ipath, "error": "PIL not installed for image conversion"})
                    report["skipped_corrupt"] += 1
                    continue
                    
                report["processed_files"] += 1
            except Exception as e:
                report["warnings"].append({"event": "skipping_corrupt_file", "file": ipath, "error": str(e)})
                report["skipped_corrupt"] += 1
                
        else:
            report["warnings"].append({"event": "skipping_unsupported", "file": ipath})
            
    # Save the final merged document
    if report["processed_files"] > 0:
        merged_pdf.save(output_pdf_path)
    else:
        report["warnings"].append({"event": "empty_output", "file": output_pdf_path})
        
    return report

def merge_pipeline(inputs: List[str], output_dir: str, order_by: str = "filename") -> Dict[str, Any]:
    """Orchestrates sorting and merging."""
    import json
    os.makedirs(output_dir, exist_ok=True)
    sorted_inputs = sort_inputs(inputs, order_by)
    out_pdf = os.path.join(output_dir, "merged.pdf")
    
    report = build_merged_pdf(sorted_inputs, out_pdf)
    
    report_path = os.path.join(output_dir, "merge_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
        
    return report
