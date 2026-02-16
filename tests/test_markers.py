import pytest
from clarityocr.converter import normalize_page_markers, count_page_markers

def test_normalize_page_markers():
    # Test {N} format
    assert "[p:1]" in normalize_page_markers("{1}\nText")

    # Test {N}--- block format
    assert "[p:12]" in normalize_page_markers("{12}\n---\nText")

    # Test 0-based shift
    normalized = normalize_page_markers("{0}\nText\n{1}\nMore")
    assert "[p:1]" in normalized
    assert "[p:2]" in normalized
    assert "[p:0]" not in normalized

def test_count_page_markers():
    text = "[p:1]\nText\n[p:5]\nMore"
    assert count_page_markers(text) == 5

    assert count_page_markers("No markers") == 0
