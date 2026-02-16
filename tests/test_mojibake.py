from clarityocr.converter import fix_mojibake

def test_fix_mojibake():
    # Example Russian text: "Основная" misread as Latin-1
    # Needs to be long enough to pass the 50-char threshold
    mojibake = "Îñíîâíàÿ çàäà÷à " * 10
    fixed = fix_mojibake(mojibake)

    # It should detect and fix it
    assert "Основная" in fixed or "основная" in fixed.lower()

def test_no_mojibake_for_english():
    text = "This is a normal English sentence."
    assert fix_mojibake(text) == text

def test_no_mojibake_for_french():
    text = "C'est l'été à Paris."
    # French uses high latin-1 but it shouldn't be "fixed" to Cyrillic
    assert fix_mojibake(text) == text
