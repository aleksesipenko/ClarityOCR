import pytest
from clarityocr.polish import estimate_tokens, is_table_line, split_into_blocks

def test_estimate_tokens():
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 40) == 10

def test_is_table_line():
    assert is_table_line("| col1 | col2 |") == True
    assert is_table_line("|---|---|") == True
    assert is_table_line("Regular text") == False

def test_split_into_blocks():
    text = "Para 1\n\n| t1 |\n|---|\n\nPara 2"
    blocks = split_into_blocks(text)
    assert len(blocks) == 3
    assert blocks[0][1] == "text"
    assert blocks[1][1] == "table"
    assert blocks[2][1] == "text"
