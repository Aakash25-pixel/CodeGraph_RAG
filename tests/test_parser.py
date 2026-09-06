import pytest
from ingestion.parser import CodeParser

def test_parser_initialization():
    parser = CodeParser()
    assert parser is not None

def test_parse_simple_function():
    parser = CodeParser()
    code = b"def test_func():\n    pass\n"
    tree = parser.parse(code)
    assert tree is not None
    assert tree.root_node.type == 'module'
