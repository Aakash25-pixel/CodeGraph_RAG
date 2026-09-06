import os
import tree_sitter_python
import tree_sitter_javascript
import tree_sitter_typescript
import tree_sitter_html
import tree_sitter_css
from tree_sitter import Language, Parser, Tree
from utils.logger import get_logger

logger = get_logger(__name__)

# Map extensions to their respective tree-sitter language functions
LANGUAGE_MAP = {
    ".py": lambda: Language(tree_sitter_python.language()),
    ".js": lambda: Language(tree_sitter_javascript.language()),
    ".jsx": lambda: Language(tree_sitter_javascript.language()),
    ".ts": lambda: Language(tree_sitter_typescript.language_typescript()),
    ".tsx": lambda: Language(tree_sitter_typescript.language_tsx()),
    ".html": lambda: Language(tree_sitter_html.language()),
    ".css": lambda: Language(tree_sitter_css.language()),
}

class CodeParser:
    """Wrapper for tree-sitter parsing with polyglot support."""
    
    def __init__(self):
        self.parsers = {}
        try:
            # Pre-initialize common languages (lazy-loading could also work, but this is fine)
            for ext, lang_func in LANGUAGE_MAP.items():
                lang = lang_func()
                parser = Parser(lang)
                self.parsers[ext] = parser
        except Exception as e:
            logger.error(f"Failed to initialize tree-sitter parsers: {e}")
            raise

    def parse(self, source_code: bytes, filepath: str = ".py") -> Tree:
        """
        Parses source code into a tree-sitter AST.
        
        Args:
            source_code: The raw bytes of the source code.
            filepath: The path or extension to determine the language.
            
        Returns:
            The tree-sitter Tree object.
        """
        ext = os.path.splitext(filepath)[1].lower()
        parser = self.parsers.get(ext)
        
        if not parser:
            logger.warning(f"No parser found for extension {ext}, falling back to Python.")
            parser = self.parsers.get(".py")
            
        return parser.parse(source_code)
