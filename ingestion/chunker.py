import os
import json
from typing import List, Optional
from tree_sitter import Node
from ingestion.parser import CodeParser
from utils.models import Chunk, ChunkMetadata
from utils.logger import get_logger

logger = get_logger(__name__)

class RepoChunker:
    """Chunks a Python repository into functions and classes."""
    
    def __init__(self, repo_path: str):
        self.repo_path = os.path.abspath(repo_path)
        self.parser = CodeParser()
        self.chunks: List[Chunk] = []
        
    def _is_ignored(self, path: str) -> bool:
        """Check if a path should be ignored (e.g., venv, .git)."""
        ignored_dirs = {'.git', 'venv', '.venv', 'env', '__pycache__', 'node_modules'}
        parts = path.split(os.sep)
        return any(part in ignored_dirs for part in parts)
        
    def _extract_docstring(self, node: Node, source_bytes: bytes) -> Optional[str]:
        """Extracts docstring from a function or class node if present."""
        if not node.children:
            return None
            
        # The body is usually a block
        block_node = None
        for child in node.children:
            if child.type == 'block':
                block_node = child
                break
                
        if not block_node or not block_node.children:
            return None
            
        first_stmt = block_node.children[0]
        if first_stmt.type == 'expression_statement':
            string_node = first_stmt.children[0]
            if string_node.type == 'string':
                return source_bytes[string_node.start_byte:string_node.end_byte].decode('utf-8')
        return None

    def _get_node_name(self, node: Node, source_bytes: bytes) -> str:
        """Extracts the name of a function or class node."""
        if node.type == 'arrow_function':
            # arrow function name is typically the left side of the variable_declarator
            if node.parent and node.parent.type == 'variable_declarator':
                for child in node.parent.children:
                    if child.type == 'identifier':
                        return source_bytes[child.start_byte:child.end_byte].decode('utf-8')
                        
        for child in node.children:
            if child.type == 'identifier':
                return source_bytes[child.start_byte:child.end_byte].decode('utf-8')
        return "unknown"

    def _traverse_ast(self, node: Node, source_bytes: bytes, filepath: str, parent_class: Optional[str] = None):
        """Recursively traverses the AST to find classes and functions."""
        func_types = ['function_definition', 'async_function_definition', 'function_declaration', 'arrow_function', 'method_definition']
        class_types = ['class_definition', 'class_declaration']
        
        if node.type in func_types:
            name = self._get_node_name(node, source_bytes)
            docstring = self._extract_docstring(node, source_bytes)
            text = source_bytes[node.start_byte:node.end_byte].decode('utf-8')
            
            chunk_type = 'method' if parent_class else 'function'
            symbol_name = f"{parent_class}.{name}" if parent_class else name
            
            metadata = ChunkMetadata(
                filepath=filepath,
                symbol_name=symbol_name,
                symbol_type=chunk_type,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                parent_class=parent_class,
                docstring=docstring
            )
            
            chunk_id = f"{filepath}:{symbol_name}"
            self.chunks.append(Chunk(id=chunk_id, text=text, metadata=metadata))
            
            return

        if node.type in class_types:
            name = self._get_node_name(node, source_bytes)
            docstring = self._extract_docstring(node, source_bytes)
            text = source_bytes[node.start_byte:node.end_byte].decode('utf-8')
            
            metadata = ChunkMetadata(
                filepath=filepath,
                symbol_name=name,
                symbol_type='class',
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                parent_class=None,
                docstring=docstring
            )
            
            chunk_id = f"{filepath}:{name}"
            self.chunks.append(Chunk(id=chunk_id, text=text, metadata=metadata))
            
            # Traverse class body for methods
            # Note: depending on the language, the body node might not be exactly 'block'
            for child in node.children:
                if child.type in ['block', 'class_body']:
                    for block_child in child.children:
                        self._traverse_ast(block_child, source_bytes, filepath, parent_class=name)
            return

        # Continue traversing for other node types
        for child in node.children:
            self._traverse_ast(child, source_bytes, filepath, parent_class)

    def process_file(self, filepath: str):
        """Parses a single file and extracts chunks."""
        try:
            with open(filepath, 'rb') as f:
                source_bytes = f.read()
                
            tree = self.parser.parse(source_bytes, filepath)
            # Make path relative to repo root for consistent IDs
            rel_path = os.path.relpath(filepath, self.repo_path)
            
            func_types = ['function_definition', 'async_function_definition', 'function_declaration', 'arrow_function', 'method_definition']
            class_types = ['class_definition', 'class_declaration']
            
            # Extract module-level code (imports, global variables, pipeline definitions)
            module_code_bytes = b""
            for child in tree.root_node.children:
                if child.type not in func_types and child.type not in class_types:
                    module_code_bytes += source_bytes[child.start_byte:child.end_byte] + b"\n"
                    
            if module_code_bytes.strip():
                lines = module_code_bytes.decode('utf-8').split('\n')
                max_lines = 150
                for i in range(0, len(lines), max_lines):
                    chunk_text = '\n'.join(lines[i:i+max_lines]).strip()
                    if not chunk_text:
                        continue
                        
                    part_suffix = f"_part{i//max_lines + 1}" if len(lines) > max_lines else ""
                    metadata = ChunkMetadata(
                        filepath=rel_path,
                        symbol_name=f"<module_level>{part_suffix}",
                        symbol_type="module",
                        start_line=1,
                        end_line=tree.root_node.end_point[0] + 1,
                        parent_class=None,
                        docstring=None
                    )
                    self.chunks.append(Chunk(id=f"{rel_path}:<module_level>{part_suffix}", text=chunk_text, metadata=metadata))
                
            self._traverse_ast(tree.root_node, source_bytes, rel_path)
            
        except Exception as e:
            logger.warning(f"Failed to process {filepath}: {e}")

    def chunk_repo(self) -> List[Chunk]:
        """Scans the repository and chunks all supported files."""
        self.chunks = []
        logger.info(f"Scanning repository at {self.repo_path}")
        
        supported_exts = ('.py', '.js', '.jsx', '.ts', '.tsx', '.html', '.css')
        
        for root, dirs, files in os.walk(self.repo_path):
            if self._is_ignored(root):
                continue
                
            for file in files:
                if file.endswith(supported_exts):
                    filepath = os.path.join(root, file)
                    self.process_file(filepath)
                    
        logger.info(f"Extracted {len(self.chunks)} chunks from repository.")
        return self.chunks

    def save_chunks(self, path: str):
        """Saves chunks to a JSONL file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            for chunk in self.chunks:
                f.write(chunk.model_dump_json() + '\n')
        logger.info(f"Saved chunks to {path}")

    @staticmethod
    def load_chunks(path: str) -> List[Chunk]:
        """Loads chunks from a JSONL file."""
        chunks = []
        if not os.path.exists(path):
            return chunks
            
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                chunks.append(Chunk.model_validate_json(line))
        return chunks
