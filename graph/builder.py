import networkx as nx
import pickle
import os
from typing import List, Dict, Optional
from tree_sitter import Node
from ingestion.parser import CodeParser
from utils.models import Chunk
from utils.logger import get_logger

logger = get_logger(__name__)

class GraphBuilder:
    """Builds a NetworkX DiGraph from Python source code."""
    
    def __init__(self, repo_path: str, chunks: List[Chunk]):
        self.repo_path = os.path.abspath(repo_path)
        self.chunks = chunks
        self.parser = CodeParser()
        self.graph = nx.DiGraph()
        self.chunk_map: Dict[str, Chunk] = {c.id: c for c in chunks}
        
    def build(self):
        """Constructs the knowledge graph nodes and edges."""
        logger.info("Building knowledge graph...")
        
        # 1. Add all chunks as nodes and link them to their parent files
        for chunk in self.chunks:
            self.graph.add_node(chunk.id, metadata=chunk.metadata.model_dump())
            
            # Create a logical FILE node to group chunks together
            file_node_id = f"FILE:{chunk.metadata.filepath}"
            if not self.graph.has_node(file_node_id):
                self.graph.add_node(file_node_id, type="file", filepath=chunk.metadata.filepath)
                
            # Add an edge showing this file CONTAINS this chunk
            self.graph.add_edge(file_node_id, chunk.id, type="CONTAINS")
            
        # 2. Extract edges by parsing files again (or using chunk text if self-contained)
        # To find accurate relationships across files, we parse the raw files.
        # For simplicity in this implementation, we look for simple call/inherit patterns.
        
        # Keep track of where classes/functions are defined globally to resolve cross-file edges
        global_defs = {}
        for chunk in self.chunks:
            # Map bare symbol name to chunk_id (naive resolution for a single-repo scope)
            # In a real compiler, we'd resolve imports perfectly.
            # Here, we store short name -> id
            short_name = chunk.metadata.symbol_name.split('.')[-1]
            global_defs[short_name] = chunk.id
            
        for chunk in self.chunks:
            self._extract_edges(chunk, global_defs)
            
        logger.info(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
        
    def _extract_edges(self, chunk: Chunk, global_defs: Dict[str, str]):
        """Parses a chunk's AST to find calls and inheritances."""
        # Use the file extension from the chunk metadata to select the right parser
        tree = self.parser.parse(chunk.text.encode('utf-8'), filepath=chunk.metadata.filepath)
        source_bytes = chunk.text.encode('utf-8')
        
        def traverse(node: Node):
            # Find function calls (Python: 'call', JS/TS: 'call_expression')
            if node.type in ['call', 'call_expression']:
                func_name_node = node.children[0]
                
                called_name = None
                
                # Direct call e.g., foo()
                if func_name_node.type == 'identifier':
                    called_name = source_bytes[func_name_node.start_byte:func_name_node.end_byte].decode('utf-8')
                
                # Attribute/member call e.g., obj.foo()
                elif func_name_node.type in ['attribute', 'member_expression']:
                    # We just take the last part of the attribute (property identifier)
                    for child in reversed(func_name_node.children):
                        if child.type in ['identifier', 'property_identifier']:
                            called_name = source_bytes[child.start_byte:child.end_byte].decode('utf-8')
                            break
                            
                if called_name and called_name in global_defs and global_defs[called_name] != chunk.id:
                    self.graph.add_edge(chunk.id, global_defs[called_name], type='CALLS')
                            
            # Find inheritance (Python: 'class_definition', JS/TS: 'class_declaration')
            if node.type in ['class_definition', 'class_declaration'] and chunk.metadata.symbol_type == 'class':
                # look for argument_list (base classes in Python) or class_heritage (JS/TS)
                for child in node.children:
                    if child.type in ['argument_list', 'class_heritage']:
                        for arg in child.children:
                            if arg.type == 'identifier':
                                base_name = source_bytes[arg.start_byte:arg.end_byte].decode('utf-8')
                                if base_name in global_defs and global_defs[base_name] != chunk.id:
                                    self.graph.add_edge(chunk.id, global_defs[base_name], type='INHERITS')
                                    
            for child in node.children:
                traverse(child)
                
        traverse(tree.root_node)
        
    def save(self, path: str):
        """Saves the graph to disk using pickle."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.graph, f)
        logger.info(f"Saved graph to {path}")
        
    @staticmethod
    def load(path: str) -> nx.DiGraph:
        """Loads the graph from disk."""
        if not os.path.exists(path):
            logger.warning(f"Graph file not found at {path}")
            return nx.DiGraph()
            
        with open(path, 'rb') as f:
            graph = pickle.load(f)
        logger.info(f"Loaded graph with {graph.number_of_nodes()} nodes.")
        return graph
