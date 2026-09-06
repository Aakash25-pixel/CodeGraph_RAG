import networkx as nx
from typing import List, Set
from utils.logger import get_logger

logger = get_logger(__name__)

class GraphTraverser:
    """Utility class to traverse the CodeGraph knowledge graph."""
    
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        
    def _get_neighbors_by_edge_type(self, node_id: str, edge_type: str, direction: str) -> List[str]:
        """Helper to get neighbors filtered by edge type and direction (in/out)."""
        if node_id not in self.graph:
            return []
            
        neighbors = []
        edges = self.graph.in_edges(node_id, data=True) if direction == 'in' else self.graph.out_edges(node_id, data=True)
        
        for u, v, data in edges:
            if data.get('type') == edge_type:
                neighbor = u if direction == 'in' else v
                neighbors.append(neighbor)
                
        return neighbors

    def get_callers(self, node_id: str) -> List[str]:
        """Returns IDs of functions/methods that call the given node."""
        return self._get_neighbors_by_edge_type(node_id, 'CALLS', 'in')
        
    def get_callees(self, node_id: str) -> List[str]:
        """Returns IDs of functions/methods called by the given node."""
        return self._get_neighbors_by_edge_type(node_id, 'CALLS', 'out')
        
    def get_ancestors(self, node_id: str) -> List[str]:
        """Returns IDs of classes this class inherits from (base classes)."""
        # Node INHERITS -> BaseClass
        return self._get_neighbors_by_edge_type(node_id, 'INHERITS', 'out')
        
    def get_descendants(self, node_id: str) -> List[str]:
        """Returns IDs of classes that inherit from this class (subclasses)."""
        # SubClass INHERITS -> Node
        return self._get_neighbors_by_edge_type(node_id, 'INHERITS', 'in')
        
    def get_context_subgraph(self, node_id: str, depth: int = 1) -> List[str]:
        """
        Retrieves a local subgraph context around a node up to a certain depth.
        Returns a list of node IDs.
        """
        if node_id not in self.graph:
            return []
            
        # For a simple local context, we take ego_graph ignoring edge direction
        undirected = self.graph.to_undirected()
        ego = nx.ego_graph(undirected, node_id, radius=depth)
        return list(ego.nodes())
