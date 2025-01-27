from enum import Enum
import numpy as np
import heapq
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Set

class MaxHeap:
    """
    A max-heap implementation that wraps Python's heapq module.
    
    This class provides max-heap behavior while using Python's min-heap under the hood.
    Instead of negating numbers, we implement comparison methods to reverse the ordering.
    """
    class Item:
        def __init__(self, priority, value):
            self.priority = priority  # The priority (distance in our case)
            self.value = value       # The value (index in our case)
            
        def __lt__(self, other):
            # Reverse comparison for max-heap behavior
            return self.priority > other.priority
        
        def __eq__(self, other):
            return self.priority == other.priority
            
        def __repr__(self):
            return f"Item(priority={self.priority}, value={self.value})"
    
    def __init__(self):
        self._heap = []
    
    def push(self, priority, value):
        """Add an item to the heap."""
        heapq.heappush(self._heap, self.Item(priority, value))
    
    def pop(self):
        """Remove and return the (priority, value) pair with highest priority."""
        item = heapq.heappop(self._heap)
        return item.priority, item.value
    
    def peek(self):
        """Return the (priority, value) pair with highest priority without removing it."""
        if not self._heap:
            raise IndexError("Heap is empty")
        item = self._heap[0]
        return item.priority, item.value
    
    def __len__(self):
        return len(self._heap)
    
    def __bool__(self):
        return bool(self._heap)

    def __iter__(self):
        # Create a copy of the heap to avoid modifying the original
        heap_copy = self._heap.copy()
        # Sort the copy by priority (remember, our Item class handles comparison)
        return iter([(item.priority, item.value) for item in sorted(heap_copy)])

class SearchStrategy(Enum):
    SCORE_BASED = "score"
    PRIORITY_QUEUE = "priority"
    PRIORITY_QUEUE_HEAPQ = "priority_heapq"

class AnnoyNode:
    def __init__(self):
        self.left = None         # Left child node
        self.right = None        # Right child node
        self.split_plane = None  # Hyperplane normal vector
        self.split_threshold = None  # Threshold for splitting
        self.indices = []        # Indices of points stored in leaf nodes

class Annoy:
    def __init__(self, n_trees: int = 10, max_size: int = 10):
        """
        Initialize the Annoy index.
        
        Args:
            n_trees: Number of random projection trees to build
            max_size: Maximum number of points in a leaf node
        """
        self.n_trees = n_trees
        self.max_size = max_size
        self.trees = []
        self.data = None
        
    def _create_split_plane(self, points: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Create a random hyperplane for splitting points.
        
        Args:
            points: Array of points to split
            
        Returns:
            Tuple of (hyperplane normal vector, split threshold)
        """
        # Sample two random points to create splitting hyperplane
        idx = np.random.choice(len(points), 2, replace=False)
        p1, p2 = points[idx[0]], points[idx[1]]
        
        # Create normalized direction vector
        direction = p2 - p1
        direction = direction / np.linalg.norm(direction)
        
        # Calculate split threshold as median projection
        projections = points @ direction
        threshold = np.median(projections)
        
        return direction, threshold
    
    def _build_tree(self, points: np.ndarray, indices: List[int]) -> AnnoyNode:
        """
        Recursively build a tree by splitting points.
        
        Args:
            points: Array of points to split
            indices: List of indices for these points
            
        Returns:
            Root node of the built tree
        """
        node = AnnoyNode()
        
        if len(points) <= self.max_size:
            node.indices = indices
            return node
            
        # Create splitting hyperplane
        direction, threshold = self._create_split_plane(points)
        
        # Split points based on projections
        projections = points @ direction
        left_mask = projections <= threshold
        
        # Store split information
        node.split_plane = direction
        node.split_threshold = threshold
        
        # Recursively build left and right subtrees
        node.left = self._build_tree(points[left_mask], [indices[i] for i in range(len(indices)) if left_mask[i]])
        node.right = self._build_tree(points[~left_mask], [indices[i] for i in range(len(indices)) if not left_mask[i]])
        
        return node
    
    def build(self, data: np.ndarray):
        """
        Build the index with multiple trees.
        
        Args:
            data: Array of shape (n_points, n_dimensions) to index
        """
        self.data = data
        self.trees = []
        
        # Build multiple trees
        for _ in range(self.n_trees):
            indices = list(range(len(data)))
            tree = self._build_tree(data, indices)
            self.trees.append(tree)
    
    def _search_tree(self, node: AnnoyNode, query: np.ndarray, n_neighbors: int) -> List[Tuple[int, float]]:
        """
        Search a single tree for nearest neighbors.
        
        Args:
            node: Current tree node
            query: Query point
            n_neighbors: Number of neighbors to find
            
        Returns:
            List of (index, distance) tuples for nearest neighbors
        """
        if node.indices:  # Leaf node
            distances = [(idx, np.linalg.norm(self.data[idx] - query)) for idx in node.indices]
            return sorted(distances, key=lambda x: x[1])[:n_neighbors]
            
        # Determine which child to search first based on splitting plane
        projection = query @ node.split_plane
        first_child, second_child = (node.left, node.right) if projection <= node.split_threshold else (node.right, node.left)
        
        # Search first child
        best = self._search_tree(first_child, query, n_neighbors)
        
        # Check if we need to search the other child
        worst_dist = best[-1][1] if best else float('inf')
        margin = abs(projection - node.split_threshold)
        
        if margin < worst_dist:  # Search other child if it might contain better matches
            other_best = self._search_tree(second_child, query, n_neighbors)
            best = sorted(best + other_best, key=lambda x: x[1])[:n_neighbors]
            
        return best
    
    def _score_based_search(self, query: np.ndarray, n_neighbors: int = 1) -> List[Tuple[int, float]]:
        """
        Search for nearest neighbors across all trees.
        
        Args:
            query: Query point
            n_neighbors: Number of neighbors to return
            
        Returns:
            List of (index, distance) tuples for nearest neighbors
        """ 
        # Collect candidates from all trees
        candidates = defaultdict(float)
        for tree in self.trees:
            for idx, dist in self._search_tree(tree, query, n_neighbors):
                # Closer points get higher scores (distance ↓ = score ↑)
                # Score is always between 0 and 1
                candidates[idx] += 1.0 / (1.0 + dist)
                
        # Return top candidates based on aggregate scores
        best_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:n_neighbors]
        return [(idx, np.linalg.norm(self.data[idx] - query)) for idx, _ in best_candidates]
    
    def _priority_queue_search(self, query: np.ndarray, n_neighbors: int) -> List[Tuple[int, float]]:
        """
        Perform nearest neighbor search using a priority queue approach.
    
        This method implements an efficient search strategy that:
        1. Maintains a fixed-size priority queue of best candidates
        2. Processes each point exactly once using a seen set
        3. Uses early pruning to skip points that can't be in the final result
    
        The priority queue (min-heap) keeps track of the worst-case distance,
        allowing us to quickly determine if new points are worth considering.
    
        Args:
            query: The query point we're searching for neighbors of
            n_neighbors: Number of nearest neighbors to return
            
        Returns:
            List of (index, distance) tuples for the nearest neighbors,
            sorted by distance (closest first)
            
        Example:
            If searching for 2 nearest neighbors:
            - Heap size will be bounded by 4 (2 * n_neighbors)
            - Points further than the 4th best distance are skipped
            - Final result will be the 2 closest points found
        """
        # Initialize priority queue (max-heap) and set of seen indices
        heap = MaxHeap()  # Will store (distance, index) tuples
        seen: Set[int] = set()  # Track indices we've already processed
    
        # Search through each tree in our forest
        for tree in self.trees:
            # Get candidate points from this tree
            candidates = self._search_tree(tree, query, n_neighbors)
            
            # Process each candidate point
            for idx, dist in candidates:
                # Skip points we've already seen to ensure each point is processed once
                if idx not in seen:
                    # Check if heap is at capacity (2 * n_neighbors)
                    if len(heap) >= n_neighbors * 2:
                        # Get the worst distance in our current set of candidates
                        worst_dist, _ = heap.peek()  # Now directly gives us worst distance
                        
                        # Early pruning: skip points that are definitely too far away
                        if dist > worst_dist:
                            continue  # This point can't be in our final n_neighbors
                    
                    # Add new candidate to priority queue
                    heap.push(dist, idx)
                    seen.add(idx)
                    
                    # Maintain heap size bound of 2 * n_neighbors
                    if len(heap) > n_neighbors * 2:
                        # Remove the closest point (remember this is a min-heap)
                        # We remove closest because we want to keep track of the worst distance
                        heap.pop()
            
            # No need to process all trees if we have enough good candidates
            if len(seen) >= n_neighbors * 10:  # Arbitrary threshold, can be tuned
                break
    
        # Return the n_neighbors closest points
        # We sort again because heap order alone doesn't guarantee final order
        # Convert heap to list of (index, distance) pairs and sort
        results = [(idx, dist) for dist, idx in heap]
        return sorted(results, key=lambda x: x[1])[:n_neighbors]
    
    def _priority_queue_heapq_search(self, query: np.ndarray, n_neighbors: int) -> List[Tuple[int, float]]:
        """
        Perform nearest neighbor search using Python's heapq with negation.
        
        This method implements the same priority queue approach but uses heapq directly
        with negative distances to create a max-heap behavior. This provides a simpler
        implementation compared to the custom MaxHeap class.
        
        Args:
            query: The query point we're searching for neighbors of
            n_neighbors: Number of nearest neighbors to return
            
        Returns:
            List of (index, distance) tuples for the nearest neighbors,
            sorted by distance (closest first)
        """
        heap = []  # Will store (-distance, index) tuples
        seen: Set[int] = set()
        
        # Search through each tree
        for tree in self.trees:
            candidates = self._search_tree(tree, query, n_neighbors)
            
            for idx, dist in candidates:
                if idx not in seen:
                    if len(heap) >= n_neighbors * 2:
                        # Note: heap[0] gives us the most negative distance
                        worst_dist = -heap[0][0]
                        if dist >= worst_dist:
                            continue
                    
                    # Store negative distance for max-heap behavior
                    heapq.heappush(heap, (-dist, idx))
                    seen.add(idx)
                    
                    if len(heap) > n_neighbors * 2:
                        heapq.heappop(heap)
            
            if len(seen) >= n_neighbors * 10:
                break
        
        # Convert back to positive distances and sort
        results = [(-dist, idx) for dist, idx in heap]
        return [(idx, dist) for dist, idx in sorted(results)][:n_neighbors]

    def search(self, query: np.ndarray, n_neighbors: int = 1, 
               strategy: SearchStrategy = SearchStrategy.PRIORITY_QUEUE) -> List[Tuple[int, float]]:
        """
        Search for nearest neighbors using specified strategy.
        
        Args:
            query: Query point
            n_neighbors: Number of neighbors to return
            strategy: Search strategy (SCORE_BASED or PRIORITY_QUEUE)
            
        Returns:
            List of (index, distance) tuples for nearest neighbors
        """
        if self.data is None:
            raise ValueError("Index not built yet!")
            
        if strategy == SearchStrategy.SCORE_BASED:
            return self._score_based_search(query, n_neighbors)
        elif strategy == SearchStrategy.PRIORITY_QUEUE:
            return self._priority_queue_search(query, n_neighbors)
        else:
            return self._priority_queue_heapq_search(query, n_neighbors)