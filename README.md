# annopy
Python implementation of Annoy 

# Usage

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from enum import Enum
import time
from annopy import Annoy, SearchStrategy

# Create sample data
dimension = 10
n_points = 1000
data = np.random.randn(n_points, dimension)

# Initialize and build index
annoy = Annoy(n_trees=50, max_size=10)
annoy.build(data)

# Create query point
query = np.random.randn(dimension)
print(f"Query: {query}")
n_neighbors = 10

# Search using priority queue strategy
priority_results = annoy.search(
    query, 
    n_neighbors=n_neighbors, 
    strategy=SearchStrategy.PRIORITY_QUEUE
)

# Search using priority queue heapq strategy
priority_heapq_results = annoy.search(
    query, 
    n_neighbors=n_neighbors, 
    strategy=SearchStrategy.PRIORITY_QUEUE_HEAPQ
)

# Search using score-based strategy
score_results = annoy.search(
    query, 
    n_neighbors=n_neighbors, 
    strategy=SearchStrategy.SCORE_BASED
)

# Print results
print("Priority Queue Results:")
for idx, dist in priority_results:
    print(f"Index: {idx}, Distance: {dist:.3f}")

print("\nPriority Queue Results:")
for idx, dist in priority_heapq_results:
    print(f"Index: {idx}, Distance: {dist:.3f}")

print("\nScore-based Results:")
for idx, dist in score_results:
    print(f"Index: {idx}, Distance: {dist:.3f}")

```