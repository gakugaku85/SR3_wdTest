import higra as hg
import numpy as np
from PIL import Image

image_array = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0.8, 0],
        [0, 0.4, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
)

# Convert the image to grayscale if needed
if len(image_array.shape) > 2:
    image_array = np.mean(image_array, axis=2).astype(np.uint8)

# Create the max tree
graph = hg.get_4_adjacency_graph(image_array.shape)
tree, altitudes = hg.component_tree_max_tree(graph, image_array)

# Access the properties of the max tree
num_nodes = tree.num_vertices()
num_edges = tree.num_edges()
root_node = tree.root()

# Print some information about the max tree
print("Number of nodes:", num_nodes)
print("Number of edges:", num_edges)
print("Root node:", root_node)
