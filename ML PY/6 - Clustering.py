"""
Manual Clustering of 6 Points into 2 Clusters (No sklearn, No math library)


Cluster these 6 points into 2 clusters:

Points:
A = (1,1)
B = (2,1)
C = (1,2)
D = (8,8)
E = (9,8)
F = (8,9)
"""

import math

points = {"A": (1, 1), "B": (2, 1), "C": (1, 2), "D": (8, 8), "E": (9, 8), "F": (8, 9)}

c1 = (1, 1)
c2 = (8, 8)

cluster1 = []
cluster2 = []


for name, p in points.items():
    d1 = math.dist(p, c1)
    d2 = math.dist(p, c2)

    if d1 < d2:
        cluster1.append(name)
    else:
        cluster2.append(name)


print("Cluster 1", cluster1)
print("Cluster 2", cluster2)
