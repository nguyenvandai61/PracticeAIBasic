import networkx as nx
import matplotlib.pyplot as plt
# Bai tap 2
def Depth_First_Search(initialState, goalTest):
    frontier = []
    frontier.append(initialState)
    explored = []
    while len(frontier) > 0:
        # print("frontier: ", frontier)
        state = frontier.pop(len(frontier)-1)  # pop(0) [1,2,3,4,5]
        explored.append(state)
        print("Đỉnh khám phá: "+state)
        if goalTest == state:
            return True
        for neighbor in G.neighbors(state):
            if neighbor not in list(set(frontier + explored)):
                frontier.append(neighbor)
    return False
if __name__ == "__main__":
    G = nx.Graph()
    G.add_nodes_from(["S", "A", "B", "C", "D", "E", "F", "G", "H"])
    G.add_weighted_edges_from(
        [
            ("S", "A", 3),
            ("S", "B", 6),
            ("S", "C", 2),
            ("A", "D", 3),
            ("B", "D", 4),
            ("B", "G", 9),
            ("B", "E", 2),
            ("C", "E", 1),
            ("D", "F", 5),
            ("E", "F", 6),
            ("E", "H", 5),
            ("F", "G", 5),
            ("H", "G", 1),
            ("F", "G", 4),
        ]
    )
    result = Depth_First_Search("S", "G")
    print(result)