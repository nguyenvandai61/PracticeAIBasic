import networkx as nx
import matplotlib.pyplot as plt
# kien truc graph
G = nx.Graph()
G.add_nodes_from(["S", "A", "B", "C", "D", "E", "F", "G", "H"])
G.add_weighted_edges_from(
    [
        ("S", "A", 3),
        ("S", "B", 6),
        ("S", "C", 2),
        ("A", "D", 3),
        ("B", "G", 9),
        ("B", "E", 2),
        ("C", "E", 1),
        ("D", "F", 5),
        ("E", "H", 5),
        ("F", "E", 6),
        ("H", "G", 8),
        ("F", "G", 5),
    ]
)
# print(G.nodes)  #['S', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
# for node in G.neighbors("A"):
#     print(node)

def Breadth_First_Search(initialState, goalTest):
    frontier = []
    frontier.append(initialState)
    explored = []
    while len(frontier) > 0:
        state = frontier.pop(0)
        explored.append(state)
        print(" frontier: ", frontier)
        if goalTest == state:
            return True
        for neighbor in G.neighbors(state):
            if neighbor not in list(set(frontier + explored)):
                frontier.append(neighbor)
    return False
if __name__ == "__main__":
    x = Breadth_First_Search("S", "G")
    print(x)
     #Draw graph
     #nx.draw(G, with_labels=True)
     #plt.draw()
     #plt.show()
