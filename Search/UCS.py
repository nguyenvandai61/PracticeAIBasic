from graph2 import Node, Graph
import heapq
def update_cost(g, current_node, prev_node):
    if g.get_edge(prev_node, current_node) is not None:
        if current_node.cost > prev_node.cost + graph.get_edge(prev_node, current_node)[2]:
            current_node.cost = prev_node.cost + graph.get_edge(prev_node, current_node)[2]
            current_node.path = prev_node.path + [current_node.get_label()]
def find_by_label(array_of_node, node):
    for idx, n in enumerate(array_of_node):
        if n == node:
            return idx
    return -1
def update_frontier(frontier, new_node):
    # Update trạng thái của frontier
    index = find_by_label(frontier, new_node)
    if index >= 0:
        if frontier[index] > new_node:
            frontier[index] = new_node
def uniform_cost_search(graph, initial_state, goalTest):
    frontier = list()
    explored = list()
    # Sử dụng CTDL Heap
    heapq.heapify(frontier)
    heapq.heappush(frontier, initial_state)
    while len(frontier) > 0:
        print(frontier)
        # remove smallest element in frontier and return smallest element
        state = heapq.heappop(frontier)
        explored.append(state)
        if goalTest == state:
            return True
        for neighbor in state.neighbors():
            update_cost(graph, current_node=neighbor, prev_node=state)
            if neighbor.get_label() not in list(set([e.get_label() for e in frontier + explored])):
                heapq.heappush(frontier, neighbor)
            elif find_by_label(frontier, neighbor) is not -1:
                update_frontier(frontier, neighbor)
    return False
if __name__ == "__main__":
    graph = Graph()
    graph.add_node("S")
    graph.add_node_from(["S", "A", "B", "C", "D", "E", "F", "G", "H"])
    graph.add_edges_from(
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
            ("E", "H", 5),
            ("F", "E", 6),
            ("H", "G", 8),
            ("F", "G", 5),
        ],
        is_duplicated=True  # Bỏ qua hướng của các đường đi
    )
    # initial setup
    graph.nodes[0].cost = 0  # cost của node S = 0
    graph.nodes[0].path = ['S']  # S -> S là S
    result = uniform_cost_search(graph, graph.nodes[0], graph.nodes[7])  # Tìm đường đi từ S -> G
    # In kết quả có tìm ra đường đi hay ko
    print(result)
    if result:
        # In ra đường đi từ S -> G
        print("Min path from S to G: ")
        print(graph.nodes[7].path)
