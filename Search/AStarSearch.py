from netrc import netrc
from graph2 import Node
from graph2 import Graph
import heapq
import pandas as pd


def update_cost(g, current_node, prev_node):
    if g.get_edge(prev_node, current_node) is not None:
        if current_node.cost > prev_node.cost + g.get_edge(prev_node, current_node)[2]:
            current_node.cost = prev_node.cost + g.get_edge(prev_node, current_node)[2]
            current_node.path = prev_node.path + [current_node.get_label()]


def find_by_label(array_of_node, node):
    for idx, n in enumerate(array_of_node):
        if n == node:
            return idx
        return -1


def update_frontier(frontier, new_node):
    index = find_by_label(frontier, new_node)
    if index >= 0:
        if frontier[index] > new_node:
            frontier[index] = new_node


def A_star_first_search(initial_state, goalTest):
    frontier = list()
    explored = list()
    # Sử dụng CTDL Heap
    heapq.heapify(frontier)
    heapq.heappush(frontier, initial_state)
    while len(frontier) > 0:
        print(list(map(lambda x: (x.get_label() + " (" + str(x.f) + ")"), frontier)))
        # remove smallest element in frontier and return smallest element
        state = heapq.heappop(frontier)
        explored.append(state)
        if state == goalTest:
            return True
        for neighbor in state.neighbors():
            update_cost(graph, neighbor, state)
            if neighbor.get_label() not in list(set([node.get_label() for node in frontier + explored])):
                neighbor.f = neighbor.cost + neighbor.goal_cost
                heapq.heappush(frontier, neighbor)
            elif find_by_label(frontier, neighbor) >= 0:
                update_frontier(frontier, neighbor)
                neighbor.f = neighbor.cost + neighbor.goal_cost
    return False


if __name__ == "__main__":
    graph = Graph()
    graph.add_node_from(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O"])
    graph.add_edges_from(
        [
            ("A", "B", 2),
            ("A", "C", 1),
            ("A", "D", 3),
            ("B", "E", 5),
            ("B", "F", 4),
            ("C", "G", 6),
            ("C", "H", 3),
            ("D", "I", 2),
            ("D", "J", 4),
            ("F", "K", 2),
            ("F", "L", 1),
            ("F", "M", 4),
            ("H", "N", 2),
            ("H", "O", 4),
        ]
    )
    # initial setup
    graph.nodes[0].goal_cost = 6  # goal_cost của node A = 6
    graph.nodes[0].cost = 0  # Set cost node A = 0
    graph.nodes[0].f = graph.nodes[0].cost + graph.nodes[0].goal_cost  # f cua Node A
    graph.nodes[1].goal_cost = 3  # goal_cost của node B = 3
    graph.nodes[2].goal_cost = 4  # goal_cost của node C = 4
    graph.nodes[3].goal_cost = 5  # goal_cost của node D = 5
    graph.nodes[4].goal_cost = 3  # goal_cost của node E = 3
    graph.nodes[5].goal_cost = 1  # goal_cost của node F = 1
    graph.nodes[6].goal_cost = 6  # goal_cost của node G = 6
    graph.nodes[7].goal_cost = 2  # goal_cost của node H = 2
    graph.nodes[8].goal_cost = 5  # goal_cost của node I = 5
    graph.nodes[9].goal_cost = 4  # goal_cost của node J = 4
    graph.nodes[10].goal_cost = 2  # goal_cost của node K = 2
    graph.nodes[11].goal_cost = 0  # goal_cost của node L = 0
    graph.nodes[12].goal_cost = 4  # goal_cost của node M = 4
    graph.nodes[13].goal_cost = 0  # goal_cost của node N = 0
    graph.nodes[14].goal_cost = 4  # goal_cost của node O = 4
    graph.set_compare_mode(Node.A_STAR)
    A_star_first_search(graph.nodes[0], graph.nodes[11])
