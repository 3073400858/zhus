import heapq
import networkx as nx
import matplotlib.pyplot as plt

# 1. 定义罗马尼亚各个城市相连的无向图及实际距离 (g(n))
graph = {
    'Arad': {'Zerind': 75, 'Timisoara': 118, 'Sibiu': 140},
    'Zerind': {'Arad': 75, 'Oradea': 71},
    'Oradea': {'Zerind': 71, 'Sibiu': 151},
    'Timisoara': {'Arad': 118, 'Lugoj': 111},
    'Lugoj': {'Timisoara': 111, 'Mehadia': 70},
    'Mehadia': {'Lugoj': 70, 'Drobeta': 75},
    'Drobeta': {'Mehadia': 75, 'Craiova': 120},
    'Craiova': {'Drobeta': 120, 'Rimnicu Vilcea': 146, 'Pitesti': 138},
    'Sibiu': {'Arad': 140, 'Oradea': 151, 'Fagaras': 99, 'Rimnicu Vilcea': 80},
    'Rimnicu Vilcea': {'Sibiu': 80, 'Craiova': 146, 'Pitesti': 97},
    'Fagaras': {'Sibiu': 99, 'Bucharest': 211},
    'Pitesti': {'Rimnicu Vilcea': 97, 'Craiova': 138, 'Bucharest': 101},
    'Bucharest': {'Fagaras': 211, 'Pitesti': 101, 'Giurgiu': 90, 'Urziceni': 85},
    'Giurgiu': {'Bucharest': 90},
    'Urziceni': {'Bucharest': 85, 'Hirsova': 98, 'Vaslui': 142},
    'Hirsova': {'Urziceni': 98, 'Eforie': 86},
    'Eforie': {'Hirsova': 86},
    'Vaslui': {'Urziceni': 142, 'Iasi': 92},
    'Iasi': {'Vaslui': 92, 'Neamt': 87},
    'Neamt': {'Iasi': 87}
}

# 2. 定义各个城市到终点 Bucharest 的直线距离启发式函数 (h(n))
heuristic = {
    'Arad': 366, 'Bucharest': 0, 'Craiova': 160, 'Drobeta': 242,
    'Eforie': 161, 'Fagaras': 178, 'Giurgiu': 77, 'Hirsova': 151,
    'Iasi': 226, 'Lugoj': 244, 'Mehadia': 241, 'Neamt': 234,
    'Oradea': 380, 'Pitesti': 98, 'Rimnicu Vilcea': 193, 'Sibiu': 253,
    'Timisoara': 329, 'Urziceni': 80, 'Vaslui': 199, 'Zerind': 374
}

# 3. A* 算法实现
def a_star_search(graph, start, goal):
    # 优先队列 open_list 存储: (f_score, g_score, current_node, path)
    open_list = []
    heapq.heappush(open_list, (heuristic[start], 0, start, [start]))
    
    # 记录已经访问过的节点及其最优 g_score
    visited = {}

    while open_list:
        f, g, current, path = heapq.heappop(open_list)

        # 如果到达终点，返回路径和总代价
        if current == goal:
            return path, g

        # 节点扩展
        if current in visited and visited[current] <= g:
            continue
        visited[current] = g

        for neighbor, cost in graph[current].items():
            new_g = g + cost
            new_f = new_g + heuristic[neighbor]
            
            if neighbor not in visited or new_g < visited.get(neighbor, float('inf')):
                heapq.heappush(open_list, (new_f, new_g, neighbor, path + [neighbor]))

    return None, float('inf')

# ================= 运行算法 =================
start_node = 'Arad'
goal_node = 'Bucharest'
shortest_path, total_cost = a_star_search(graph, start_node, goal_node)

print(f"起点: {start_node} -> 终点: {goal_node}")
print(f"最短路径: {' -> '.join(shortest_path)}")
print(f"总代价(最短距离): {total_cost}")

# ================= 绘制可视化图 =================
def draw_graph(graph, path):
    G = nx.Graph()
    # 添加边和权重
    for city, neighbors in graph.items():
        for neighbor, weight in neighbors.items():
            G.add_edge(city, neighbor, weight=weight)
            
    # 近似罗马尼亚地图的节点排布坐标
    pos = {
        'Arad': (0.5, 4.5), 'Zerind': (0.8, 5.5), 'Oradea': (1.2, 6.2), 'Timisoara': (0.6, 3.2),
        'Lugoj': (1.6, 2.5), 'Mehadia': (1.8, 1.5), 'Drobeta': (1.6, 0.5), 'Craiova': (3.2, 0.2),
        'Sibiu': (2.4, 4.0), 'Rimnicu Vilcea': (2.8, 2.8), 'Fagaras': (3.8, 4.2), 'Pitesti': (4.0, 2.0),
        'Bucharest': (5.2, 1.5), 'Giurgiu': (4.8, 0.2), 'Urziceni': (6.0, 2.2), 'Hirsova': (7.0, 2.2),
        'Eforie': (7.2, 1.2), 'Vaslui': (6.4, 3.8), 'Iasi': (5.8, 5.0), 'Neamt': (4.8, 5.5)
    }

    plt.figure(figsize=(12, 8))
    
    # 绘制基础图
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold', edge_color='gray')
    
    # 绘制边上的权重
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
    
    # 高亮最短路径
    path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=3)
    nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='lightgreen', node_size=2500)

    plt.title(f"A* Algorithm Path from {path[0]} to {path[-1]} (Total Cost: {total_cost})", fontsize=15)
    plt.show()

# 调用绘图函数
draw_graph(graph, shortest_path)