import heapq
import sys

class Graph:

        def __init__(self):
                self.graph = {}
                self.edge_weights = {}

        def add_edge(self, u, v, cost):
                if u not in self.graph:
                        self.graph[u] = [v]
                else:
                        self.graph[u].append(v)

                if v not in self.graph:
                        self.graph[v] = [u]
                else:
                        self.graph[v].append(u)


                self.edge_weights[(u, v)] = cost
                self.edge_weights[(v, u)] = cost


        def Dijkstra(self, s):
                inSPT = {}
                key = {}
                parent = {}
                for node in self.graph:
                        inSPT[node] = False
                        key[node] = float('inf')
                        parent[node] = None

                key[s] = 0
                parent[s] = -1
                pq = []
                heapq.heapify(pq)
                heapq.heappush(pq, (key[s], s))

                while len(pq) > 0:
                        val, v = heapq.heappop(pq)
                        if inSPT[v] == True:
                                continue

                        inSPT[v] = True

                        for u in self.graph[v]:
                                if inSPT[u] == False and key[u] > key[v] + self.edge_weights[(v, u)]:
                                        key[u] = key[v] + self.edge_weights[(v, u)]
                                        heapq.heappush(pq, (key[u], u))
                                        parent[u] = v

                return parent
