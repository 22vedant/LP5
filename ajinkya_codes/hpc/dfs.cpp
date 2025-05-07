#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

class Graph {
public:
    int V;
    vector<vector<int>> adj;
    vector<bool> visited;

    Graph(int V) {
        this->V = V;
        adj.resize(V);
        visited.resize(V, false);
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u); // Assuming an undirected graph
    }

    void parallelDFS(int node) {
        visited[node] = true;
        cout << node << " ";
        
        #pragma omp parallel for
        for (size_t i = 0; i < adj[node].size(); i++) {
            int neighbor = adj[node][i];
            if (!visited[neighbor]) {
                parallelDFS(neighbor);
            }
        }
    }
};

int main() {
    Graph g(6);
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 3);
    g.addEdge(1, 4);
    g.addEdge(2, 5);
    
    cout << "Parallel DFS starting from node 0:" << endl;
    g.parallelDFS(0);
    cout << endl;
    return 0;
}

