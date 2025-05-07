#include <iostream>
#include <vector>
#include <omp.h>
#include <atomic>
using namespace std;

class Graph {
    int V;
    vector<vector<int>> adj;

public:
    Graph(int V) {
        this->V = V;
        adj.resize(V);
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u); // Undirected graph
    }

    void parallelDFS(int start) {
        vector<atomic<bool>> visited(V);
        for (int i = 0; i < V; i++) visited[i] = false;

        cout << "DFS starting from node " << start << ": ";

        #pragma omp parallel
        {
            #pragma omp single nowait
            {
                dfsUtil(start, visited);
            }
        }

        cout << endl;
    }

private:
    void dfsUtil(int u, vector<atomic<bool>>& visited) {
        if (visited[u].exchange(true)) return; // Already visited

        #pragma omp critical
        cout << u << " ";

        for (int v : adj[u]) {
            if (!visited[v]) {
                #pragma omp task
                dfsUtil(v, visited);
            }
        }

        #pragma omp taskwait
    }
};

int main() {
    Graph g(6);
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 3);
    g.addEdge(1, 4);
    g.addEdge(2, 4);
    g.addEdge(3, 5);
    g.addEdge(4, 5);

    g.parallelDFS(0);

    return 0;
}
