#include <iostream>
#include <vector>
#include <omp.h>
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
        vector<bool> visited(V, false);

        cout << "DFS starting from node " << start << ": ";

        #pragma omp parallel
        {
            #pragma omp single
            {
                dfsUtil(start, visited);
            }
        }

        cout << endl;
    }

private:
    void dfsUtil(int u, vector<bool>& visited) {
        bool process = false;

        // Only one thread should check and mark visited[u]
        #pragma omp critical
        {
            if (!visited[u]) {
                visited[u] = true;
                process = true;
                cout << u << " ";
            }
        }

        if (!process) return;

        for (int v : adj[u]) {
            // Always create a task; inside the task, we'll check visited status
            #pragma omp task
            {
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

    omp_set_num_threads(4); // Set based on available hardware

    g.parallelDFS(0);

    return 0;
}

