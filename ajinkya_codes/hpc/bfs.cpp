#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
#include <mutex>
using namespace std;

class Graph {
public:
    int V;
    vector<vector<int>> adj;

    Graph(int V) {
        this->V = V;
        adj.resize(V);
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u); // Assuming undirected
    }

    void parallelBFS(int start) {
        vector<bool> visited(V, false);
        queue<int> q;
        mutex q_mutex, visited_mutex;
        
        visited[start] = true;
        q.push(start);

        while (!q.empty()) {
            int size = q.size();
            vector<int> current_level;

            // Copy the current level to a temporary buffer
            for (int i = 0; i < size; ++i) {
                int node = q.front(); q.pop();
                current_level.push_back(node);
                cout << node << " ";
            }

            // Parallel processing of neighbors
            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < current_level.size(); ++i) {
                int node = current_level[i];
                for (int neighbor : adj[node]) {
                    // Protect access to visited and queue
                    bool do_enqueue = false;
                    visited_mutex.lock();
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        do_enqueue = true;
                    }
                    visited_mutex.unlock();

                    if (do_enqueue) {
                        q_mutex.lock();
                        q.push(neighbor);
                        q_mutex.unlock();
                    }
                }
            }
        }
        cout << endl;
    }
};

int main() {
    Graph g(6);
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 3);
    g.addEdge(1, 4);
    g.addEdge(2, 5);

    cout << "Parallel BFS starting from node 0:" << endl;
    g.parallelBFS(0);

    return 0;
}
