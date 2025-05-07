#include <bits/stdc++.h>
#include <omp.h>
#include <atomic>
using namespace std;

// BFS from source node 0 with OpenMP parallelization
vector<int> bfs(vector<vector<int>>& adj) {
    int V = adj.size();
    int s = 0; // source node

    vector<int> res;              // Stores traversal order
    queue<int> q;                 // BFS queue
    vector<atomic<bool>> visited(V); // Atomic visited array

    for (int i = 0; i < V; i++) visited[i] = false;

    visited[s] = true;
    q.push(s);

    vector<int> next_level;

    while (!q.empty()) {
        int level_size = q.size();
        next_level.clear();

        // Copy current level nodes to a vector to parallelize
        vector<int> curr_level(level_size);
        for (int i = 0; i < level_size; ++i) {
            curr_level[i] = q.front();
            q.pop();
        }

        // Collect results for this level (thread-safe)
        vector<int> level_result;

        #pragma omp parallel
        {
            vector<int> private_next;
            vector<int> private_result;

            #pragma omp for nowait
            for (int i = 0; i < curr_level.size(); ++i) {
                int curr = curr_level[i];
                private_result.push_back(curr);

                for (int j = 0; j < adj[curr].size(); ++j) {
                    int x = adj[curr][j];

                    // Atomically check and mark visited
                    if (!visited[x].exchange(true)) {
                        private_next.push_back(x);
                    }
                }
            }

            // Merge private results
            #pragma omp critical
            {
                next_level.insert(next_level.end(), private_next.begin(), private_next.end());
                level_result.insert(level_result.end(), private_result.begin(), private_result.end());
            }
        }

        // Add current level result to overall result
        res.insert(res.end(), level_result.begin(), level_result.end());

        // Enqueue next level nodes
        for (int node : next_level) {
            q.push(node);
        }
    }

    return res;
}

int main() {
    // Set number of threads
    omp_set_num_threads(4);

    // Example graph (adjacency list)
    vector<vector<int>> adj = {
        {1, 2},
        {0, 2, 3},
        {0, 4},
        {1, 4},
        {2, 3}
    };

    cout << "Parallel BFS: ";
    vector<int> ans = bfs(adj);
    for (int i : ans) {
        cout << i << " ";
    }
    cout << endl;

    return 0;
}

