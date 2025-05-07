#include<bits/stdc++.h>
#include<omp.h>
using namespace std;

// BFS from given source s with OpenMP parallelization
vector<int> bfs(vector<vector<int>>& adj) {
    int V = adj.size();
    
    int s = 0; // source node 
    // create an array to store the traversal
    vector<int> res;

    // Create a queue for BFS
    queue<int> q;  
    
    // Initially mark all the vertices as not visited
    vector<bool> visited(V, false);

    // Mark source node as visited and enqueue it
    visited[s] = true;
    q.push(s);

    // We'll use this to collect neighbors in parallel
    vector<int> next_level;
    
    // Iterate over the queue
    while (!q.empty()) {
        // Process current level
        int level_size = q.size();
        next_level.clear();
        
        // Reserve some space for the results from this level
        res.reserve(res.size() + level_size);
        
        // Process all nodes at current level
        for (int i = 0; i < level_size; i++) {
            int curr = q.front();
            q.pop();
            res.push_back(curr);
            
            // Add all neighbors to next_level in parallel
            #pragma omp parallel
            {
                vector<int> private_next;
                
                #pragma omp for nowait
                for (int j = 0; j < adj[curr].size(); j++) {
                    int x = adj[curr][j];
                    bool was_visited = false;
                    
                    // Use atomic operation to check and update visited status
                    #pragma omp atomic capture
                    {
                        was_visited = visited[x];
                        if (!was_visited)
                            visited[x] = true;
                    }
                    
                    if (!was_visited) {
                        private_next.push_back(x);
                    }
                }
                
                // Merge thread-local results
                #pragma omp critical
                {
                    next_level.insert(next_level.end(), private_next.begin(), private_next.end());
                }
            }
        }
        
        // Add all nodes from next_level to the queue
        for (int node : next_level) {
            q.push(node);
        }
    }

    return res;
}

int main() {
    // Set number of threads (adjust as needed)
    omp_set_num_threads(4);
    
    vector<vector<int>> adj = {{1,2}, {0,2,3}, {0,4}, {1,4}, {2,3}};
    
    cout << "Original BFS: ";
    vector<int> ans = bfs(adj);
    for(auto i : ans) {
        cout << i << " ";
    }
    cout << endl;
    
    return 0;
}