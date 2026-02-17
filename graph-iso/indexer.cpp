#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <openssl/sha.h>
#include <bliss/graph.hh>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace std;

int main(int argc, char* argv[]) {

    if (argc < 2) {
        cerr << "Usage: ./indexer graph.json\n";
        return 1;
    }

    ifstream f(argv[1]);
    if (!f) {
        cerr << "Cannot open file\n";
        return 1;
    }

    json j;
    f >> j;

    int n = j.size();
    if (n <= 0 || n > 1000) {
        cerr << "Invalid node count\n";
        return 1;
    }

    vector<vector<int>> adj(n);

    for (int i = 0; i < n; i++) {
        for (auto& v : j[i]) {
            int nb = v.get<int>();
            if (nb < 0 || nb >= n || nb == i) {
                cerr << "Invalid edge\n";
                return 1;
            }
            adj[i].push_back(nb);
        }
        sort(adj[i].begin(), adj[i].end());
        adj[i].erase(unique(adj[i].begin(), adj[i].end()), adj[i].end());
    }

    // symmetry check
    for (int i = 0; i < n; i++)
        for (int j2 : adj[i])
            if (!binary_search(adj[j2].begin(), adj[j2].end(), i)) {
                cerr << "Graph not symmetric\n";
                return 1;
            }

    bliss::Graph g(n);

    for (int i = 0; i < n; i++)
        g.change_color(i, 0);

    for (int i = 0; i < n; i++)
        for (int j2 : adj[i])
            if (i < j2)
                g.add_edge(i, j2);

    bliss::Stats stats;
    const unsigned int* perm =
        g.canonical_form(stats, nullptr, nullptr);

    if (!perm) {
        cerr << "Canonicalization failed\n";
        return 1;
    }

    // Build adjacency matrix
    vector<vector<bool>> matrix(n, vector<bool>(n, false));
    for (int i = 0; i < n; i++)
        for (int j2 : adj[i])
            matrix[i][j2] = true;

    // Bit-pack canonical upper triangle
    vector<uint8_t> packed;
    packed.reserve((n * (n - 1)) / 16 + 1);

    uint8_t current_byte = 0;
    int bit_pos = 0;

    for (int i = 0; i < n; i++) {
        for (int j2 = i + 1; j2 < n; j2++) {

            bool edge =
                matrix[perm[i]][perm[j2]];

            if (edge)
                current_byte |= (1 << bit_pos);

            bit_pos++;

            if (bit_pos == 8) {
                packed.push_back(current_byte);
                current_byte = 0;
                bit_pos = 0;
            }
        }
    }

    if (bit_pos != 0)
        packed.push_back(current_byte);

    // SHA256
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(packed.data(), packed.size(), hash);

    // Print hash
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++)
        cout << hex << setw(2) << setfill('0')
             << (int)hash[i];

    cout << endl;

    // Print canonical permutation
    cout << "PERM:";
    for (int i = 0; i < n; i++)
        cout << " " << perm[i];

    cout << endl;

    return 0;
}
