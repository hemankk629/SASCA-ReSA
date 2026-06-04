#ifndef GRAPH_H
#define GRAPH_H

#include "pcg_random.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class Graph {
public:
  /*
  Input: std::string edgelist, std::string nodelist, bool start_from_checkpoint,
  std::string num_authors_bag, int author_max_lifetime Output: Graph object
  Description: Initializes the in-memory graph representation by parsing
  edgelist and nodelist files, setting up historical authorship data and
  structural topology.
  */
  Graph(std::string edgelist, std::string nodelist, bool start_from_checkpoint,
        std::string num_authors_bag, int author_max_lifetime);
  /*
  Input: std::pair<int, int> edge
  Output: void
  Description: Adds a directed edge between two nodes, updating the forward and
  backward adjacency maps as well as in/out degrees.
  */
  void AddEdge(std::pair<int, int> edge);

  /*
  Input: std::string filepath
  Output: char
  Description: Automatically sniffs the column delimiter (comma, tab, space) of
  a given CSV/TSV file by reading its first line.
  */
  static inline char GetDelimiter(std::string filepath) {
    std::ifstream edgelist(filepath);
    std::string line;
    getline(edgelist, line);
    if (line.find(',') != std::string::npos) {
      return ',';
    } else if (line.find('\t') != std::string::npos) {
      return '\t';
    } else if (line.find(' ') != std::string::npos) {
      return ' ';
    }
    throw std::invalid_argument("Could not detect filetype for " + filepath);
  }

  static inline std::
      unordered_map<std::string, int>
      /*
      Input: char delimiter, std::string filepath
      Output: std::unordered_map<std::string, int>
      Description: Parses the header row of a dataset and maps column string
      names to their respective integer indices for robust parsing.
      */
      GetHeaderToIndexMap(char delimiter, std::string filepath) {
    std::unordered_map<std::string, int> header_to_index_map;
    std::ifstream input_nodelist(filepath);
    std::string line;
    std::getline(input_nodelist, line);
    std::stringstream ss(line);
    std::string current_value;
    int index = 0;
    while (std::getline(ss, current_value, delimiter)) {
      header_to_index_map[current_value] = index;
      index++;
    }
    return header_to_index_map;
  }

  const std::set<int> &GetNodeSet() const;
  const std::unordered_map<int, std::vector<int>> &GetForwardAdjMap() const;
  const std::unordered_map<int, std::vector<int>> &GetBackwardAdjMap() const;
  /*
  Input: std::string attribute_key, int node, int attribute_value
  Output: void
  Description: Stores an integer-based metadata attribute on a specific node
  within the centralized NodeAttributes struct.
  */
  void SetIntAttribute(std::string attribute_key, int node,
                       int attribute_value);
  /*
  Input: None
  Output: void
  Description: Caches the initial historical reputation score of all existing
  authors at the beginning of the simulation.
  */
  void SaveInitialAuthorReputations();
  /*
  Input: std::string attribute_key, int node
  Output: int
  Description: Retrieves a previously stored integer metadata attribute for a
  specific node.
  */
  int GetIntAttribute(std::string attribute_key, int node) const;
  /*
  Input: std::string attribute_key, int node, std::string attribute_value
  Output: void
  Description: Stores a string-based metadata attribute on a specific node.
  */
  void SetStringAttribute(std::string attribute_key, int node,
                          std::string attribute_value);
  /*
  Input: std::string attribute_key, int node
  Output: std::string
  Description: Retrieves a previously stored string metadata attribute for a
  specific node.
  */
  std::string GetStringAttribute(std::string attribute_key, int node) const;
  /*
  Input: std::string attribute_key, int node, double attribute_value
  Output: void
  Description: Stores a floating-point metadata attribute on a specific node.
  */
  void SetDoubleAttribute(std::string attribute_key, int node,
                          double attribute_value);
  /*
  Input: std::string attribute_key, int node
  Output: double
  Description: Retrieves a previously stored floating-point metadata attribute
  for a specific node.
  */
  double GetDoubleAttribute(std::string attribute_key, int node) const;
  /*
  Input: std::string attribute_key, int node
  Output: bool
  Description: Checks if a specific integer attribute has been assigned a valid
  (non-default) value for a given node.
  */
  bool HasIntAttribute(std::string attribute_key, int node) const;
  /*
  Input: None
  Output: void
  Description: Consumes the empirical nodelist file, seeding the graph with its
  base nodes and their initial attributes (year, author, etc.).
  */
  void ParseNodelist();
  /*
  Input: None
  Output: void
  Description: Consumes the empirical edgelist file to construct the initial
  historical citation topology.
  */
  void ParseEdgelist();
  /*
  Input: int node
  Output: int
  Description: Returns the current in-degree (number of citations received) for
  a specific node.
  */
  int GetInDegree(int node) const;
  /*
  Input: int node
  Output: int
  Description: Returns the current out-degree (number of citations made) for a
  specific node.
  */
  int GetOutDegree(int node) const;
  /*
  Input: None
  Output: void
  Description: Aggregates the in-degrees of all papers published by each author
  to compute a real-time reputation score.
  */
  void ComputeAuthorReputations();
  /*
  Input: int node
  Output: int
  Description: Looks up the author(s) of a given node and returns their current
  aggregated reputation score.
  */
  int GetAuthorReputationForNode(int node) const;
  /*
  Input: int u
  Output: void
  Description: Injects a new empty node into the graph, initializing its
  attribute struct.
  */
  void AddNode(int u);
  /*
  Input: None
  Output: void
  Description: Dumps the entire structural topology of the graph to standard
  output for debugging purposes.
  */
  void PrintGraph() const;
  /*
  Input: std::string output_file
  Output: void
  Description: Flushes the complete generated citation topology (edgelist) to a
  specified file on disk.
  */
  void WriteGraph(std::string output_file) const;
  /*
  Input: std::string auxiliary_information_file
  Output: void
  Description: Flushes all accumulated node metadata (attributes) into a CSV
  file for subsequent analysis.
  */
  void WriteAttributes(std::string auxiliary_information_file) const;
  /*
  Input: int current_year, const std::set<int> &exclusion_set
  Output: int
  Description: Generates or selects an author ID for a newly simulated paper,
  modeling birth/death rates and Lotka's law, avoiding any excluded IDs.
  */
  int GetNextAuthor(int current_year, const std::set<int> &exclusion_set);
  /*
  Input: None
  Output: int
  Description: Randomly samples from the historical empirical distribution to
  determine how many authors a new paper should have.
  */
  int GetNextNumAuthors();
  /*
  Input: None
  Output: void
  Description: Pre-loads the bag of empirical author counts from disk into
  memory for fast O(1) sampling.
  */
  void ReadNumAuthorsBag();
  /*
  Input: int author
  Output: int (0 or 1)
  Description: Evaluates whether a specific author ID is considered to have
  received funding based on graph parameters.
  */
  int IsAuthorFunded(int author) const;
  /*
  Input: int author, int node
  Output: void
  Description: Registers a new publication (node) against an author's historical
  record, updating internal maps.
  */
  void UpdateAuthorPublicationMap(int author, int node);
  /*
  Input: int author, int cartel_id
  Output: void
  Description: Marks an author as belonging to a specific cartel syndicate.
  */
  void SetCartelID(int author, int cartel_id);
  /*
  Input: int author
  Output: int
  Description: Retrieves the cartel identifier for an author, returning -1 if
  they are not in a cartel.
  */
  int GetCartelID(int author) const;
  /*
  Input: int cartel_id
  Output: std::set<int>
  Description: Returns all author IDs that are members of the given cartel.
  */
  std::set<int> GetCartelAuthors(int cartel_id) const;
  /*
  Input: int author_id
  Output: std::vector<int>
  Description: Retrieves the list of all node IDs (papers) authored by a
  specific individual.
  */
  std::vector<int> GetAuthorPublications(int author_id) const;
  /*
  Input: int author_id
  Output: void
  Description: Manually injects an author into the tracking maps, primarily used
  during cartel injection or specialized setups.
  */
  void UpdateAuthorManual(int author_id);
  /*
  Input: None
  Output: std::set<int>
  Description: Returns the IDs of all authors currently participating in any
  cartel.
  */
  std::set<int> GetCartelSet() const;

private:
  std::set<int> node_set;
  std::string edgelist;
  std::string nodelist;
  std::set<int> cartel_set;
  bool start_from_checkpoint;
  std::string num_authors_bag;
  int author_max_lifetime;
  int IsNextAuthorFunded();
  std::vector<int> num_authors_bag_vec;

protected:
  std::unordered_map<int, std::vector<int>> publication_count_to_author_map;
  int next_author_id = 0;
  int lotka_exponent = 2;
  std::unordered_map<int, std::vector<int>> forward_adj_map;
  std::unordered_map<int, std::vector<int>> backward_adj_map;
  std::unordered_map<int, int> author_birth_year_map;
  std::unordered_map<int, std::vector<int>> author_publication_map;
  std::unordered_map<int, int> author_reputation_map;
  struct NodeAttributes {
    int year = -1;
    double alpha = -1.0;
    double pa_weight = -1.0;
    double fit_weight = -1.0;
    double num_authors_weight = -1.0;
    double author_reputation_weight = -1.0;
    int fitness_lag_duration = -1;
    int fitness_peak_value = -1;
    int fitness_peak_duration = -1;
    int assigned_out_degree = -1;
    int planted_nodes_line_number = -1;
    int sampled_neighborhood_size = -1;
    std::string generator_node_string = "no_generators";
    int fully_random_citations = -1;
    int author_id = -1;
    int initial_author_reputation = -1;
    int num_authors = -1;
    int cartel_id = -1;
    std::string type = "";
    int in_degree = 0;
    int out_degree = 0;
  };
  std::unordered_map<int, NodeAttributes> node_attributes;
  std::unordered_map<int, int> author_cartel_map;
  std::unordered_map<int, std::set<int>> cartel_author_map;
};

#endif
