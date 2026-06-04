#include "metrics_engine.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <random>
#include <span>

void MetricsEngine::FillInDegreeSpan(
    Graph *graph, const std::unordered_map<int, int> &continuous_node_mapping,
    std::span<int> in_degree_span) {
  for (auto const &node : graph->GetNodeSet()) {
    if (!continuous_node_mapping.contains(node)) {
      std::cerr << std::to_string(node) << " not in continuous node mapping "
                << std::endl;
    }
    int continuous_id = continuous_node_mapping.at(node);
    in_degree_span[continuous_id] = graph->GetInDegree(node);
  }
}

void MetricsEngine::FillAuthorReputationSpan(
    Graph *graph, const std::unordered_map<int, int> &continuous_node_mapping,
    std::span<int> author_reputation_span) {
  graph->ComputeAuthorReputations();
  for (auto const &node : graph->GetNodeSet()) {
    if (!continuous_node_mapping.contains(node)) {
      std::cerr << std::to_string(node) << " not in continuous node mapping "
                << std::endl;
    }
    int continuous_id = continuous_node_mapping.at(node);
    author_reputation_span[continuous_id] =
        graph->GetAuthorReputationForNode(node);
  }
}

void MetricsEngine::FillFitnessSpan(
    Graph *graph, const std::unordered_map<int, int> &continuous_node_mapping,
    int current_year, std::span<int> fitness_span, int fitness_decay_alpha) {
  for (auto const &node : graph->GetNodeSet()) {
    int fitness_peak_value = graph->GetIntAttribute("fitness_peak_value", node);
    int fitness_lag_duration =
        graph->GetIntAttribute("fitness_lag_duration", node);
    int fitness_peak_duration =
        graph->GetIntAttribute("fitness_peak_duration", node);
    int published_year = graph->GetIntAttribute("year", node);
    int continuous_index = continuous_node_mapping.at(node);
    if (published_year + fitness_lag_duration > current_year) {
      fitness_span[continuous_index] = 1;
    } else if (published_year + fitness_lag_duration + fitness_peak_duration >=
               current_year) {
      fitness_span[continuous_index] = fitness_peak_value;
    } else {
      double decayed_fitness_value =
          fitness_peak_value /
          pow(current_year - published_year - fitness_lag_duration -
                  fitness_peak_duration + 1,
              fitness_decay_alpha);
      fitness_span[continuous_index] = decayed_fitness_value;
    }
  }
}

void MetricsEngine::PopulateWeightSpans(
    std::span<double> pa_weight_span, std::span<double> fit_weight_span,
    std::span<double> num_authors_weight_span,
    std::span<double> author_reputation_weight_span, double preferential_weight,
    double fitness_weight, double num_authors_weight,
    double author_reputation_weight) {
  if (preferential_weight != -1 && fitness_weight != -1 &&
      num_authors_weight != -1 && author_reputation_weight != -1) {
#pragma omp parallel for
    for (int i = 0; i < pa_weight_span.size(); i++) {
      pcg32 &generator = Utils::GetThreadLocalPRNG();
      double pa_uniform = preferential_weight;
      double fit_uniform = fitness_weight;
      double num_authors_uniform = num_authors_weight;
      double author_reputation_uniform = author_reputation_weight;
      double sum = pa_uniform + fit_uniform + num_authors_uniform +
                   author_reputation_uniform;
      pa_weight_span[i] = (double)pa_uniform / sum;
      fit_weight_span[i] = (double)fit_uniform / sum;
      num_authors_weight_span[i] = (double)num_authors_uniform / sum;
      author_reputation_weight_span[i] =
          (double)author_reputation_uniform / sum;
    }
  } else {
#pragma omp parallel for
    for (int i = 0; i < pa_weight_span.size(); i++) {
      pcg32 &generator = Utils::GetThreadLocalPRNG();
      std::uniform_real_distribution<double> weights_uniform_distribution{0, 1};
      double pa_uniform = weights_uniform_distribution(generator);
      double fit_uniform = weights_uniform_distribution(generator);
      double num_authors_uniform = weights_uniform_distribution(generator);
      double author_reputation_uniform =
          weights_uniform_distribution(generator);
      double sum = pa_uniform + fit_uniform + num_authors_uniform +
                   author_reputation_uniform;
      pa_weight_span[i] = (double)pa_uniform / sum;
      fit_weight_span[i] = (double)fit_uniform / sum;
      num_authors_weight_span[i] = (double)num_authors_uniform / sum;
      author_reputation_weight_span[i] =
          (double)author_reputation_uniform / sum;
    }
  }
}

void MetricsEngine::PopulateNumAuthorsSpan(Graph *graph,
                                           std::span<int> num_authors_span) {
#pragma omp parallel for
  for (int i = 0; i < num_authors_span.size(); i++) {
    pcg32 &generator = Utils::GetThreadLocalPRNG();
    num_authors_span[i] = graph->GetNextNumAuthors();
  }
}

void MetricsEngine::PopulateFitnessSpans(
    std::span<int> fitness_lag_duration_span,
    std::span<int> fitness_peak_value_span,
    std::span<int> fitness_peak_duration_span, int lag_min, int lag_max,
    int peak_val_min, int peak_val_max, int peak_dur_min, int peak_dur_max,
    int fitness_alpha) {
  std::uniform_int_distribution<int> fitness_lag_duration_uniform_distribution(
      lag_min, lag_max);
  std::uniform_int_distribution<int> fitness_peak_duration_uniform_distribution(
      peak_dur_min, peak_dur_max);
  std::uniform_real_distribution<double> fitness_value_uniform_distribution(0,
                                                                            1);

#pragma omp parallel for
  for (int i = 0; i < fitness_lag_duration_span.size(); i++) {
    pcg32 &generator = Utils::GetThreadLocalPRNG();
    fitness_lag_duration_span[i] =
        fitness_lag_duration_uniform_distribution(generator);
    double fitness_uniform = fitness_value_uniform_distribution(generator);
    double adjusted_alpha = fitness_alpha + 1;
    double base_left = (pow(peak_val_max, adjusted_alpha) -
                        pow(peak_val_min, adjusted_alpha)) *
                       fitness_uniform;
    double base_right = pow(peak_val_min, adjusted_alpha);
    double exponent = 1.0 / adjusted_alpha;
    int fitness_power = pow(base_left + base_right, exponent);
    fitness_peak_value_span[i] = fitness_power;
    fitness_peak_duration_span[i] =
        fitness_peak_duration_uniform_distribution(generator);
  }
}

void MetricsEngine::PopulateAlphaSpan(std::span<double> alpha_span,
                                      bool use_alpha, double alpha,
                                      double minimum_alpha) {
  std::uniform_real_distribution<double> alpha_uniform_distribution(0, 1);

  if (!use_alpha) {
#pragma omp parallel for
    for (int i = 0; i < alpha_span.size(); i++) {
      pcg32 &generator = Utils::GetThreadLocalPRNG();
      alpha_span[i] = -1;
    }
  } else if (alpha == -1) {
#pragma omp parallel for
    for (int i = 0; i < alpha_span.size(); i++) {
      pcg32 &generator = Utils::GetThreadLocalPRNG();
      double alpha_uniform = alpha_uniform_distribution(generator);
      alpha_uniform = std::round(alpha_uniform * 1000.0) / 1000.0;
      alpha_span[i] = alpha_uniform;
    }
  } else if (minimum_alpha > 0) {
#pragma omp parallel for
    for (int i = 0; i < alpha_span.size(); i++) {
      pcg32 &generator = Utils::GetThreadLocalPRNG();
      std::uniform_real_distribution<double> minimum_alpha_uniform_distribution{
          minimum_alpha, 1};
      double alpha_uniform = minimum_alpha_uniform_distribution(generator);
      alpha_span[i] = alpha_uniform;
    }
  } else {
#pragma omp parallel for
    for (int i = 0; i < alpha_span.size(); i++) {
      alpha_span[i] = alpha;
    }
  }
}

void MetricsEngine::PopulateOutDegreeSpan(
    std::span<int> out_degree_span,
    const std::vector<int> &out_degree_bag_vec) {
  std::uniform_int_distribution<int> outdegree_index_uniform_distribution{
      0, (int)(out_degree_bag_vec.size() - 1)};
#pragma omp parallel for
  for (int i = 0; i < out_degree_span.size(); i++) {
    pcg32 &generator = Utils::GetThreadLocalPRNG();
    int index_uniform = outdegree_index_uniform_distribution(generator);
    out_degree_span[i] = out_degree_bag_vec[index_uniform];
  }
}

void MetricsEngine::CalculateTanhScores(
    std::unordered_map<int, double> &cached_results, std::span<int> src_span,
    std::span<double> dst_span, double peak_constant, double delay_constant) {
  double sum = 0;
#pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i < src_span.size(); i++) {
    double current_dst = -1;
    if (src_span[i] < 10000) {
      current_dst = cached_results[src_span[i]];
    } else {
      current_dst = peak_constant *
                    std::tanh((std::pow(src_span[i], 3) / delay_constant) *
                              (1 / peak_constant));
    }
    dst_span[i] = current_dst;
    sum += current_dst;
  }
#pragma omp parallel for
  for (int i = 0; i < src_span.size(); i++) {
    dst_span[i] /= sum;
  }
}

void MetricsEngine::CalculateExpScores(
    std::unordered_map<int, double> &cached_results, std::span<int> src_span,
    std::span<double> dst_span, double gamma) {
  double sum = 0;
#pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i < src_span.size(); i++) {
    double current_dst = std::max(std::pow(src_span[i], gamma), 1.0) + 1;
    dst_span[i] = current_dst;
    sum += current_dst;
  }
#pragma omp parallel for
  for (int i = 0; i < src_span.size(); i++) {
    dst_span[i] /= sum;
  }
}
