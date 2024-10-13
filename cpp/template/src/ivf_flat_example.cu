/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "common.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/neighbors/ivf_flat.cuh>
#include <raft/neighbors/brute_force.cuh>
#include <raft/neighbors/cagra.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/stats/neighborhood_recall.cuh>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>

#include <cstdint>
#include <optional>
#include <chrono>

float *file_read(const char *fname, int64_t *n, int64_t *d, int64_t limit) {
  FILE *f = fopen(fname, "r");
  if (!f) {
    fprintf(stderr, "Could not open %s\n", fname);
    perror("");
    abort();
  }
  int64_t e;
  int64_t N;
  e = fread(&N, sizeof(uint32_t), 1, f);
  *n = std::min(N, limit);
  int64_t D;
  e = fread(&D, sizeof(uint32_t), 1, f);
  *d = D;
  int64_t len = std::min(N, limit) * D;
  float *v = new float[len];
  e = fread(v, sizeof(float), len, f);
  std::cout << "Read " << e << " elements" << std::endl;
  fclose(f);
  return v;
}

void read_dataset(const char *filename, float *&xb, int64_t *d, int64_t *n,
                  int64_t limit) {
  xb = file_read(filename, d, n, limit);
}


void cagra_search(raft::device_resources const& res,
                  raft::device_matrix_view<const float, int64_t> dataset,
                  raft::device_matrix_view<const float, int64_t> queries,
                  int64_t top_k)
{
  using namespace raft::neighbors;
  std::cout << "Performing CAGRA search" << std::endl;

  // Build the CAGRA index
  cagra::index_params index_params;
  auto s = std::chrono::high_resolution_clock::now();
  auto index = cagra::build<float, int64_t>(res, index_params, dataset);
  auto e = std::chrono::high_resolution_clock::now();
  std::cout
    << "[TIME] Train and Index: "
    << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count()
    << " ms" << std::endl;
  std::cout << "[INFO] CAGRA index has " << index.size() << " vectors" << std::endl;
  std::cout << "[INFO] CAGRA graph has degree " << index.graph_degree() << ", graph size ["
            << index.graph().extent(0) << ", " << index.graph().extent(1) << "]" << std::endl;

  // Define arrays to hold search output results  
  int64_t n_queries = queries.extent(0);
  auto neighbors    = raft::make_device_matrix<int64_t, int64_t>(res, n_queries, top_k);
  auto distances    = raft::make_device_matrix<float, int64_t>(res, n_queries, top_k);

  // Perform the search operation
  cagra::search_params search_params;
  search_params.itopk_size = top_k;
  s = std::chrono::high_resolution_clock::now();
  cagra::search<float, int64_t>(
    res, search_params, index, queries, neighbors.view(), distances.view());
  e = std::chrono::high_resolution_clock::now();
  std::cout
      << "[TIME] Search: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count()
      << " ms" << std::endl;

  // Brute force search for reference
  auto reference_neighbors = raft::make_device_matrix<int64_t, int64_t>(res, n_queries, top_k);
  auto reference_distances = raft::make_device_matrix<float, int64_t>(res, n_queries, top_k); 
  auto brute_force_index = raft::neighbors::brute_force::build(res, dataset);
  raft::neighbors::brute_force::search(res,
                                     brute_force_index,
                                     queries,
                                     reference_neighbors.view(),
                                     reference_distances.view());
  float const recall_scalar = 0.0;
  auto recall_value = raft::make_host_scalar(recall_scalar);
  raft::stats::neighborhood_recall(res,
                                  raft::make_const_mdspan(neighbors.view()),
                                  raft::make_const_mdspan(reference_neighbors.view()),
                                  recall_value.view());
  std::cout << "[INFO] Recall@" << top_k << ": " << recall_value(0) << std::endl;
}

void ivf_search(raft::device_resources const& res,
                raft::device_matrix_view<const float, int64_t> dataset,
                raft::device_matrix_view<const float, int64_t> queries,
                int64_t n_list,
                int64_t n_probe,
                int64_t top_k)
{
  using namespace raft::neighbors;
  std::cout << "Performing IVF-FLAT search" << std::endl;

  // Build the IVF-FLAT index
  ivf_flat::index_params index_params;
  index_params.n_lists                  = n_list;
  index_params.kmeans_trainset_fraction = 0.1;
  index_params.kmeans_n_iters           = 100;
  index_params.metric                   = raft::distance::DistanceType::L2Expanded;
  auto s = std::chrono::high_resolution_clock::now();
  auto index = ivf_flat::build<float, int64_t>(res, index_params, dataset);
  auto e = std::chrono::high_resolution_clock::now();
  std::cout
    << "[TIME] Train and Index: "
    << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count()
    << " ms" << std::endl;
  std::cout << "[INFO] Number of clusters " << index.n_lists() << ", number of vectors added to index "
            << index.size() << std::endl;

  // Define arrays to hold search output results
  int64_t n_queries = queries.extent(0);
  auto neighbors    = raft::make_device_matrix<int64_t, int64_t>(res, n_queries, top_k);
  auto distances    = raft::make_device_matrix<float, int64_t>(res, n_queries, top_k);

  // Perform the search operation
  ivf_flat::search_params search_params;
  search_params.n_probes = n_probe;
  s = std::chrono::high_resolution_clock::now();
  ivf_flat::search<float, int64_t>(
    res, search_params, index, queries, neighbors.view(), distances.view());
  e = std::chrono::high_resolution_clock::now();
  std::cout
      << "[TIME] Search: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count()
      << " ms" << std::endl;


  // Brute force search for reference
  auto reference_neighbors = raft::make_device_matrix<int64_t, int64_t>(res, n_queries, top_k);
  auto reference_distances = raft::make_device_matrix<float, int64_t>(res, n_queries, top_k); 
  auto brute_force_index = raft::neighbors::brute_force::build(res, dataset);
  raft::neighbors::brute_force::search(res,
                                     brute_force_index,
                                     queries,
                                     reference_neighbors.view(),
                                     reference_distances.view());
  float const recall_scalar = 0.0;
  auto recall_value = raft::make_host_scalar(recall_scalar);
  raft::stats::neighborhood_recall(res,
                                  raft::make_const_mdspan(neighbors.view()),
                                  raft::make_const_mdspan(reference_neighbors.view()),
                                  recall_value.view());
  std::cout << "[INFO] Recall@" << top_k << ": " << recall_value(0) << std::endl;
}

int main(int argc, char **argv)
{
  if (argc != 5) {
    std::cout << argv[0] << " <learn_limit> <n_probe> <algo> <dataset_dir>" << std::endl;
    exit(1);
  }

  // Get params from the user
  int64_t learn_limit = std::stoi(argv[1]);
  int64_t n_probe = std::stoi(argv[2]);
  std::string algo = argv[3];
  std::string dataset_dir = argv[4];

  // Set the memory resources
  raft::device_resources res;
  rmm::mr::managed_memory_resource managed_mr;
  auto stats_mr =
    rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>(&managed_mr);
  rmm::mr::set_current_device_resource(&stats_mr);

  // Read the dataset files
  float *dataset, *queries;
  int64_t n_dataset, n_queries;
  int64_t d_dataset, d_queries;
  std::string dataset_path = dataset_dir + "/dataset.bin";
  std::string query_path = dataset_dir + "/query.bin";
  read_dataset(dataset_path.c_str(), dataset, &n_dataset, &d_dataset, learn_limit);
  read_dataset(query_path.c_str(), queries, &n_queries, &d_queries, 10'000);

  auto dataset_device = [&]() {
    auto dataset_host = raft::make_host_matrix<float, int64_t>(res, n_dataset, d_dataset);
    for (int64_t i = 0; i < n_dataset; i++) {
      for (int64_t j = 0; j < d_dataset; j++) {
        dataset_host(i, j) = dataset[i * d_dataset + j];
      }
    }
    auto dev_matrix = raft::make_device_matrix<float, int64_t>(res, n_dataset, d_dataset);
    raft::copy(res, dev_matrix.view(), dataset_host.view());
    return dev_matrix;
  }();

  auto queries_device = [&]() {
    auto queries_host = raft::make_host_matrix<float, int64_t>(res, n_queries, d_queries);
    for (int64_t i = 0; i < n_queries; i++) {
      for (int64_t j = 0; j < d_queries; j++) {
        queries_host(i, j) = queries[i * d_queries + j];
      }
    }
    auto dev_matrix = raft::make_device_matrix<float, int64_t>(res, n_queries, d_queries);
    raft::copy(res, dev_matrix.view(), queries_host.view());
    return dev_matrix;
  }();

  delete[] dataset;
  delete[] queries;

  std::cout << "Dataset: " << n_dataset << "x" << d_dataset << std::endl;
  std::cout << "Queries: " << n_queries << "x" << d_queries << std::endl;

  // Set the index and search params
  int64_t n_list = int64_t(4 * std::sqrt(n_dataset));
  int64_t top_k = 100;

  if (algo == "ivf") {
    ivf_search(res,
              raft::make_const_mdspan(dataset_device.view()),
              raft::make_const_mdspan(queries_device.view()),
              n_list,
              n_probe,
              top_k);
  }

  if (algo == "cagra") {
    cagra_search(res,
                raft::make_const_mdspan(dataset_device.view()),
                raft::make_const_mdspan(queries_device.view()),
                top_k);
  }

  res.sync_stream();
  std::cout << "[INFO] Peak memory usage: " << (stats_mr.get_bytes_counter().peak / 1048576.0) << " MB\n";
}
