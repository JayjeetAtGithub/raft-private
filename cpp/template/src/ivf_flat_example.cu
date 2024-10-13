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
#include <raft/neighbors/cagra.cuh>
#include <raft/neighbors/brute_force.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/stats/neighborhood_recall.cuh>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>

#include <cstdint>
#include <optional>
#include <chrono>

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
  auto neighbors    = raft::make_device_matrix<int64_t>(res, n_queries, top_k);
  auto distances    = raft::make_device_matrix<float>(res, n_queries, top_k);

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
  auto bfknn_index = raft::neighbors::brute_force::build(res, dataset);
  raft::neighbors::brute_force::search(res,
                                     bfknn_index,
                                     queries,
                                     reference_neighbors.view(),
                                     reference_distances.view());
  float const recall_scalar = 0.0;
  auto recall_value = raft::make_host_scalar(recall_scalar);
  raft::stats::neighborhood_recall(res,
                                  raft::make_const_mdspan(neighbors.view()),
                                  raft::make_const_mdspan(reference_neighbors.view()),
                                  recall_value.view());
  res.sync_stream();
  std::cout << "Recall@" << top_k << ": " << recall_value(0) << std::endl;
}

int main(int argc, char **argv)
{
  if (argc != 5) {
    std::cout << argv[0] << "<n_learn> <n_probe> <algo> <mem_type>" << std::endl;
    exit(1);
  }

  // Get params from the user
  int64_t n_samples = std::atoi(argv[1]);
  int64_t n_probe = std::stoi(argv[2]);
  std::string algo = argv[3];
  std::string mem_type = argv[4];
  int64_t n_dim     = 96;
  int64_t n_queries = 10'000;
  int64_t n_list = int64_t(4 * std::sqrt(n_samples));
  int64_t top_k = 100;

  // Set the memory resources
  raft::device_resources res;
  if (mem_type == "managed") {
    rmm::mr::managed_memory_resource managed_mr;
    rmm::mr::set_current_device_resource(&managed_mr);
  } else if (mem_type == "cuda") {
    rmm::mr::cuda_memory_resource cuda_mr;
    rmm::mr::set_current_device_resource(&cuda_mr);
  } else if (mem_type == "pool") {
    rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
      rmm::mr::get_current_device_resource(), 2 * 1024 * 1024 * 1024ull);
    rmm::mr::set_current_device_resource(&pool_mr);
  } else {
    std::cout << "[INFO] Invalid memory type" << std::endl;
    exit(1);
  }

  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
  rmm::mr::get_current_device_resource(), 50 * 1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(&pool_mr);

  // Create input arrays.
  auto dataset      = raft::make_device_matrix<float, int64_t>(res, n_samples, n_dim);
  auto queries      = raft::make_device_matrix<float, int64_t>(res, n_queries, n_dim);
  generate_dataset(res, dataset.view(), queries.view());

  // Simple build and search example.
  if (algo == "ivf") {
    ivf_search(res,
              raft::make_const_mdspan(dataset.view()),
              raft::make_const_mdspan(queries.view()),
              n_list,
              n_probe,
              top_k);
  }

  res.sync_stream();
}
