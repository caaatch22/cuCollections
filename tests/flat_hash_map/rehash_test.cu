/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <test_utils.hpp>

#include <cuco/flat_hash_map.cuh>

#include <cuda/functional>
#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>

#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE_SIG("flat_hash_map rehash tests",
                       "",
                       ((typename Key, typename Value), Key, Value),
                       (int32_t, int32_t),
                       (int32_t, int64_t),
                       (int64_t, int32_t),
                       (int64_t, int64_t))
{
  constexpr std::size_t num_keys = 1'000;
  cuco::flat_hash_map<Key, Value> map{num_keys * 2,
                                      cuco::empty_key<Key>{-1},
                                      cuco::empty_value<Value>{-1},
                                      cuco::erased_key<Key>{-2}};

  SECTION("Check basic rehash functionality")
  {
    auto keys_begin  = thrust::counting_iterator<Key>(1);
    auto pairs_begin = thrust::make_transform_iterator(
      keys_begin, cuda::proclaim_return_type<cuco::pair<Key, Value>>([] __device__(Key const& x) {
        return cuco::pair<Key, Value>(x, static_cast<Value>(x));
      }));

    map.insert(pairs_begin, pairs_begin + num_keys);
    map.rehash();
    REQUIRE(map.size() == num_keys);

    map.rehash(num_keys * 2);
    REQUIRE(map.size() == num_keys);

    REQUIRE(map.capacity() == num_keys * 2);
  }

  SECTION("rehash and find")
  {
    thrust::device_vector<Key> d_keys(num_keys);
    thrust::device_vector<Value> d_values(num_keys);

    thrust::sequence(thrust::device, d_keys.begin(), d_keys.end(), 1);
    thrust::sequence(thrust::device, d_values.begin(), d_values.end(), 1);

    auto pairs_begin =
      thrust::make_zip_iterator(cuda::std::tuple{d_keys.begin(), d_values.begin()});
    map.insert(pairs_begin, pairs_begin + num_keys);
    REQUIRE(map.size() == num_keys);

    map.rehash(num_keys * 2);
    REQUIRE(map.size() == num_keys);

    thrust::device_vector<Value> d_found_values(num_keys);
    map.find(d_keys.begin(), d_keys.end(), d_found_values.begin());

    auto zip_equal = cuda::proclaim_return_type<bool>(
      [] __device__(auto const& p) { return cuda::std::get<0>(p) == cuda::std::get<1>(p); });
    auto zip =
      thrust::make_zip_iterator(cuda::std::tuple{d_values.begin(), d_found_values.begin()});
    REQUIRE(cuco::test::all_of(zip, zip + num_keys, zip_equal));
  }
}
