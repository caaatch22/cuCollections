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

TEMPLATE_TEST_CASE_SIG("flat_hash_map insert tests",
                       "",
                       ((typename Key, typename Value), Key, Value),
                       (int32_t, int32_t),
                       (int32_t, int64_t),
                       (int64_t, int32_t),
                       (int64_t, int64_t))
{
  constexpr std::size_t num_keys = 1'000'000;
  cuco::flat_hash_map<Key, Value> map{num_keys * 2,
                                      cuco::empty_key<Key>{-1},
                                      cuco::empty_value<Value>{-1},
                                      cuco::erased_key<Key>{-2}};

  SECTION("Check basic insert functionality")
  {
    thrust::device_vector<Key> d_keys(num_keys);
    thrust::device_vector<Value> d_values(num_keys);
    thrust::device_vector<bool> d_keys_exist(num_keys);

    thrust::sequence(thrust::device, d_keys.begin(), d_keys.end(), 1);
    thrust::sequence(thrust::device, d_values.begin(), d_values.end(), 1);

    auto pairs_begin =
      thrust::make_zip_iterator(cuda::std::tuple{d_keys.begin(), d_values.begin()});

    // Insert all key-value pairs
    map.insert(pairs_begin, pairs_begin + num_keys);

    REQUIRE(map.size() == num_keys);

    // Verify all keys exist after insertion
    map.contains(d_keys.begin(), d_keys.end(), d_keys_exist.begin());
    REQUIRE(cuco::test::all_of(d_keys_exist.begin(), d_keys_exist.end(), cuda::std::identity{}));

    // Verify values are correct by finding them
    thrust::device_vector<Value> d_found_values(num_keys);
    map.find(d_keys.begin(), d_keys.end(), d_found_values.begin());

    auto zip_equal = cuda::proclaim_return_type<bool>(
      [] __device__(auto const& p) { return cuda::std::get<0>(p) == cuda::std::get<1>(p); });
    auto zip =
      thrust::make_zip_iterator(cuda::std::tuple{d_values.begin(), d_found_values.begin()});
    REQUIRE(cuco::test::all_of(zip, zip + num_keys, zip_equal));
  }

  SECTION("Check insert with duplicate keys")
  {
    thrust::device_vector<Key> d_keys(num_keys);
    thrust::device_vector<Value> d_values(num_keys);
    thrust::device_vector<Value> d_duplicate_values(num_keys);

    thrust::sequence(thrust::device, d_keys.begin(), d_keys.end(), 1);
    thrust::sequence(thrust::device, d_values.begin(), d_values.end(), 1);
    thrust::sequence(thrust::device, d_duplicate_values.begin(), d_duplicate_values.end(), 1001);

    auto pairs_begin =
      thrust::make_zip_iterator(cuda::std::tuple{d_keys.begin(), d_values.begin()});
    auto duplicate_pairs_begin =
      thrust::make_zip_iterator(cuda::std::tuple{d_keys.begin(), d_duplicate_values.begin()});

    // Insert original key-value pairs
    map.insert(pairs_begin, pairs_begin + num_keys);
    REQUIRE(map.size() == num_keys);

    // Try to insert duplicate keys with different values
    map.insert(duplicate_pairs_begin, duplicate_pairs_begin + num_keys);

    // Size should remain the same (no new insertions)
    REQUIRE(map.size() == num_keys);

    // Verify original values are preserved
    thrust::device_vector<Value> d_found_values(num_keys);
    map.find(d_keys.begin(), d_keys.end(), d_found_values.begin());

    auto zip_equal = cuda::proclaim_return_type<bool>(
      [] __device__(auto const& p) { return cuda::std::get<0>(p) == cuda::std::get<1>(p); });
    auto zip =
      thrust::make_zip_iterator(cuda::std::tuple{d_values.begin(), d_found_values.begin()});
    REQUIRE(cuco::test::all_of(zip, zip + num_keys, zip_equal));
  }

  SECTION("Check insert after erase")
  {
    thrust::device_vector<Key> d_keys(num_keys);
    thrust::device_vector<Value> d_values(num_keys);
    thrust::device_vector<Value> d_new_values(num_keys);
    thrust::device_vector<bool> d_keys_exist(num_keys);

    thrust::sequence(thrust::device, d_keys.begin(), d_keys.end(), 1);
    thrust::sequence(thrust::device, d_values.begin(), d_values.end(), 1);
    thrust::sequence(thrust::device, d_new_values.begin(), d_new_values.end(), 2001);

    auto pairs_begin =
      thrust::make_zip_iterator(cuda::std::tuple{d_keys.begin(), d_values.begin()});
    auto new_pairs_begin =
      thrust::make_zip_iterator(cuda::std::tuple{d_keys.begin(), d_new_values.begin()});

    // Insert original key-value pairs
    map.insert(pairs_begin, pairs_begin + num_keys);
    REQUIRE(map.size() == num_keys);

    // Erase first half of keys
    map.erase(d_keys.begin(), d_keys.begin() + num_keys / 2);
    REQUIRE(map.size() == num_keys / 2);

    // Insert new values for the erased keys
    map.insert(new_pairs_begin, new_pairs_begin + num_keys / 2);
    REQUIRE(map.size() == num_keys);

    // Verify all keys exist
    map.contains(d_keys.begin(), d_keys.end(), d_keys_exist.begin());
    REQUIRE(cuco::test::all_of(d_keys_exist.begin(), d_keys_exist.end(), cuda::std::identity{}));

    // Verify values are correct
    thrust::device_vector<Value> d_found_values(num_keys);
    map.find(d_keys.begin(), d_keys.end(), d_found_values.begin());

    auto zip_equal = cuda::proclaim_return_type<bool>(
      [] __device__(auto const& p) { return cuda::std::get<0>(p) == cuda::std::get<1>(p); });

    // First half should have new values
    auto first_half_zip =
      thrust::make_zip_iterator(cuda::std::tuple{d_new_values.begin(), d_found_values.begin()});
    REQUIRE(cuco::test::all_of(first_half_zip, first_half_zip + num_keys / 2, zip_equal));

    // Second half should have original values
    auto second_half_zip = thrust::make_zip_iterator(
      cuda::std::tuple{d_values.begin() + num_keys / 2, d_found_values.begin() + num_keys / 2});
    REQUIRE(cuco::test::all_of(second_half_zip, second_half_zip + num_keys / 2, zip_equal));
  }

  SECTION("Check insert with custom hash function")
  {
    constexpr double default_load_factor = 0.60;
    cuco::flat_hash_map<Key, Value> identity_hash_map{num_keys,
                                                      cuco::empty_key<Key>{-1},
                                                      cuco::empty_value<Value>{-1},
                                                      cuco::erased_key<Key>{-2},
                                                      default_load_factor};

    thrust::device_vector<Key> d_keys(num_keys);
    thrust::device_vector<Value> d_values(num_keys);
    thrust::device_vector<bool> d_keys_exist(num_keys);

    thrust::sequence(thrust::device, d_keys.begin(), d_keys.end(), 1);
    thrust::sequence(thrust::device, d_values.begin(), d_values.end(), 1);

    auto pairs_begin =
      thrust::make_zip_iterator(cuda::std::tuple{d_keys.begin(), d_values.begin()});

    // Insert with identity hash function
    identity_hash_map.insert(pairs_begin, pairs_begin + num_keys, cuco::identity_hash<Key>());
    REQUIRE(identity_hash_map.size() == num_keys);

    // Verify all keys exist
    identity_hash_map.contains(
      d_keys.begin(), d_keys.end(), d_keys_exist.begin(), cuco::identity_hash<Key>());
    REQUIRE(cuco::test::all_of(d_keys_exist.begin(), d_keys_exist.end(), cuda::std::identity{}));

    // Verify values are correct
    thrust::device_vector<Value> d_found_values(num_keys);
    identity_hash_map.find(
      d_keys.begin(), d_keys.end(), d_found_values.begin(), cuco::identity_hash<Key>());

    auto zip_equal = cuda::proclaim_return_type<bool>(
      [] __device__(auto const& p) { return cuda::std::get<0>(p) == cuda::std::get<1>(p); });
    auto zip =
      thrust::make_zip_iterator(cuda::std::tuple{d_values.begin(), d_found_values.begin()});
    REQUIRE(cuco::test::all_of(zip, zip + num_keys, zip_equal));
  }

  SECTION("Check insert with mixed existing and new keys")
  {
    thrust::device_vector<Key> d_initial_keys(500);
    thrust::device_vector<Value> d_initial_values(500);
    thrust::device_vector<Key> d_mixed_keys(1000);
    thrust::device_vector<Value> d_mixed_values(1000);
    thrust::device_vector<bool> d_keys_exist(1000);

    thrust::sequence(thrust::device, d_initial_keys.begin(), d_initial_keys.end(), 1);
    thrust::sequence(thrust::device, d_initial_values.begin(), d_initial_values.end(), 1);

    auto initial_pairs_begin =
      thrust::make_zip_iterator(cuda::std::tuple{d_initial_keys.begin(), d_initial_values.begin()});

    // Insert initial keys
    map.insert(initial_pairs_begin, initial_pairs_begin + 500);
    REQUIRE(map.size() == 500);

    // Create mixed keys: first 500 are existing, last 500 are new
    thrust::copy(d_initial_keys.begin(), d_initial_keys.end(), d_mixed_keys.begin());
    thrust::sequence(thrust::device, d_mixed_keys.begin() + 500, d_mixed_keys.end(), 501);
    thrust::sequence(thrust::device, d_mixed_values.begin(), d_mixed_values.end(), 3001);

    auto mixed_pairs_begin =
      thrust::make_zip_iterator(cuda::std::tuple{d_mixed_keys.begin(), d_mixed_values.begin()});

    // Insert mixed keys
    map.insert(mixed_pairs_begin, mixed_pairs_begin + 1000);
    REQUIRE(map.size() == 1000);  // Should have 500 existing + 500 new

    // Verify all keys exist
    map.contains(d_mixed_keys.begin(), d_mixed_keys.end(), d_keys_exist.begin());
    REQUIRE(cuco::test::all_of(d_keys_exist.begin(), d_keys_exist.end(), cuda::std::identity{}));
  }
}
