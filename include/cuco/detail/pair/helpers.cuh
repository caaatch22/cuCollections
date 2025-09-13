/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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
 */

#pragma once

#include <cuda/functional>
#include <cuda/std/bit>
#include <cuda/std/type_traits>

#include <cstdint>

namespace cuco::detail {

/**
 * @brief Gives value to use as alignment for a pair type that is at least the
 * size of the sum of the size of the first type and second type, or 16,
 * whichever is smaller.
 */
template <typename First, typename Second>
__host__ __device__ constexpr std::size_t pair_alignment()
{
  constexpr std::size_t alignment = cuda::std::bit_ceil(sizeof(First) + sizeof(Second));
  return cuda::std::min(std::size_t{16}, alignment);
}

template <typename value_type>
using select_packed_type = cuda::std::conditional_t<sizeof(value_type) == 1, cuda::std::uint8_t,
                           cuda::std::conditional_t<sizeof(value_type) == 2, cuda::std::uint16_t,
                           cuda::std::conditional_t<sizeof(value_type) == 4, cuda::std::uint32_t,
                           cuda::std::conditional_t<sizeof(value_type) == 8, cuda::std::uint64_t,
#if (__CUDA_ARCH__ >= 900)
                           cuda::std::conditional_t<sizeof(value_type) == 16, __int128_t,
                           void>>>>>;
#endif
                           void>>>>;


/**
 * @brief Indicates if a pair type can be packed.
 *
 * When the size of the key,value pair being inserted into the hash table is
 * equal in size to a type where atomicCAS is natively supported, it is more
 * efficient to "pack" the pair and insert it with a single atomicCAS.
 *
 * Pair types whose key and value have the same object representation may be
 * packed. Also, the `Pair` must not contain any padding bits otherwise
 * accessing the packed value would be undefined.
 *
 * @tparam Pair The pair type that will be packed
 *
 * @return true If the pair type can be packed
 * @return false  If the pair type cannot be packed
 */
template <typename Pair>
__host__ __device__ constexpr bool is_packable()
{
  return not cuda::std::is_void_v<packed_t<Pair>> and
         cuda::std::has_unique_object_representations_v<Pair>;
}

}  // namespace cuco::detail
