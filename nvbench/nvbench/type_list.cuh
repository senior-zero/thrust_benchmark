#pragma once

#include "detail/type_list_impl.h"

#include <tuple>
#include <type_traits>

namespace nvbench
{

template <typename... Ts>
struct type_list
{};

namespace tl
{

/**
 * Get the size of a type_list as a `std::integral_constant<size_t, N>`.
 *
 * ```c++
 * using TL = nvbench::type_list<T0, T1, T2, T3, T4>;
 * static_assert(nvbench::tl::size<TL>::value == 5);
 * ```
 */
template <typename TypeList>
using size = decltype(detail::size(TypeList{}));

/**
 * Get the type at the specified index of a type_list.
 *
 * ```c++
 * using TL = nvbench::type_list<T0, T1, T2, T3, T4>;
 * static_assert(std::is_same_v<nvbench::tl::get<0, TL>, T0>);
 * static_assert(std::is_same_v<nvbench::tl::get<1, TL>, T1>);
 * static_assert(std::is_same_v<nvbench::tl::get<2, TL>, T2>);
 * static_assert(std::is_same_v<nvbench::tl::get<3, TL>, T3>);
 * static_assert(std::is_same_v<nvbench::tl::get<4, TL>, T4>);
 * ```
 */
template <std::size_t Index, typename TypeList>
using get = decltype(detail::get<Index>(TypeList{}));

/**
 * Concatenate two type_lists.
 *
 * ```c++
 * using TL01 = nvbench::type_list<T0, T1>;
 * using TL23 = nvbench::type_list<T2, T3>;
 * using TL0123 = nvbench::type_list<T0, T1, T2, T3>;
 * static_assert(std::is_same_v<nvbench::tl::concat<TL01, TL23>, T0123>);
 * ```
 */
template <typename TypeList1, typename TypeList2>
using concat = decltype(detail::concat(TypeList1{}, TypeList2{}));

/**
 * Given a type `T` and a type_list of type_lists `TypeLists`, create
 * a new type_list containing each entry from TypeLists prepended with T.
 *
 * ```c++
 *  using TypeLists = type_list<type_list<T0, T1>,
 *                              type_list<T2, T3>>;
 *  using Result = nvbench::tl::prepend_each<T, TypeLists>;
 *  using Reference = type_list<type_list<T, T0, T1>,
 *                              type_list<T, T2, T3>>;
 *  static_assert(std::is_same_v<Result, Reference>);
 * ```
 */
template <typename T, typename TypeLists>
using prepend_each = typename detail::prepend_each<T, TypeLists>::type;

/**
 * Given a type_list of type_lists, compute the cartesian product across all
 * nested type_lists. Supports arbitrary numbers and sizes of nested type_lists.
 *
 * Beware that the result grows very quickly in size.
 *
 * ```c++
 * using T01 = type_list<T0, T1>;
 * using U012 = type_list<U0, U1, U2>;
 * using V01 = type_list<V0, V1>;
 * using TLs = type_list<T01, U012, V01>;
 * using CartProd = type_list<type_list<T0, U0, V0>,
 *                            type_list<T0, U0, V1>,
 *                            type_list<T0, U1, V0>,
 *                            type_list<T0, U1, V1>,
 *                            type_list<T0, U2, V0>,
 *                            type_list<T0, U2, V1>,
 *                            type_list<T1, U0, V0>,
 *                            type_list<T1, U0, V1>,
 *                            type_list<T1, U1, V0>,
 *                            type_list<T1, U1, V1>,
 *                            type_list<T1, U2, V0>,
 *                            type_list<T1, U2, V1>>;
 *  static_assert(std::is_same_v<bench::tl::cartesian_product<TLs>, CartProd>);
 * ```
 */
template <typename TypeLists>
using cartesian_product = typename detail::cartesian_product<TypeLists>::type;

} // namespace tl

} // namespace nvbench
