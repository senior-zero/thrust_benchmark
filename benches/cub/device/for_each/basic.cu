#include <nvbench/nvbench.cuh>

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/detail/raw_pointer_cast.h>

#include <cub/device/device_for.cuh>

template <typename T>
struct select_op
{
  int* m_counter{};

  __device__ void operator()(T val)
  {
    if (val == static_cast<T>(42))
    {
      atomicAdd(m_counter, 1);
    }
  }
};

template <typename T>
static void basic(nvbench::state &state,
                  nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<T> input(elements);
  thrust::device_vector<int> counter(1);

  const T *d_input = thrust::raw_pointer_cast(input.data());
  int *d_counter = thrust::raw_pointer_cast(counter.data());

  state.add_element_count(elements);
  state.add_global_memory_reads(elements * sizeof(T));

  state.exec([&](nvbench::launch &launch) {
    cub::DeviceFor::ForEachN(d_input, elements, select_op<T>{d_counter});
  });
}

using types =
  nvbench::type_list<nvbench::uint32_t, nvbench::uint64_t>;

NVBENCH_BENCH_TYPES(basic,
                    NVBENCH_TYPE_AXES(types))
  .set_name("cub::DeviceFor::ForEachN")
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 2));

