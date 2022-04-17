#include <nvbench/nvbench.cuh>
#include <nvbench/detail/throw.cuh>

#include <thrust/device_vector.h>
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


template <typename T,
          cub::ForEachAlgorithm Algorithm>
static void basic(nvbench::state &state,
                  nvbench::type_list<T, 
                                     nvbench::enum_type<Algorithm>>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<T> input(elements);
  thrust::device_vector<int> counter(1);

  const T *d_input = thrust::raw_pointer_cast(input.data());
  int *d_counter = thrust::raw_pointer_cast(counter.data());

  state.add_element_count(elements);
  state.add_global_memory_reads(elements * sizeof(T));

  auto tuning = cub::TuneForEach<Algorithm>(
    cub::ForEachConfigurationSpace{}.Add<256, 4>());

  state.exec([&](nvbench::launch &launch) {
    cub::DeviceFor::ForEachN(d_input,
                             elements,
                             select_op<T>{d_counter},
                             launch.get_stream(),
                             false,
                             tuning);
  });
}


using types =
  nvbench::type_list<nvbench::uint32_t>;


using algorithms = nvbench::enum_type_list<cub::ForEachAlgorithm::BLOCK_STRIPED,
                                           cub::ForEachAlgorithm::VECTORIZED>;


NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  cub::ForEachAlgorithm,
  [](cub::ForEachAlgorithm algorithm) {
    switch (algorithm)
    {
      case cub::ForEachAlgorithm::BLOCK_STRIPED:
        return "BLOCK_STRIPED";
      case cub::ForEachAlgorithm::VECTORIZED:
        return "VECTORIZED";
      default:
        break;
    }
    NVBENCH_THROW(std::runtime_error, "{}", "Unknown data_pattern");
  },
  [](cub::ForEachAlgorithm algorithm) {
    switch (algorithm)
    {
      case cub::ForEachAlgorithm::BLOCK_STRIPED:
        return "Single element per load - block striped";
      case cub::ForEachAlgorithm::VECTORIZED:
        return "Four elements per load - block striped";
      default:
        break;
    }
    NVBENCH_THROW(std::runtime_error, "{}", "Unknown data_pattern");
  })


NVBENCH_BENCH_TYPES(basic,
                    NVBENCH_TYPE_AXES(types, algorithms))
  .set_name("cub::DeviceFor::ForEachN")
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 2));

