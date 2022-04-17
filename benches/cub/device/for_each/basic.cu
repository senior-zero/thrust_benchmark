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
          cub::ForEachAlgorithm Algorithm,
          cub::CacheLoadModifier LoadModifier>
static void basic(nvbench::state &state,
                  nvbench::type_list<T, 
                                     nvbench::enum_type<Algorithm>,
                                     nvbench::enum_type<LoadModifier>>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<T> input(elements);
  thrust::device_vector<int> counter(1);

  const T *d_input = thrust::raw_pointer_cast(input.data());
  int *d_counter = thrust::raw_pointer_cast(counter.data());

  state.add_element_count(elements);
  state.add_global_memory_reads(elements * sizeof(T));

  auto tuning = cub::TuneForEach<Algorithm, LoadModifier>(
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


using modifiers = nvbench::enum_type_list<cub::CacheLoadModifier::LOAD_DEFAULT,
                                          cub::CacheLoadModifier::LOAD_CA,
                                          cub::CacheLoadModifier::LOAD_CS>;


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
    NVBENCH_THROW(std::runtime_error, "{}", "Unknown algorithm");
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
    NVBENCH_THROW(std::runtime_error, "{}", "Unknown algorithm");
  })


NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  cub::CacheLoadModifier,
  [](cub::CacheLoadModifier modifier) {
    switch (modifier)
    {
      case cub::CacheLoadModifier::LOAD_DEFAULT:
        return "LOAD_DEFAULT";
      case cub::CacheLoadModifier::LOAD_CA:
        return "LOAD_CA";
      case cub::CacheLoadModifier::LOAD_CG:
        return "LOAD_CG";
      case cub::CacheLoadModifier::LOAD_CS:
        return "LOAD_CS";
      case cub::CacheLoadModifier::LOAD_CV:
        return "LOAD_CV";
      case cub::CacheLoadModifier::LOAD_LDG:
        return "LOAD_LDG";
      case cub::CacheLoadModifier::LOAD_VOLATILE:
        return "LOAD_VOLATILE";
      default:
        break;
    }
    NVBENCH_THROW(std::runtime_error, "{}", "Unknown load modifier");
  },
  [](cub::CacheLoadModifier modifier) {
    switch (modifier)
    {
      case cub::CacheLoadModifier::LOAD_DEFAULT:
        return "Default (no modifier)";
      case cub::CacheLoadModifier::LOAD_CA:
        return "Cache at all levels";
      case cub::CacheLoadModifier::LOAD_CG:
        return "Cache at global level";
      case cub::CacheLoadModifier::LOAD_CS:
        return "Cache streaming (likely to be accessed once)";
      case cub::CacheLoadModifier::LOAD_CV:
        return "Cache as volatile (including cached system lines)";
      case cub::CacheLoadModifier::LOAD_LDG:
        return "Cache as texture";
      case cub::CacheLoadModifier::LOAD_VOLATILE:
        return "Volatile (any memory space)";
      default:
        break;
    }
    NVBENCH_THROW(std::runtime_error, "{}", "Unknown load modifier");
  })


NVBENCH_BENCH_TYPES(basic,
                    NVBENCH_TYPE_AXES(types, algorithms, modifiers))
  .set_name("cub::DeviceFor::ForEachN")
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 2));

