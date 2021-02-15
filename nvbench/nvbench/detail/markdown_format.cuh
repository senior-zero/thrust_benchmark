#pragma once

#include <nvbench/benchmark_base.cuh>

#include <memory>
#include <vector>

namespace nvbench::detail
{

struct markdown_format
{
  using benchmark_vector = std::vector<std::unique_ptr<benchmark_base>>;

  static void print_device_info();
  static void print_log_preamble();
  static void print_log_epilogue();

  static void print_benchmark_summaries(const benchmark_vector &benchmarks);

  // Hacked in to just print a basic summary table to stdout. There's lots of
  // room for improvement here.
  void print_benchmark_results(const benchmark_vector &benchmarks);
};

} // namespace nvbench::detail
