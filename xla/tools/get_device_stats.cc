#include <iostream>
#include <fstream>
#include <string>

#include "xla/tsl/platform/env.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xla/util.h"
#include "tsl/platform/init_main.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/debug_options_flags.h"
#include "absl/container/flat_hash_map.h"

using tensorflow::profiler::XPlane;
using tensorflow::profiler::XSpace;

namespace {
const char* const kUsage = R"(
This tool prints the compute cost (flops and memory traffic) of an HLO module.

The input file can be obtained from XProf graph viewer by clicking
"Download as short text".

Usage:

    bazel run compute_cost -- --input=path/to/xspace_pb --format=[hlo|pb|pbtxt] [--gpu] [--all]
)";


absl::Status CalculateDeviceTimeAndMemcpy(const XSpace& xspace,
                                          const std::string& device_name,
                                          double* device_time_us,
                                          double* device_memcpy_time_us) {
  int64_t total_time_ps = 0;
  int64_t memcpy_time_ps = 0;

  for (const XPlane& plane : xspace.planes()) {
    if (plane.name() == device_name) {
      absl::flat_hash_map<std::string, int64_t> stat_metadata_map;
      for (const auto& stat_metadata : plane.stat_metadata()) {
        stat_metadata_map[stat_metadata.second.name()] =
            stat_metadata.second.id();
      }
      int64_t memcpy_details_id = stat_metadata_map.contains("memcpy_details")
                                      ? stat_metadata_map["memcpy_details"]
                                      : -1;

      for (const auto& line : plane.lines()) {
        for (const auto& event : line.events()) {
          bool is_memcpy = false;
          for (const auto& stat : event.stats()) {
            if (stat.metadata_id() == memcpy_details_id) {
              is_memcpy = true;
              break;
            }
          }
          total_time_ps += event.duration_ps();
          if (is_memcpy) {
            memcpy_time_ps += event.duration_ps();
          }
        }
      }
    }
  }

  *device_time_us = static_cast<double>(total_time_ps) / 1e6;
  *device_memcpy_time_us = static_cast<double>(memcpy_time_ps) / 1e6;
  return absl::OkStatus();
}
}  // namespace

int main(int argc, char* argv[]) {
  std::string input, format;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("input", &input, "input file")};
  xla::AppendDebugOptionsFlags(&flag_list);
  const std::string kUsageString =
      absl::StrCat(kUsage, "\n\n", tsl::Flags::Usage(argv[0], flag_list));
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  tsl::port::InitMain(kUsageString.c_str(), &argc, &argv);

  std::cout << "Input file: " << input << std::endl;
  tsl::Env* env = tsl::Env::Default();
  XSpace xspace_proto;
  tsl::Status status = tsl::ReadTextProto(env, input, &xspace_proto);
  std::cout << "Reading and parsing protobuf: " << status.ToString() << "\n";

  if (!status.ok()) {
    std::cerr << "Error reading and parsing protobuf: " << status.ToString() << "\n";
    return 1;
  }

  // Example: Accessing and printing data from the parsed protobuf.
  // Replace with your actual logic to process the XSpace proto.
  std::cout << "Successfully parsed XSpace proto.\n";

  if (status.ok()) {
    std::cout << "XSpace Name: " << xspace_proto.DebugString() << std::endl;
  }

  double device_time_us;
  double device_memcpy_time_us;
  std::string device_name = "/device:GPU:0";
  auto s = CalculateDeviceTimeAndMemcpy(
    xspace_proto, device_name,
      &device_time_us, &device_memcpy_time_us);
  if (!s.ok()) {
    std::cerr << "Error calculating device time and memcpy: " << s <<
    std::endl; return 1;
  }

  std::cout << absl::StrFormat("Device Time: %.2f us\n", device_time_us)
            << absl::StrFormat("Device Memcpy Time: %.2f us\n",
                               device_memcpy_time_us);

  return 0;
}
