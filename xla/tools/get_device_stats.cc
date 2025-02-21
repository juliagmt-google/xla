#include <iostream>
#include <fstream>
#include <string>

#include "xla/tsl/platform/env.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xla/util.h"
#include "tsl/platform/init_main.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/debug_options_flags.h"

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
  tsl::Status status = tsl::ReadBinaryProto(env, input, &xspace_proto);
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

  return 0;
}
