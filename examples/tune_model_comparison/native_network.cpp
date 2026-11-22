// Read the complete report/stub and exercise the production array solvers.
#include <filesystem>
#include <fstream>
#include <iostream>
#include <kids/io/toltec.h>
#include <kids/model/joint.h>
#include <numeric>
#include <yaml-cpp/yaml.h>

using namespace kids::reduce;
namespace fs = std::filesystem;

Matrix read_matrix(const std::string &path, Index rows, Index cols) {
  Matrix value(rows, cols);
  std::ifstream stream(path, std::ios::binary);
  if (!stream.read(reinterpret_cast<char *>(value.data()),
                   value.size() * sizeof(double)))
    throw std::runtime_error("Incomplete matrix: " + path);
  return value;
}
template <class T> void write_matrix(const fs::path &path, const T &value) {
  std::ofstream stream(path, std::ios::binary);
  stream.write(reinterpret_cast<const char *>(value.data()),
               value.size() * sizeof(typename T::Scalar));
  if (!stream)
    throw std::runtime_error("Cannot write: " + path.string());
}

int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  try {
    const auto request = YAML::LoadFile(argv[1]);
    const auto rows = request["samples"].as<Index>(),
               cols = request["channels"].as<Index>();
    const auto i = read_matrix(request["I"].as<std::string>(), rows, cols);
    const auto q = read_matrix(request["Q"].as<std::string>(), rows, cols);
    std::vector<Index> channels(cols);
    std::iota(channels.begin(), channels.end(), 0);
    for (const auto mode : {"ecsv", "stub"}) {
      auto table = std::string(mode) == "ecsv"
                       ? kids::io::read_model_params_ecsv(
                             request[mode].as<std::string>())
                       : kids::io::read_model_params_header(
                             request[mode].as<std::string>(),
                             request["block"].as<int>());
      const fs::path root =
          fs::path(request["output"].as<std::string>()) / mode;
      fs::create_directories(root);
      const TimeStreamSolver solver(table, channels);
      auto result =
          solve_timestream(i, q, solver, request["normalization"].as<double>());
      write_matrix(root / "r.bin", result.r);
      write_matrix(root / "x.bin", result.x);
      write_matrix(root / "flags.bin", result.flags);
      YAML::Node summary;
      summary["samples"] = rows;
      summary["channels"] = cols;
      summary["accumlen"] = table.accumlen.value_or(-1);
      summary["joint"] = YAML::Node(YAML::NodeType::Sequence);
      for (const auto &group : result.joint) {
        const auto &cal = group.calibration;
        const auto path = root / ("group-" + cal.group_id);
        fs::create_directories(path);
        write_matrix(path / "r.bin", group.r);
        write_matrix(path / "fr.bin", group.fr);
        write_matrix(path / "flags.bin", group.flags);
        write_matrix(path / "residual.bin", group.residual_norm);
        YAML::Node item;
        item["group_id"] = cal.group_id;
        item["resonator_ids"] = cal.resonator_ids;
        item["channels"] = cal.channels;
        item["fit_flag"] = cal.fit_flag;
        item["rank"] = group.rank;
        item["condition"] = group.condition;
        item["model_report"] = YAML::Load(cal.report_model);
        summary["joint"].push_back(item);
      }
      YAML::Emitter text;
      text.SetDoublePrecision(17);
      text << summary;
      std::ofstream(root / "summary.yaml") << text.c_str();
      std::cerr << mode << ": " << rows << " samples, " << cols << " channels, "
                << result.joint.size() << " joint groups\n";
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
