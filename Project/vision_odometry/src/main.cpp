#include "odometry.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <set>
#include <sstream>
#include <stdexcept>
namespace fs = std::filesystem;
static std::vector<std::string> split(const std::string &line) {
    std::vector<std::string> out;
    std::string cell;
    bool quoted = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
                cell += '"';
                i++;
            } else
                quoted = !quoted;
        } else if (c == ',' && !quoted) {
            out.push_back(cell);
            cell.clear();
        } else if (c != '\r')
            cell += c;
    }
    if (quoted)
        throw std::runtime_error("unterminated CSV quote");
    out.push_back(cell);
    return out;
}
struct Row {
    int index;
    double timestamp;
    fs::path left, right;
};
static std::vector<Row> manifest(const std::string &path, const std::string &mode,
                                 double tolerance) {
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("cannot open manifest");
    std::string line;
    std::getline(f, line);
    auto headers = split(line);
    std::map<std::string, size_t> col;
    for (size_t i = 0; i < headers.size(); ++i)
        col[headers[i]] = i;
    for (auto name : {"index", "timestamp", "left"})
        if (!col.count(name))
            throw std::runtime_error("manifest missing column");
    if (mode == "stereo" && (!col.count("right") || !col.count("right_timestamp")))
        throw std::runtime_error("missing stereo pair/timestamp");
    std::vector<Row> rows;
    std::set<fs::path> ls, rs;
    double last = -INFINITY;
    int lasti = -1;
    auto root = fs::absolute(path).parent_path();
    while (std::getline(f, line)) {
        auto cells = split(line);
        if (cells.size() != headers.size())
            throw std::runtime_error("invalid manifest CSV row");
        auto get = [&](const std::string &k) { return cells.at(col.at(k)); };
        Row row{std::stoi(get("index")), std::stod(get("timestamp")), root / get("left"), {}};
        if (!std::isfinite(row.timestamp) || row.timestamp <= last || row.index <= lasti)
            throw std::runtime_error("manifest timestamps/indices must strictly increase");
        last = row.timestamp;
        lasti = row.index;
        if (get("left").empty() || !fs::is_regular_file(row.left) ||
            !ls.insert(fs::canonical(row.left)).second)
            throw std::runtime_error("left image absent or reused");
        if (mode == "stereo") {
            if (get("right").empty() || get("right_timestamp").empty())
                throw std::runtime_error("missing stereo pair/timestamp");
            double rt = std::stod(get("right_timestamp"));
            if (!std::isfinite(rt) || std::abs(rt - row.timestamp) > tolerance)
                throw std::runtime_error("stereo pair out of synchronization tolerance");
            row.right = root / get("right");
            if (!fs::is_regular_file(row.right) ||
                fs::canonical(row.right) == fs::canonical(row.left) ||
                !rs.insert(fs::canonical(row.right)).second)
                throw std::runtime_error("right image absent, duplicated or reused");
        }
        rows.push_back(row);
    }
    if (rows.empty())
        throw std::runtime_error("empty manifest");
    return rows;
}
int main(int argc, char **argv) {
    try {
        const std::string help =
            "Usage: vision_odometry --mode mono|stereo --calibration "
            "calibration.json (--manifest frames.csv | --video mono.mp4) --output "
            "trajectory.csv [--stride 1 --max-frames 0 --sync-tolerance 0.002]\n";
        std::map<std::string, std::string> args;
        std::set<std::string> allowed = {"--mode",       "--calibration",   "--manifest",
                                         "--video",      "--output",        "--stride",
                                         "--max-frames", "--sync-tolerance"};
        for (int i = 1; i < argc; i += 2) {
            std::string key = argv[i];
            if (key == "--help") {
                std::cout << help;
                return 0;
            }
            if (!allowed.count(key) || i + 1 >= argc || args.count(key))
                throw std::runtime_error(help);
            args[key] = argv[i + 1];
        }
        if (!args.count("--mode") || !args.count("--calibration") || !args.count("--output") ||
            args.count("--manifest") + args.count("--video") != 1)
            throw std::runtime_error(help);
        auto mode = args.at("--mode");
        if (mode != "mono" && mode != "stereo")
            throw std::runtime_error("mode must be mono or stereo");
        bool video = args.count("--video");
        if (video && mode == "stereo")
            throw std::runtime_error("stereo requires real pair manifest; video is mono only");
        int stride = args.count("--stride") ? std::stoi(args.at("--stride")) : 1,
            limit = args.count("--max-frames") ? std::stoi(args.at("--max-frames")) : 0;
        double tol = args.count("--sync-tolerance") ? std::stod(args.at("--sync-tolerance")) : .002;
        if (stride < 1 || limit < 0 || !std::isfinite(tol) || tol < 0)
            throw std::runtime_error("invalid stride/frame limit/tolerance");
        cv::setRNGSeed(42);
        cv::setNumThreads(1);
        vo::Calibration cal(args.at("--calibration"), mode);
        vo::Odometry engine(cal, mode);
        cv::VideoCapture cap;
        double fps = 0;
        std::vector<Row> rows;
        if (video) {
            cap.open(args.at("--video"));
            fps = cap.get(cv::CAP_PROP_FPS);
            if (!cap.isOpened() || fps <= 0)
                throw std::runtime_error("cannot open video or invalid FPS");
        } else
            rows = manifest(args.at("--manifest"), mode, tol);
        fs::path output = args.at("--output");
        if (!output.parent_path().empty())
            fs::create_directories(output.parent_path());
        std::ofstream out(output);
        if (!out)
            throw std::runtime_error("cannot write CSV");
        out << "index,timestamp,segment,state,accepted,pose_valid,unit,x,y,z,r00,"
               "r01,r02,r10,r11,r12,r20,r21,r22,inliers,inlier_ratio,reprojection_"
               "px,parallax_px,cheirality_ratio,epipolar_px,map_points\n";
        out << std::setprecision(17);
        int k = 0, count = 0;
        while (!limit || count < limit) {
            cv::Mat left, right;
            int index = k;
            double ts;
            if (video) {
                if (!cap.read(left))
                    break;
                ts = k / fps;
            } else {
                if (k >= int(rows.size()))
                    break;
                auto row = rows[k];
                index = row.index;
                ts = row.timestamp;
                if (k % stride == 0) {
                    left = cv::imread(row.left.string(), cv::IMREAD_GRAYSCALE);
                    if (mode == "stereo")
                        right = cv::imread(row.right.string(), cv::IMREAD_GRAYSCALE);
                }
            }
            if (k++ % stride)
                continue;
            auto z = engine.step(left, right);
            out << index << ',' << ts << ',' << engine.segment << ',' << z.state << ','
                << z.accepted << ',' << engine.has_pose << ',' << cal.unit;
            for (int i = 0; i < 3; ++i)
                out << ',' << engine.pose.C.at<double>(i);
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    out << ',' << engine.pose.R.at<double>(i, j);
            out << ',' << z.inliers << ',' << z.ratio << ',' << z.error << ',' << z.parallax_px
                << ',' << z.cheirality << ',' << engine.epipolar << ',' << engine.landmarks.size()
                << '\n';
            count++;
        }
        if (!count)
            throw std::runtime_error("no frames processed");
        if (!out)
            throw std::runtime_error("CSV write failed");
        std::cout << count << " frames; " << engine.segment + 1 << " segments; CSV: " << output
                  << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << '\n';
        return 2;
    }
}
