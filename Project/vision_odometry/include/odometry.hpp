#pragma once
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <string>
#include <vector>

namespace vo {
constexpr int MIN_POINTS = 20;
struct Pose {
    cv::Mat R = cv::Mat::eye(3, 3, CV_64F), C = cv::Mat::zeros(3, 1, CV_64F);
};
Pose compose(const Pose &, const cv::Mat &, const cv::Mat &, double scale = 1.);
Pose cameraPose(const cv::Mat &, const cv::Mat &);
bool coverage(const std::vector<cv::Point2d> &, cv::Size);
struct Calibration {
    cv::Size size;
    cv::Mat K, K0, D0, P0, P1, R1;
    std::vector<std::pair<cv::Mat, cv::Mat>> maps;
    std::string unit;
    Calibration(const std::string &, const std::string &);
    cv::Mat prepare(const cv::Mat &, int side = 0) const;
};
struct Features {
    std::vector<cv::Point2d> points;
    cv::Mat descriptors;
};
Features detect(const cv::Mat &);
using Pair = std::pair<int, int>;
std::vector<Pair> match(const Features &, const Features &);
struct Triangulation {
    std::vector<cv::Point3d> X;
    std::vector<uchar> valid;
    std::vector<double> error;
};
std::vector<double> parallax(const std::vector<cv::Point2d> &, const std::vector<cv::Point2d> &,
                             const cv::Mat &, const cv::Mat &);
Triangulation triangulate(const std::vector<cv::Point2d> &, const std::vector<cv::Point2d> &,
                          const cv::Mat &, const cv::Mat &, const cv::Mat &,
                          double min_parallax = .7);
struct Estimate {
    bool accepted = false;
    std::string state = "INSUFFICIENT";
    cv::Mat R = cv::Mat::eye(3, 3, CV_64F), t = cv::Mat::zeros(3, 1, CV_64F);
    std::vector<uchar> mask;
    std::vector<cv::Point3d> X;
    int inliers = 0;
    double ratio = 0, error = NAN, parallax_px = NAN, cheirality = NAN;
};
Estimate bootstrap(const std::vector<cv::Point2d> &, const std::vector<cv::Point2d> &,
                   const cv::Mat &, cv::Size);
Estimate pnp(const std::vector<cv::Point3d> &, const std::vector<cv::Point2d> &, const cv::Mat &,
             cv::Size);
using Map = std::map<int, cv::Point3d>;
std::pair<Map, double> stereoMap(const Features &, const Features &, const Calibration &);
class Odometry {
  public:
    Calibration cal;
    std::string mode;
    Pose pose, refPose;
    int segment = 0, failures = 0, age = 0;
    bool has_pose = false, has_ref = false;
    Features ref;
    Map landmarks;
    double epipolar = NAN;
    Odometry(const Calibration &c, const std::string &m) : cal(c), mode(m) {}
    Estimate step(const cv::Mat &, const cv::Mat &right = cv::Mat());

  private:
    void setRef(const Features &, const Map &);
    Estimate reject(const std::string &, Estimate);
};
cv::Mat column(const cv::Point3d &);
cv::Point3d point(const cv::Mat &);
} // namespace vo
