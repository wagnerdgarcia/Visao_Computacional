#include "odometry.hpp"
#include <cmath>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x))                                                                                  \
            throw std::runtime_error(std::string("failed: ") + #x);                                \
    } while (0)
using namespace cv;
static std::vector<Point2d> project(const std::vector<Point3d> &X, const Mat &K, const Mat &R,
                                    const Mat &t) {
    std::vector<Point2d> p;
    Mat rv;
    Rodrigues(R, rv);
    projectPoints(X, rv, t, K, noArray(), p);
    return p;
}
int main(int argc, char **argv) {
    try {
        CHECK(argc == 2);
        setRNGSeed(42);
        setNumThreads(1);
        vo::Calibration cal(argv[1], "stereo");
        Mat K = cal.K, I = Mat::eye(3, 3, CV_64F), Z = Mat::zeros(3, 1, CV_64F), R;
        Mat rv = (Mat_<double>(3, 1) << .08, -.12, .03);
        Rodrigues(rv, R);
        Mat t = (Mat_<double>(3, 1) << -.45, .02, .08);
        vo::Pose old;
        Mat rvold = (Mat_<double>(3, 1) << .2, .4, -.3);
        Rodrigues(rvold, old.R);
        old.C = (Mat_<double>(3, 1) << 2, 3, 4);
        auto next = vo::compose(old, R, t);
        Mat X = (Mat_<double>(3, 1) << 1, 2, 7), world = old.R * X + old.C, now = R * X + t;
        CHECK(norm(next.R * now + next.C - world) < 1e-10);
        auto inv = vo::cameraPose(R, t);
        CHECK(norm(inv.R * now + inv.C - X) < 1e-10);
        std::cout << "PASS composition and PnP sign in rotated frames\n";
        RNG rng(7);
        std::vector<Point3d> pts;
        for (int i = 0; i < 240; ++i)
            pts.emplace_back(rng.uniform(-3., 3.), rng.uniform(-2., 2.), rng.uniform(4., 10.));
        auto a = project(pts, K, I, Z), b = project(pts, K, R, t);
        auto mono = vo::bootstrap(a, b, K, cal.size);
        CHECK(mono.accepted);
        CHECK(norm(mono.R - R) < 1e-4);
        CHECK(norm(mono.t / norm(mono.t) - t / norm(t)) < 1e-3);
        std::cout << "PASS two-view known 3D scene and relative gauge\n";
        auto est = vo::pnp(pts, b, K, cal.size);
        CHECK(est.accepted);
        CHECK(norm(est.R - R) < 1e-6);
        CHECK(norm(est.t - t) < 1e-6);
        std::cout << "PASS PnP absolute world-to-camera pose\n";
        auto contaminated = b;
        for (int i = 0; i < 48; ++i)
            contaminated[i] = Point2d(rng.uniform(0., 640.), rng.uniform(0., 480.));
        auto robust = vo::pnp(pts, contaminated, K, cal.size);
        CHECK(robust.accepted);
        CHECK(robust.inliers >= 180);
        CHECK(norm(robust.t - t) < 1e-4);
        int out = 0;
        for (int i = 0; i < 48; ++i)
            out += robust.mask[i];
        CHECK(out <= 2);
        auto bm = vo::bootstrap(a, contaminated, K, cal.size);
        CHECK(bm.accepted);
        out = 0;
        for (int i = 0; i < 48; ++i)
            out += bm.mask[i];
        CHECK(out <= 2);
        std::cout << "PASS robust outliers and retained RANSAC/cheirality mask\n";
        Mat baseline = (Mat_<double>(3, 1) << -.2, 0, 0);
        auto right = project(pts, K, I, baseline);
        auto tri = vo::triangulate(a, right, K, I, baseline, .5);
        CHECK(tri.X.size() == pts.size());
        for (size_t i = 0; i < pts.size(); ++i) {
            CHECK(tri.valid[i]);
            CHECK(norm(tri.X[i] - pts[i]) < 1e-8);
            CHECK(tri.error[i] < 1e-8);
        }
        auto metric = vo::pnp(tri.X, b, K, cal.size);
        CHECK(metric.accepted);
        CHECK(norm(metric.t - t) < 1e-6);
        auto wrong = vo::triangulate(a, right, K, I, -baseline, .5);
        int positive = 0;
        for (auto v : wrong.valid)
            positive += v;
        CHECK(positive == 0);
        std::cout << "PASS stereo metric scale, depth, reprojection and wrong "
                     "baseline sign\n";
        CHECK(!vo::bootstrap(a, a, K, cal.size).accepted);
        CHECK(vo::bootstrap(a, a, K, cal.size).state == "STATIONARY");
        auto rotated = project(pts, K, R, Z);
        auto pure = vo::bootstrap(a, rotated, K, cal.size);
        CHECK(!pure.accepted);
        CHECK(pure.state == "ROTATION_ONLY");
        std::vector<Point3d> plane = pts;
        for (auto &x : plane)
            x.z = 6.;
        CHECK(
            !vo::bootstrap(project(plane, K, I, Z), project(plane, K, R, t), K, cal.size).accepted);
        std::cout << "PASS stationary, pure rotation and planar degeneration\n";
        CHECK(!vo::pnp({}, {}, K, cal.size).accepted);
        CHECK(!vo::bootstrap({}, {}, K, cal.size).accepted);
        vo::Features empty, one;
        one.descriptors = Mat::zeros(1, 128, CV_32F);
        CHECK(vo::match(empty, one).empty());
        CHECK(vo::match(one, one).empty());
        vo::Odometry engine(cal, "mono");
        engine.pose.C = (Mat_<double>(3, 1) << 1, 2, 3);
        auto before = engine.pose.C.clone();
        Mat blank = Mat::zeros(cal.size, CV_8U);
        CHECK(!engine.step(blank).accepted);
        CHECK(norm(engine.pose.C - before) == 0);
        CHECK(!engine.step(blank).accepted);
        CHECK(norm(engine.pose.C - before) == 0);
        auto reset = engine.step(blank);
        CHECK(reset.state == "REINITIALIZING");
        CHECK(engine.segment == 1);
        CHECK(!engine.has_pose);
        bool failed = false;
        try {
            cal.prepare(Mat::zeros(10, 10, CV_8U));
        } catch (std::exception &) {
            failed = true;
        }
        CHECK(failed);
        failed = false;
        try {
            vo::Odometry(cal, "stereo").step(blank);
        } catch (std::exception &) {
            failed = true;
        }
        CHECK(failed);
        std::cout << "PASS absent descriptors, few points, loss/reset, no rejected "
                     "update, bad resolution and missing pair\n";
        // Non-rectified rotation is validated by the same OpenCV rectification
        // path.
        Mat Rst;
        Mat rvst = (Mat_<double>(3, 1) << .01, -.02, .005);
        Rodrigues(rvst, Rst);
        Mat R1, R2, P1, P2, Q;
        stereoRectify(K, cal.D0, K, cal.D0, cal.size, Rst, baseline, R1, R2, P1, P2, Q,
                      CALIB_ZERO_DISPARITY, 0);
        auto rawright = project(pts, K, Rst, baseline);
        std::vector<Point2d> al, ar;
        undistortPoints(a, al, K, cal.D0, R1, P1);
        undistortPoints(rawright, ar, K, cal.D0, R2, P2);
        for (size_t i = 0; i < al.size(); ++i)
            CHECK(std::abs(al[i].y - ar[i].y) < 1e-7);
        std::cout << "PASS nontrivial stereo rectification/epipolar "
                     "consistency\nAll C++ geometry checks passed\n";
        return 0;
    } catch (std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
