#include "odometry.hpp"
#include <algorithm>
#include <cmath>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <set>
#include <stdexcept>

namespace vo {
using namespace cv;
static const Mat I = Mat::eye(3, 3, CV_64F), Z = Mat::zeros(3, 1, CV_64F);
Mat column(const Point3d &p) { return (Mat_<double>(3, 1) << p.x, p.y, p.z); }
Point3d point(const Mat &p) { return {p.at<double>(0), p.at<double>(1), p.at<double>(2)}; }
Pose compose(const Pose &p, const Mat &R, const Mat &t, double scale) {
    Pose q;
    q.R = p.R * R.t();
    q.C = p.C - scale * q.R * t;
    return q;
}
Pose cameraPose(const Mat &R, const Mat &t) {
    Pose p;
    p.R = R.t();
    p.C = -p.R * t;
    return p;
}
static double med(std::vector<double> x) {
    if (x.empty())
        return NAN;
    std::sort(x.begin(), x.end());
    size_t n = x.size();
    return n % 2 ? x[n / 2] : (x[n / 2 - 1] + x[n / 2]) / 2;
}
bool coverage(const std::vector<Point2d> &p, Size s) {
    if (p.size() < MIN_POINTS)
        return false;
    std::set<std::pair<int, int>> cells;
    double x0 = s.width, y0 = s.height, x1 = 0, y1 = 0;
    for (auto a : p) {
        cells.insert(
            {std::clamp(int(4 * a.x / s.width), 0, 3), std::clamp(int(3 * a.y / s.height), 0, 2)});
        x0 = std::min(x0, a.x);
        y0 = std::min(y0, a.y);
        x1 = std::max(x1, a.x);
        y1 = std::max(y1, a.y);
    }
    return cells.size() >= 6 && (x1 - x0) / s.width >= .2 && (y1 - y0) / s.height >= .2;
}
static Mat readArray(const FileNode &node, int cols) {
    if (node.empty() || !node.isSeq())
        throw std::runtime_error("missing calibration array");
    std::vector<double> a;
    for (auto x : node) {
        if (x.isSeq())
            for (auto y : x)
                a.push_back(double(y));
        else
            a.push_back(double(x));
    }
    if (a.empty() || a.size() % cols)
        throw std::runtime_error("invalid calibration dimensions");
    Mat m(a, true);
    m = m.reshape(1, int(a.size() / cols)).clone();
    if (!checkRange(m))
        throw std::runtime_error("nonfinite calibration");
    return m;
}
static void checkK(const Mat &K, const Mat &D) {
    Mat row = (Mat_<double>(1, 3) << 0, 0, 1);
    if (K.size() != Size(3, 3) || K.at<double>(0, 0) <= 0 || K.at<double>(1, 1) <= 0 ||
        norm(K.row(2) - row) > 1e-8 ||
        (D.total() != 4 && D.total() != 5 && D.total() != 8 && D.total() != 12 && D.total() != 14))
        throw std::runtime_error("invalid intrinsics/distortion");
}
static Mat projection(const Mat &K, const Mat &R, const Mat &t) {
    Mat e;
    hconcat(R, t, e);
    return K * e;
}
Calibration::Calibration(const std::string &path, const std::string &mode) {
    FileStorage fs(path, FileStorage::READ);
    if (!fs.isOpened())
        throw std::runtime_error("cannot read calibration");
    size = {int(fs["width"]), int(fs["height"])};
    if (size.width <= 0 || size.height <= 0)
        throw std::runtime_error("invalid calibration resolution");
    K0 = readArray(fs["K_left"], 3);
    D0 = readArray(fs["D_left"], 1);
    checkK(K0, D0);
    K = K0.clone();
    R1 = I.clone();
    unit = "relative";
    if (mode == "stereo") {
        fs["translation_unit"] >> unit;
        if (unit != "m" && unit != "cm" && unit != "mm" && unit != "arbitrary")
            throw std::runtime_error("missing/invalid translation unit");
        Mat K2 = readArray(fs["K_right"], 3), D2 = readArray(fs["D_right"], 1),
            R = readArray(fs["R_right_left"], 3), t = readArray(fs["T_right_left"], 1);
        checkK(K2, D2);
        if (R.size() != Size(3, 3) || t.total() != 3 || norm(R.t() * R - I) > 1e-5 ||
            std::abs(determinant(R) - 1) > 1e-5 || norm(t) < 1e-8)
            throw std::runtime_error("invalid stereo extrinsics");
        bool rectified = int(fs["already_rectified"]) != 0;
        if (rectified) {
            if (norm(D0) > 1e-10 || norm(D2) > 1e-10 || norm(R - I) > 1e-8 ||
                std::abs(t.at<double>(1)) + std::abs(t.at<double>(2)) > 1e-8 ||
                t.at<double>(0) >= 0 || norm(K0 - K2) > 1e-8)
                throw std::runtime_error(
                    "already_rectified requires common K, zero D, R=I, T=(-b,0,0)");
            P0 = projection(K, I, Z);
            P1 = projection(K2, R, t);
        } else {
            Mat R2, Q;
            stereoRectify(K0, D0, K2, D2, size, R, t, R1, R2, P0, P1, Q, CALIB_ZERO_DISPARITY, 0);
            if (std::abs(P1.at<double>(1, 3)) > 1e-7 || P1.at<double>(0, 3) >= 0)
                throw std::runtime_error("only horizontal left->right stereo is supported");
            K = P0(Rect(0, 0, 3, 3)).clone();
            maps.resize(2);
            initUndistortRectifyMap(K0, D0, R1, K, size, CV_32FC1, maps[0].first, maps[0].second);
            initUndistortRectifyMap(K2, D2, R2, P1(Rect(0, 0, 3, 3)), size, CV_32FC1, maps[1].first,
                                    maps[1].second);
        }
    } else if (mode != "mono")
        throw std::runtime_error("mode must be mono or stereo");
    else if (norm(D0) > 0) {
        maps.resize(1);
        initUndistortRectifyMap(K0, D0, I, K, size, CV_32FC1, maps[0].first, maps[0].second);
    }
}
Mat Calibration::prepare(const Mat &image, int side) const {
    if (image.empty() || image.size() != size)
        throw std::runtime_error("image missing or resolution incompatible with calibration");
    Mat gray;
    if (image.channels() == 3)
        cvtColor(image, gray, COLOR_BGR2GRAY);
    else
        gray = image;
    if (!maps.empty()) {
        Mat result;
        remap(gray, result, maps.at(side).first, maps.at(side).second, INTER_LINEAR);
        return result;
    }
    return gray;
}
Features detect(const Mat &im) {
    Features f;
    std::vector<KeyPoint> kp;
    SIFT::create(2400, 3, .025)->detectAndCompute(im, noArray(), kp, f.descriptors);
    for (auto k : kp)
        f.points.emplace_back(k.pt);
    return f;
}
std::vector<Pair> match(const Features &a, const Features &b) {
    std::vector<Pair> pairs;
    if (a.descriptors.empty() || b.descriptors.empty() ||
        std::min(a.descriptors.rows, b.descriptors.rows) < 2)
        return pairs;
    BFMatcher bf(NORM_L2);
    std::vector<std::vector<DMatch>> forward;
    std::vector<DMatch> reverse;
    bf.knnMatch(a.descriptors, b.descriptors, forward, 2);
    bf.match(b.descriptors, a.descriptors, reverse);
    std::map<int, int> back;
    for (auto m : reverse)
        back[m.queryIdx] = m.trainIdx;
    for (auto m : forward)
        if (m.size() >= 2 && m[0].distance < .75 * m[1].distance &&
            back[m[0].trainIdx] == m[0].queryIdx)
            pairs.push_back({m[0].queryIdx, m[0].trainIdx});
    return pairs;
}
static Mat ray(Point2d p, const Mat &K) { return K.inv() * (Mat_<double>(3, 1) << p.x, p.y, 1.); }
std::vector<double> parallax(const std::vector<Point2d> &a, const std::vector<Point2d> &b,
                             const Mat &K, const Mat &R) {
    std::vector<double> out;
    for (size_t i = 0; i < a.size(); ++i) {
        Mat q = R * ray(a[i], K), v = ray(b[i], K);
        out.push_back(
            std::hypot((q.at<double>(0) / q.at<double>(2) - v.at<double>(0)) * K.at<double>(0, 0),
                       (q.at<double>(1) / q.at<double>(2) - v.at<double>(1)) * K.at<double>(1, 1)));
    }
    return out;
}
static bool rotationOnly(const std::vector<Point2d> &a, const std::vector<Point2d> &b,
                         const Mat &K) {
    if (a.size() < MIN_POINTS)
        return false;
    Mat mask, H = findHomography(a, b, RANSAC, 1.5, mask);
    if (H.empty() || double(countNonZero(mask)) / a.size() < .8)
        return false;
    Mat A = K.inv() * H * K;
    if (determinant(A) < 0)
        A = -A;
    SVD sv(A);
    Mat R = sv.u * sv.vt;
    return determinant(R) > 0 && med(parallax(a, b, K, R)) < .7;
}
Triangulation triangulate(const std::vector<Point2d> &a, const std::vector<Point2d> &b,
                          const Mat &K, const Mat &R, const Mat &t, double minimum) {
    Triangulation z;
    if (a.empty())
        return z;
    Mat h;
    triangulatePoints(projection(K, I, Z), projection(K, R, t), a, b, h);
    h.convertTo(h, CV_64F);
    auto px = parallax(a, b, K, R);
    for (int i = 0; i < h.cols; ++i) {
        double w = h.at<double>(3, i);
        Mat X = h.col(i).rowRange(0, 3) / w, Y = R * X + t, p = K * X, q = K * Y;
        double e = std::max(std::hypot(p.at<double>(0) / p.at<double>(2) - a[i].x,
                                       p.at<double>(1) / p.at<double>(2) - a[i].y),
                            std::hypot(q.at<double>(0) / q.at<double>(2) - b[i].x,
                                       q.at<double>(1) / q.at<double>(2) - b[i].y));
        z.X.push_back(point(X));
        z.error.push_back(e);
        z.valid.push_back(std::abs(w) > 1e-10 && checkRange(X) && X.at<double>(2) > 0 &&
                          Y.at<double>(2) > 0 && e < 2. && px[i] >= minimum);
    }
    return z;
}
static std::vector<Point2d> select(const std::vector<Point2d> &p, const std::vector<uchar> &m) {
    std::vector<Point2d> out;
    for (size_t i = 0; i < p.size(); ++i)
        if (m[i])
            out.push_back(p[i]);
    return out;
}
static Mat spread(const std::vector<Point3d> &p) {
    Mat a(int(p.size()), 3, CV_64F);
    Point3d mean;
    for (auto x : p)
        mean += x;
    mean *= 1. / p.size();
    for (size_t i = 0; i < p.size(); ++i) {
        a.at<double>(int(i), 0) = p[i].x - mean.x;
        a.at<double>(int(i), 1) = p[i].y - mean.y;
        a.at<double>(int(i), 2) = p[i].z - mean.z;
    }
    Mat w;
    SVD::compute(a, w);
    return w;
}
Estimate bootstrap(const std::vector<Point2d> &a, const std::vector<Point2d> &b, const Mat &K,
                   Size size) {
    Estimate z;
    z.mask.resize(a.size(), 0);
    if (a.size() != b.size() || a.size() < MIN_POINTS || !coverage(a, size) || !coverage(b, size))
        return z;
    std::vector<double> motion;
    for (size_t i = 0; i < a.size(); ++i)
        motion.push_back(norm(a[i] - b[i]));
    if (med(motion) < .4) {
        z.state = "STATIONARY";
        return z;
    }
    if (rotationOnly(a, b, K)) {
        z.state = "ROTATION_ONLY";
        return z;
    }
    Mat rm, E = findEssentialMat(a, b, K, RANSAC, .999, 1., rm);
    if (E.empty() || E.size() != Size(3, 3) || rm.empty()) {
        z.state = "ESSENTIAL_REJECTED";
        return z;
    }
    int nr = countNonZero(rm);
    if (nr < MIN_POINTS || double(nr) / a.size() < .45) {
        z.state = "RANSAC_REJECTED";
        return z;
    }
    Mat pm = rm.clone(), R, t;
    int n = recoverPose(E, a, b, K, R, t, pm);
    z.cheirality = double(n) / nr;
    auto tri = triangulate(a, b, K, R, t);
    std::vector<double> errors, pp;
    auto px = parallax(a, b, K, R);
    std::vector<Point3d> good;
    for (size_t i = 0; i < a.size(); ++i)
        if (tri.valid[i] && rm.at<uchar>(int(i)) && pm.at<uchar>(int(i))) {
            z.mask[i] = 1;
            z.inliers++;
            errors.push_back(tri.error[i]);
            pp.push_back(px[i]);
            good.push_back(tri.X[i]);
        }
    z.ratio = double(z.inliers) / a.size();
    z.error = med(errors);
    z.parallax_px = med(pp);
    z.state = "DEGENERATE";
    if (z.inliers < MIN_POINTS || z.ratio < .45 || z.cheirality < .7 || z.parallax_px < 1. ||
        !coverage(select(a, z.mask), size) || !coverage(select(b, z.mask), size))
        return z;
    Mat hm;
    findHomography(a, b, RANSAC, 1.5, hm);
    if (!hm.empty() && double(countNonZero(hm)) / a.size() > .9) {
        z.state = "PLANAR";
        return z;
    }
    Mat s = spread(good);
    if (s.at<double>(1) / s.at<double>(0) < .02 || s.at<double>(2) / s.at<double>(0) < .005)
        return z;
    z.accepted = true;
    z.state = "INITIALIZED";
    z.R = R;
    z.t = t;
    z.X = tri.X;
    return z;
}
Estimate pnp(const std::vector<Point3d> &X, const std::vector<Point2d> &p, const Mat &K,
             Size size) {
    Estimate z;
    z.mask.resize(p.size(), 0);
    if (p.size() != X.size() || p.size() < MIN_POINTS || !coverage(p, size))
        return z;
    for (auto x : X)
        if (!std::isfinite(x.x) || !std::isfinite(x.y) || !std::isfinite(x.z))
            return z;
    Mat s = spread(X);
    if (s.at<double>(0) <= 0 || s.at<double>(1) / s.at<double>(0) < .01) {
        z.state = "DEGENERATE";
        return z;
    }
    Mat rv, tv, idx;
    bool ok = solvePnPRansac(X, p, K, noArray(), rv, tv, false, 200, 2., .999, idx, SOLVEPNP_EPNP);
    if (!ok || idx.total() < MIN_POINTS || double(idx.total()) / p.size() < .45) {
        z.state = "PNP_REJECTED";
        return z;
    }
    std::vector<Point3d> Xi;
    std::vector<Point2d> pi;
    for (int j = 0; j < int(idx.total()); ++j) {
        int i = idx.at<int>(j);
        Xi.push_back(X[i]);
        pi.push_back(p[i]);
    }
    solvePnPRefineLM(Xi, pi, K, noArray(), rv, tv);
    Mat R;
    Rodrigues(rv, R);
    std::vector<Point2d> q;
    projectPoints(X, rv, tv, K, noArray(), q);
    std::vector<double> errors;
    int positive = 0;
    for (int j = 0; j < int(idx.total()); ++j) {
        int i = idx.at<int>(j);
        Mat Y = R * column(X[i]) + tv;
        bool depth = Y.at<double>(2) > 0;
        positive += depth;
        double e = norm(q[i] - p[i]);
        if (depth && e < 2.) {
            z.mask[i] = 1;
            z.inliers++;
            errors.push_back(e);
        }
    }
    z.ratio = double(z.inliers) / p.size();
    z.error = med(errors);
    z.cheirality = double(positive) / idx.total();
    if (z.inliers < MIN_POINTS || z.ratio < .45 || z.error > 1.5 || z.cheirality < .9 ||
        !coverage(select(p, z.mask), size)) {
        z.state = "PNP_REJECTED";
        return z;
    }
    z.accepted = true;
    z.state = "TRACKING";
    z.R = R;
    z.t = tv;
    return z;
}
std::pair<Map, double> stereoMap(const Features &left, const Features &right,
                                 const Calibration &cal) {
    Map map;
    auto pairs = match(left, right);
    if (pairs.size() < MIN_POINTS)
        return {map, NAN};
    std::vector<double> ep;
    std::vector<Point2d> a, b;
    std::vector<int> ids;
    int aligned = 0;
    for (auto ij : pairs) {
        auto x = left.points[ij.first], y = right.points[ij.second];
        double e = std::abs(x.y - y.y);
        ep.push_back(e);
        aligned += e < 2.;
    }
    double epi = med(ep);
    if (epi > 1.5 || double(aligned) / pairs.size() < .6)
        return {map, epi};
    for (size_t j = 0; j < pairs.size(); ++j) {
        auto x = left.points[pairs[j].first], y = right.points[pairs[j].second];
        if (ep[j] < 1.5 && x.x - y.x > .5) {
            a.push_back(x);
            b.push_back(y);
            ids.push_back(pairs[j].first);
        }
    }
    Mat t = (Mat_<double>(3, 1) << cal.P1.at<double>(0, 3) / cal.K.at<double>(0, 0), 0, 0);
    auto tri = triangulate(a, b, cal.K, I, t, .5);
    for (size_t j = 0; j < ids.size(); ++j)
        if (tri.valid[j])
            map[ids[j]] = tri.X[j];
    return {map, epi};
}
void Odometry::setRef(const Features &f, const Map &map) {
    ref = f;
    landmarks = map;
    refPose.R = pose.R.clone();
    refPose.C = pose.C.clone();
    age = 0;
    has_ref = true;
}
Estimate Odometry::reject(const std::string &state, Estimate z) {
    if (state != "STATIONARY" && state != "ROTATION_ONLY" && state != "PLANAR" &&
        state != "DEGENERATE")
        failures++;
    if (failures >= 3) {
        segment++;
        pose = Pose();
        has_pose = false;
        has_ref = false;
        landmarks.clear();
        failures = 0;
        z.state = "REINITIALIZING";
    } else
        z.state = state;
    return z;
}
Estimate Odometry::step(const Mat &im, const Mat &right) {
    Features f = detect(cal.prepare(im));
    epipolar = NAN;
    Map stereo;
    if (mode == "stereo") {
        if (right.empty())
            throw std::runtime_error("stereo requires synchronized right image");
        auto result = stereoMap(f, detect(cal.prepare(right, 1)), cal);
        stereo = result.first;
        epipolar = result.second;
        if (stereo.size() < MIN_POINTS)
            return reject("STEREO_REJECTED", Estimate());
    }
    if (f.points.size() < MIN_POINTS)
        return reject("NO_FEATURES", Estimate());
    if (!has_ref) {
        setRef(f, stereo);
        has_pose = mode == "stereo";
        Estimate z;
        z.state = "ANCHOR";
        z.accepted = has_pose;
        return z;
    }
    auto pairs = match(ref, f);
    age++;
    if (pairs.size() < MIN_POINTS)
        return reject("LOST", Estimate());
    std::vector<Point2d> a, b;
    for (auto ij : pairs) {
        a.push_back(ref.points[ij.first]);
        b.push_back(f.points[ij.second]);
    }
    Estimate z;
    if (landmarks.empty()) {
        z = bootstrap(a, b, cal.K, cal.size);
        if (!z.accepted)
            return reject(z.state, z);
        pose = compose(refPose, z.R, z.t);
        Map map;
        for (size_t j = 0; j < pairs.size(); ++j)
            if (z.mask[j])
                map[pairs[j].second] = point(refPose.R * column(z.X[j]) + refPose.C);
        setRef(f, map);
        has_pose = true;
    } else {
        std::vector<int> ids;
        std::vector<Point3d> X;
        std::vector<Point2d> p;
        for (size_t j = 0; j < pairs.size(); ++j)
            if (landmarks.count(pairs[j].first)) {
                ids.push_back(int(j));
                X.push_back(landmarks.at(pairs[j].first));
                p.push_back(b[j]);
            }
        z = pnp(X, p, cal.K, cal.size);
        if (!z.accepted)
            return reject(z.state, z);
        Pose candidate = cameraPose(z.R, z.t);
        if (!checkRange(candidate.C))
            return reject("NONFINITE", Estimate());
        pose = candidate;
        if (mode == "stereo") {
            Map map;
            for (auto ix : stereo)
                map[ix.first] = point(pose.R * column(ix.second) + pose.C);
            setRef(f, map);
        } else if (age >= 8 || z.inliers < .6 * landmarks.size()) {
            Map map;
            for (size_t j = 0; j < ids.size(); ++j)
                if (z.mask[j])
                    map[pairs[ids[j]].second] = X[j];
            Mat R = pose.R.t() * refPose.R, t = pose.R.t() * (refPose.C - pose.C);
            auto tri = triangulate(a, b, cal.K, R, t, .7);
            for (size_t j = 0; j < pairs.size(); ++j)
                if (tri.valid[j] && !map.count(pairs[j].second))
                    map[pairs[j].second] = point(refPose.R * column(tri.X[j]) + refPose.C);
            if (map.size() >= MIN_POINTS)
                setRef(f, map);
        }
    }
    failures = 0;
    return z;
}
} // namespace vo
