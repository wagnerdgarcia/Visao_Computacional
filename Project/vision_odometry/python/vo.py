"""Headless visual odometry. OpenCV camera axes: x right, y down, z forward.
R_wc maps camera vectors to the segment world; C_w is the camera centre.
No IMU, reference trajectory or ground truth enters this estimator.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
import cv2 as cv
import numpy as np

MIN_POINTS = 20


def compose(R_wc, C_w, R_rel, t_rel, scale=1.0):
    """X_current = R_rel X_previous + t_rel. scale only at mono bootstrap."""
    R_new = R_wc @ R_rel.T
    return R_new, C_w - scale * (R_new @ np.asarray(t_rel).reshape(3))


def camera_pose(R_cw, t_cw):
    """PnP with WORLD landmarks returns world->camera, not camera motion."""
    return R_cw.T, -R_cw.T @ np.asarray(t_cw).reshape(3)


def median(x, default=float('nan')):
    return float(np.median(x)) if len(x) else default


def coverage(p, size):
    if len(p) < MIN_POINTS:
        return False
    p = np.asarray(p)
    w, h = size
    cells = set(zip(np.clip((4*p[:, 0]/w).astype(int), 0, 3),
                    np.clip((3*p[:, 1]/h).astype(int), 0, 2)))
    span = np.ptp(p, axis=0) / [w, h]
    return len(cells) >= 6 and min(span) >= .2


class Calibration:
    def __init__(self, obj, mode):
        self.obj = obj
        self.size = (int(obj['width']), int(obj['height']))
        if min(self.size) <= 0:
            raise ValueError('invalid calibration resolution')
        self.K0 = self.matrix('K_left', (3, 3))
        self.D0 = self.matrix('D_left').reshape(-1)
        self.check_intrinsics(self.K0, self.D0)
        self.K = self.K0.copy()
        self.maps = None
        self.R1 = np.eye(3)
        self.unit = 'relative' if mode == 'mono' else obj.get('translation_unit')
        if mode == 'stereo':
            if self.unit not in ('m', 'cm', 'mm', 'arbitrary'):
                raise ValueError('translation_unit must be m, cm, mm or arbitrary')
            K1 = self.matrix('K_right', (3, 3))
            D1 = self.matrix('D_right').reshape(-1)
            self.check_intrinsics(K1, D1)
            R = self.matrix('R_right_left', (3, 3))
            t = self.matrix('T_right_left').reshape(3)
            if (not np.allclose(R.T @ R, np.eye(3), atol=1e-5)
                    or abs(np.linalg.det(R)-1) > 1e-5 or np.linalg.norm(t) < 1e-8):
                raise ValueError('invalid stereo extrinsics')
            if obj.get('already_rectified', False):
                if (np.linalg.norm(self.D0) > 1e-10 or np.linalg.norm(D1) > 1e-10
                        or not np.allclose(R, np.eye(3), atol=1e-8)
                        or np.linalg.norm(t[1:]) > 1e-8 or t[0] >= 0
                        or not np.allclose(self.K0, K1)):
                    raise ValueError('already_rectified requires common K, zero D, R=I, T=(-b,0,0)')
                self.P0 = self.K @ np.column_stack((np.eye(3), np.zeros(3)))
                self.P1 = K1 @ np.column_stack((R, t))
            else:
                self.R1, R2, self.P0, self.P1, _, _, _ = cv.stereoRectify(
                    self.K0, self.D0, K1, D1, self.size, R, t,
                    flags=cv.CALIB_ZERO_DISPARITY, alpha=0)
                if abs(self.P1[1, 3]) > 1e-7 or self.P1[0, 3] >= 0:
                    raise ValueError('only horizontal stereo with right camera at +x is supported')
                self.K = self.P0[:, :3]
                self.maps = (cv.initUndistortRectifyMap(self.K0, self.D0, self.R1, self.K, self.size, cv.CV_32FC1),
                             cv.initUndistortRectifyMap(K1, D1, R2, self.P1[:, :3], self.size, cv.CV_32FC1))
        elif mode != 'mono':
            raise ValueError('mode must be mono or stereo')
        elif np.linalg.norm(self.D0):
            self.maps = (cv.initUndistortRectifyMap(self.K0, self.D0, np.eye(3), self.K, self.size, cv.CV_32FC1),)

    def matrix(self, name, shape=None):
        a = np.asarray(self.obj[name], dtype=np.float64)
        if not np.isfinite(a).all() or (shape is not None and a.shape != shape):
            raise ValueError(f'invalid {name}')
        return a

    @staticmethod
    def check_intrinsics(K, D):
        if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]) or len(D) not in (4, 5, 8, 12, 14):
            raise ValueError('invalid intrinsics/distortion')

    def prepare(self, im, side=0):
        if im is None or im.size == 0 or im.shape[1::-1] != self.size:
            raise ValueError('image missing or resolution incompatible with calibration')
        if im.ndim == 3:
            im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        return cv.remap(im, *self.maps[side], cv.INTER_LINEAR) if self.maps else im


@dataclass
class Features:
    points: np.ndarray
    descriptors: np.ndarray | None


def detect(im):
    kp, d = cv.SIFT_create(nfeatures=2400, contrastThreshold=.025).detectAndCompute(im, None)
    return Features(np.asarray([k.pt for k in kp], dtype=np.float64).reshape(-1, 2), d)


def match(a, b):
    """Lowe ratio + mutual nearest neighbour, no image displacement ceiling."""
    if a.descriptors is None or b.descriptors is None or min(len(a.descriptors), len(b.descriptors)) < 2:
        return np.empty((0, 2), dtype=int)
    bf = cv.BFMatcher(cv.NORM_L2)
    forward = bf.knnMatch(a.descriptors, b.descriptors, k=2)
    reverse = bf.match(b.descriptors, a.descriptors)
    back = {m.queryIdx: m.trainIdx for m in reverse}
    pairs = [(m[0].queryIdx, m[0].trainIdx) for m in forward
             if len(m) >= 2 and m[0].distance < .75*m[1].distance
             and back.get(m[0].trainIdx) == m[0].queryIdx]
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


def rays(p, K):
    return (np.linalg.inv(K) @ np.column_stack((p, np.ones(len(p)))).T).T


def compensated_parallax(a, b, K, R):
    q = rays(a, K) @ R.T
    projected = q[:, :2] / q[:, 2:3]
    return np.linalg.norm((projected - rays(b, K)[:, :2]) * [K[0, 0], K[1, 1]], axis=1)


def rotation_only(a, b, K):
    """Robust rotation-only homography; translation/planarity is checked below."""
    if len(a) < MIN_POINTS:
        return False
    H, mask = cv.findHomography(a, b, cv.RANSAC, 1.5)
    if H is None or mask is None or np.mean(mask != 0) < .8:
        return False
    A = np.linalg.inv(K) @ H @ K
    if np.linalg.det(A) < 0:
        A = -A
    u, _, vt = np.linalg.svd(A)
    R = u @ vt
    return np.linalg.det(R) > 0 and median(compensated_parallax(a, b, K, R)) < .7


def triangulate(a, b, K, R, t, min_parallax=.7):
    """Points in first camera. Validate cheirality, reprojection and parallax."""
    if not len(a):
        return np.empty((0, 3)), np.zeros(0, bool), np.empty(0)
    P = K @ np.column_stack((np.eye(3), np.zeros(3)))
    Q = K @ np.column_stack((R, np.asarray(t).reshape(3)))
    h = cv.triangulatePoints(P, Q, np.asarray(a).T, np.asarray(b).T).T
    goodw = np.abs(h[:, 3]) > 1e-10
    X = h[:, :3] / np.where(goodw, h[:, 3], np.nan)[:, None]
    Y = X @ R.T + np.asarray(t).reshape(3)
    p = X @ K.T
    q = Y @ K.T
    with np.errstate(invalid='ignore', divide='ignore'):
        e = np.maximum(np.linalg.norm(p[:, :2]/p[:, 2:] - a, axis=1),
                       np.linalg.norm(q[:, :2]/q[:, 2:] - b, axis=1))
    good = (goodw & np.isfinite(X).all(axis=1) & (X[:, 2] > 0) & (Y[:, 2] > 0)
            & (e < 2.) & (compensated_parallax(a, b, K, R) >= min_parallax))
    return X, good, e


@dataclass
class Estimate:
    accepted: bool = False
    state: str = 'INSUFFICIENT'
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mask: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    X: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    inliers: int = 0
    ratio: float = 0.
    error: float = float('nan')
    parallax: float = float('nan')
    cheirality: float = float('nan')


def bootstrap(a, b, K, size):
    z = Estimate(mask=np.zeros(len(a), bool))
    if len(a) < MIN_POINTS or not coverage(a, size) or not coverage(b, size):
        return z
    if median(np.linalg.norm(b-a, axis=1)) < .4:
        z.state = 'STATIONARY'
        return z
    if rotation_only(a, b, K):
        z.state = 'ROTATION_ONLY'
        return z
    E, ransac = cv.findEssentialMat(a, b, K, method=cv.RANSAC, prob=.999, threshold=1.)
    if E is None or E.shape != (3, 3) or ransac is None:
        z.state = 'ESSENTIAL_REJECTED'
        return z
    n_ransac = int(np.count_nonzero(ransac))
    if n_ransac < MIN_POINTS or n_ransac / len(a) < .45:
        z.state = 'RANSAC_REJECTED'
        return z
    # recoverPose consumes AND updates the RANSAC mask (cheirality + depth).
    n, R, t, posemask = cv.recoverPose(E, a, b, K, mask=ransac.copy())
    z.cheirality = n / n_ransac
    X, valid, errors = triangulate(a, b, K, R, t)
    valid &= (ransac.ravel() != 0) & (posemask.ravel() != 0)
    z.mask = valid
    z.inliers = int(valid.sum())
    z.ratio = z.inliers / len(a)
    z.parallax = median(compensated_parallax(a[valid], b[valid], K, R))
    z.error = median(errors[valid])
    z.state = 'DEGENERATE'
    if (z.inliers < MIN_POINTS or z.ratio < .45 or z.cheirality < .7
            or not coverage(a[valid], size) or not coverage(b[valid], size)
            or z.parallax < 1.):
        return z
    # Planar scenes do not provide a reliable two-view map initialization here.
    _, hm = cv.findHomography(a, b, cv.RANSAC, 1.5)
    if hm is not None and np.mean(hm != 0) > .9:
        z.state = 'PLANAR'
        return z
    s = np.linalg.svd(X[valid] - X[valid].mean(0), compute_uv=False)
    if s[1] / s[0] < .02 or s[2] / s[0] < .005:
        return z
    z.accepted, z.state, z.R, z.t, z.X = True, 'INITIALIZED', R, t.ravel(), X
    return z


def pnp(X, p, K, size):
    z = Estimate(mask=np.zeros(len(p), bool))
    if len(p) < MIN_POINTS or not coverage(p, size) or not np.isfinite(X).all():
        return z
    s = np.linalg.svd(X - X.mean(0), compute_uv=False)
    if s[0] <= 0 or s[1]/s[0] < .01:
        z.state = 'DEGENERATE'
        return z
    ok, rv, tv, idx = cv.solvePnPRansac(np.ascontiguousarray(X), np.ascontiguousarray(p), K, None,
                                     iterationsCount=200, reprojectionError=2., confidence=.999,
                                     flags=cv.SOLVEPNP_EPNP)
    if not ok or idx is None or len(idx) < MIN_POINTS or len(idx)/len(p) < .45:
        z.state = 'PNP_REJECTED'
        return z
    ids = idx.ravel()
    rv, tv = cv.solvePnPRefineLM(X[ids], p[ids], K, None, rv, tv)
    R, _ = cv.Rodrigues(rv)
    q, _ = cv.projectPoints(X, rv, tv, K, None)
    e = np.linalg.norm(q.reshape(-1, 2)-p, axis=1)
    depth = (X @ R.T + tv.ravel())[:, 2]
    # Never add points rejected by RANSAC after refinement.
    z.mask[ids] = (e[ids] < 2.) & (depth[ids] > 0)
    z.inliers = int(z.mask.sum())
    z.ratio = z.inliers/len(p)
    z.error = median(e[z.mask])
    z.cheirality = float(np.mean(depth[ids] > 0))
    if (z.inliers < MIN_POINTS or z.ratio < .45 or z.error > 1.5
            or z.cheirality < .9 or not coverage(p[z.mask], size)):
        z.state = 'PNP_REJECTED'
        return z
    z.accepted, z.state, z.R, z.t = True, 'TRACKING', R, tv.ravel()
    return z


def stereo_map(left, right, cal):
    pairs = match(left, right)
    if len(pairs) < MIN_POINTS:
        return {}, float('nan')
    a, b = left.points[pairs[:, 0]], right.points[pairs[:, 1]]
    ep = np.abs(a[:, 1]-b[:, 1])
    epi = median(ep)
    # Reject a globally inconsistent pair before retaining convenient matches.
    if epi > 1.5 or np.mean(ep < 2.) < .6:
        return {}, epi
    disp = a[:, 0]-b[:, 0]
    good = (ep < 1.5) & (disp > .5)
    ids = np.flatnonzero(good)
    baseline_t = np.array([cal.P1[0, 3]/cal.K[0, 0], 0., 0.])
    X, valid, _ = triangulate(a[ids], b[ids], cal.K, np.eye(3), baseline_t, .5)
    return {int(pairs[ids[j], 0]): X[j] for j in np.flatnonzero(valid)}, epi


class Odometry:
    def __init__(self, cal, mode):
        self.cal, self.mode = cal, mode
        self.segment = 0
        self.R = np.eye(3)
        self.C = np.zeros(3)
        self.ref = None
        self.landmarks = {}
        self.refR, self.refC = self.R.copy(), self.C.copy()
        self.failures = 0
        self.age = 0
        self.has_pose = False
        self.epipolar = float('nan')

    def set_ref(self, f, landmarks):
        self.ref, self.landmarks = f, landmarks
        self.refR, self.refC = self.R.copy(), self.C.copy()
        self.age = 0

    def reject(self, state, f, z):
        if state not in ('STATIONARY', 'ROTATION_ONLY', 'PLANAR', 'DEGENERATE'):
            self.failures += 1
        if self.failures >= 3:
            # New origin and scale; never connect this to the previous map.
            self.segment += 1
            self.R, self.C = np.eye(3), np.zeros(3)
            self.has_pose = False
            self.ref, self.landmarks = None, {}
            self.failures = 0
            z.state = 'REINITIALIZING'
        else:
            z.state = state
        return z

    def step(self, im, right=None):
        f = detect(self.cal.prepare(im))
        self.epipolar = float('nan')
        current_stereo = {}
        if self.mode == 'stereo':
            if right is None:
                raise ValueError('stereo requires a synchronized right image')
            g = detect(self.cal.prepare(right, 1))
            current_stereo, self.epipolar = stereo_map(f, g, self.cal)
            if len(current_stereo) < MIN_POINTS:
                return self.reject('STEREO_REJECTED', f, Estimate())
        if len(f.points) < MIN_POINTS:
            return self.reject('NO_FEATURES', f, Estimate())
        if self.ref is None:
            self.set_ref(f, current_stereo)
            self.has_pose = self.mode == 'stereo'
            return Estimate(state='ANCHOR', accepted=self.has_pose)
        pairs = match(self.ref, f)
        self.age += 1
        if len(pairs) < MIN_POINTS:
            return self.reject('LOST', f, Estimate())
        a, b = self.ref.points[pairs[:, 0]], f.points[pairs[:, 1]]
        if not self.landmarks:
            z = bootstrap(a, b, self.cal.K, self.cal.size)
            if not z.accepted:
                return self.reject(z.state, f, z)
            # One arbitrary baseline ONLY establishes the initial map gauge.
            self.R, self.C = compose(self.refR, self.refC, z.R, z.t)
            world = z.X @ self.refR.T + self.refC
            self.set_ref(f, {int(pairs[j, 1]): world[j] for j in np.flatnonzero(z.mask)})
            self.has_pose = True
        else:
            ids = [j for j, (i, _) in enumerate(pairs) if int(i) in self.landmarks]
            X = np.array([self.landmarks[int(pairs[j, 0])] for j in ids]).reshape(-1, 3)
            p = b[ids]
            z = pnp(X, p, self.cal.K, self.cal.size)
            if not z.accepted:
                return self.reject(z.state, f, z)
            newR, newC = camera_pose(z.R, z.t)
            if not np.isfinite(newC).all():
                return self.reject('NONFINITE', f, Estimate())
            self.R, self.C = newR, newC
            if self.mode == 'stereo':
                self.set_ref(f, {i: self.R @ x + self.C for i, x in current_stereo.items()})
            elif self.age >= 8 or z.inliers < .6*len(self.landmarks):
                # Propagate known landmarks and triangulate new points using the
                # SAME map baseline estimated by PnP (no new unit translations).
                keep = {int(pairs[ids[j], 1]): X[j] for j in np.flatnonzero(z.mask)}
                relR = self.R.T @ self.refR
                relt = self.R.T @ (self.refC - self.C)
                pts, valid, _ = triangulate(a, b, self.cal.K, relR, relt, .7)
                for j in np.flatnonzero(valid):
                    keep.setdefault(int(pairs[j, 1]), self.refR @ pts[j] + self.refC)
                if len(keep) >= MIN_POINTS:
                    self.set_ref(f, keep)
        self.failures = 0
        return z


def manifest_rows(path, mode, tolerance=.002):
    root = Path(path).resolve().parent
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('empty manifest')
    last_t = -float('inf')
    last_i = -float('inf')
    used_l, used_r = set(), set()
    for row in rows:
        i, t = int(row['index']), float(row['timestamp'])
        if not np.isfinite(t) or t <= last_t or i <= last_i:
            raise ValueError('manifest indices/timestamps must be finite and strictly increasing')
        last_i, last_t = i, t
        left = root / row['left']
        if not left.is_file() or str(left.resolve()) in used_l:
            raise ValueError('left image absent or reused')
        used_l.add(str(left.resolve()))
        right = None
        if mode == 'stereo':
            if not row.get('right') or not row.get('right_timestamp'):
                raise ValueError('missing stereo pair/timestamp')
            rt = float(row['right_timestamp'])
            right = root / row['right']
            if not np.isfinite(rt) or abs(rt-t) > tolerance:
                raise ValueError('stereo pair out of synchronization tolerance')
            if not right.is_file() or right.resolve() == left.resolve() or str(right.resolve()) in used_r:
                raise ValueError('right image absent, duplicated or reused')
            used_r.add(str(right.resolve()))
        yield i, t, left, right


FIELDS = ['index', 'timestamp', 'segment', 'state', 'accepted', 'pose_valid', 'unit',
          'x', 'y', 'z'] + [f'r{r}{c}' for r in range(3) for c in range(3)] + [
          'inliers', 'inlier_ratio', 'reprojection_px', 'parallax_px', 'cheirality_ratio', 'epipolar_px', 'map_points']


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=['mono', 'stereo'], required=True)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--manifest')
    source.add_argument('--video')
    ap.add_argument('--calibration', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--sync-tolerance', type=float, default=.002)
    args = ap.parse_args(argv)
    if args.stride < 1 or args.max_frames < 0 or args.sync_tolerance < 0:
        ap.error('invalid stride/frame limit/tolerance')
    if args.video and args.mode == 'stereo':
        ap.error('stereo requires a real pair manifest; video is mono only')
    cal = Calibration(json.loads(Path(args.calibration).read_text()), args.mode)
    engine = Odometry(cal, args.mode)
    cv.setRNGSeed(42)
    cv.setNumThreads(1)
    cap = None
    if args.video:
        cap = cv.VideoCapture(args.video)
        fps = cap.get(cv.CAP_PROP_FPS)
        if not cap.isOpened() or fps <= 0:
            raise ValueError('cannot open video or invalid FPS')
        def video_rows():
            i = 0
            while True:
                ok, im = cap.read()
                if not ok:
                    break
                yield i, i/fps, im, None
                i += 1
        rows = video_rows()
    else:
        # Validate the complete manifest before writing output.
        rows = list(manifest_rows(args.manifest, args.mode, args.sync_tolerance))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(args.output, 'w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS)
        writer.writeheader()
        for k, (i, ts, left, right) in enumerate(rows):
            if k % args.stride:
                continue
            if args.max_frames and count >= args.max_frames:
                break
            if not args.video:
                left = cv.imread(str(left), cv.IMREAD_GRAYSCALE)
                right = cv.imread(str(right), cv.IMREAD_GRAYSCALE) if right else None
            z = engine.step(left, right)
            values = [i, format(ts, '.9f'), engine.segment, z.state, int(z.accepted),
                      int(engine.has_pose), cal.unit, *engine.C, *engine.R.ravel(),
                      z.inliers, z.ratio, z.error, z.parallax, z.cheirality,
                      engine.epipolar, len(engine.landmarks)]
            writer.writerow(dict(zip(FIELDS, values)))
            count += 1
    if cap:
        cap.release()
    if not count:
        raise ValueError('no frames processed')
    print(f'{count} frames; {engine.segment+1} segments; CSV: {args.output}')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError, cv.error) as e:
        print(f'error: {e}', file=sys.stderr)
        sys.exit(2)
