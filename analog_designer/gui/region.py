"""Geometry for pro's design-space viewer's profile-region overlay: given
the 2-D points of variations matching a profile, produce a rounded 'blob'
outline enclosing them (convex hull + Chaikin corner-cutting). Pure
geometry, no Tk/plotting imports -- same separation as data.py. Core ships
this module (small, generic, reusable geometry) but has no viewer of its
own that calls it yet."""


def blob_region(points, padding_frac=0.08, chaikin_iterations=3):
    """points: list of (x, y). Returns a closed polygon (list of (x, y))
    enclosing them with a rounded outline, or None if there are fewer than
    3 distinct non-collinear points (nothing meaningful to enclose)."""
    pts = sorted(set(points))
    if len(pts) < 3:
        return None
    hull = _convex_hull(pts)
    if len(hull) < 3:
        return None
    hull = _pad_hull(hull, padding_frac)
    return _chaikin(hull, chaikin_iterations)


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _convex_hull(points):
    """Andrew's monotone chain. Returns hull vertices in CCW order, with
    collinear points on an edge dropped. Degenerates to <=2 points if the
    input is collinear or has fewer than 2 distinct points."""
    pts = sorted(points)
    if len(pts) < 2:
        return pts

    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def _pad_hull(hull, padding_frac):
    """Expand each vertex outward from the hull's centroid so points don't
    sit exactly on the drawn boundary."""
    cx = sum(x for x, _ in hull) / len(hull)
    cy = sum(y for _, y in hull) / len(hull)
    return [
        (cx + (x - cx) * (1 + padding_frac), cy + (y - cy) * (1 + padding_frac))
        for x, y in hull
    ]


def _chaikin(polygon, iterations):
    """Corner-cutting subdivision: rounds a polygon into an organic 'blob'
    outline without needing spline fitting."""
    pts = polygon
    for _ in range(iterations):
        new_pts = []
        n = len(pts)
        for i in range(n):
            p0, p1 = pts[i], pts[(i + 1) % n]
            new_pts.append((0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1]))
            new_pts.append((0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1]))
        pts = new_pts
    return pts
