import numpy as np

def chaikin_smooth(pts, iterations=3):
    pts = np.array(pts, dtype=np.float32)
    if len(pts) < 3: return pts
        
    for _ in range(iterations):
        p0 = pts[:-1]
        p1 = pts[1:]
        q1 = 0.75 * p0 + 0.25 * p1
        q2 = 0.25 * p0 + 0.75 * p1
        
        new_pts = np.empty((len(pts) * 2 - 2, 2), dtype=np.float32)
        new_pts[0::2] = q1
        new_pts[1::2] = q2
        
        new_pts[0] = pts[0]
        new_pts[-1] = pts[-1]
        
        pts = new_pts
        
    return pts

def point_line_distance(px, py, p1x, p1y, p2x, p2y):
    l2 = (p2x - p1x)**2 + (p2y - p1y)**2
    if l2 == 0:
        return (px - p1x)**2 + (py - p1y)**2
    t = max(0, min(1, ((px - p1x) * (p2x - p1x) + (py - p1y) * (p2y - p1y)) / l2))
    proj_x = p1x + t * (p2x - p1x)
    proj_y = p1y + t * (p2y - p1y)
    return (px - proj_x)**2 + (py - proj_y)**2
