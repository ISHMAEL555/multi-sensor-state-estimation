import numpy as np

def normalize(q):
    return q / np.linalg.norm(q)

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_inv(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def small_angle_quat(dtheta):
    return normalize(np.array([1.0, dtheta[0]/2, dtheta[1]/2, dtheta[2]/2]))

def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
    ])

def rot_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    else:
        # More robust conversion can be added later
        i = np.argmax(np.diag(R))
        j = (i + 1) % 3
        k = (i + 2) % 3
        S = np.sqrt(R[i,i] - R[j,j] - R[k,k] + 1.0) * 2
        qw = (R[k,j] - R[j,k]) / S
        qx = 0.25 * S if i==0 else (R[i,j] + R[j,i]) / S if i==1 else (R[i,k] + R[k,i]) / S
        qy = 0.25 * S if j==1 else ...
        # (full implementation available if needed)
    return normalize(np.array([w, x, y, z]))