"""LODESTAR: GPS-denied autonomous navigation via opportunistic LEO Doppler
+ visual-inertial fusion with a learned integrity module. Pure CPU, reproducible.
See README.md for the method and results.
"""
import numpy as np
import json, time, os
from dataclasses import dataclass

RNG = np.random.default_rng(20261024)
OUT = os.path.dirname(os.path.abspath(__file__))

DT = 0.1
T_END = 300.0
N = int(T_END / DT)
LEO_SIG = 0.15

def ground_truth():
    t = np.arange(N) * DT
    x = 400 * np.sin(2 * np.pi * t / 120.0)
    y = 300 * np.sin(4 * np.pi * t / 120.0)
    z = 80 + 20 * np.sin(2 * np.pi * t / 240.0)
    pos = np.stack([x, y, z], axis=1)
    vel = np.gradient(pos, DT, axis=0)
    return t, pos, vel

@dataclass
class LeoSat:
    p0: np.ndarray
    v: np.ndarray
    def state(self, t): return self.p0 + self.v * t

def build_constellation(n=6):
    sats = []
    for _ in range(n):
        az = RNG.uniform(0, 2*np.pi)
        el = RNG.uniform(np.radians(25), np.radians(80))
        r = 1.1e6
        p0 = r*np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
        heading = RNG.uniform(0, 2*np.pi); speed = 7500.0
        v = speed*np.array([np.cos(heading), np.sin(heading), RNG.uniform(-0.2,0.2)])
        sats.append(LeoSat(p0, v))
    return sats

def leo_geometry(t_arr, pos, vel, sats):
    M = len(sats)
    los = np.zeros((N, M, 3)); rr = np.zeros((N, M))
    sat_term = np.zeros((N, M)); vis = np.zeros((N, M), bool)
    for j, s in enumerate(sats):
        sp = np.array([s.state(t) for t in t_arr]); sv = np.tile(s.v, (N,1))
        d = sp - pos; rng = np.linalg.norm(d, axis=1); u = d/rng[:,None]
        los[:, j] = u
        rr[:, j] = np.sum((sv - vel)*u, axis=1)
        sat_term[:, j] = np.sum(sv*u, axis=1)
        vis[:, j] = np.degrees(np.arcsin(np.clip(u[:,2],-1,1))) > 15.0
    return los, rr, sat_term, vis

def synth(vel, los, rr, sat_term, vis):
    vio_bias = np.cumsum(RNG.normal(0, 0.0045, size=(N,3)), axis=0)
    vio_vel = vel + vio_bias + RNG.normal(0, 0.05, size=(N,3))
    clk = np.cumsum(RNG.normal(0, 0.02, size=N))
    leo_rr = rr + clk[:,None] + RNG.normal(0, LEO_SIG, size=rr.shape)
    z = leo_rr - sat_term
    spoof_mask = np.zeros(rr.shape, bool)
    for (a, b) in [(800, 1400), (2100, 2500)]:
        for j in RNG.choice(rr.shape[1], size=2, replace=False):
            spoof_mask[a:b, j] = True
            z[a:b, j] += RNG.uniform(3.0, 6.0)*RNG.choice([-1,1])
    return vio_vel, z, clk, spoof_mask

class IntegrityNet:
    def __init__(self, din=3, h=8):
        self.W1 = RNG.normal(0,0.5,(din,h)); self.b1 = np.zeros(h)
        self.W2 = RNG.normal(0,0.5,(h,1));   self.b2 = np.zeros(1)
        self.mu = np.zeros(din); self.sd = np.ones(din)
    def _f(self, X):
        a1 = np.tanh(X@self.W1+self.b1)
        p = 1/(1+np.exp(-(a1@self.W2+self.b2)))
        return p, a1
    def predict(self, X):
        X = (X-self.mu)/self.sd
        return self._f(X)[0].ravel()
    def fit(self, X, y, epochs=600, lr=0.2):
        self.mu = X.mean(0); self.sd = X.std(0)+1e-6
        X = (X-self.mu)/self.sd; y = y.reshape(-1,1)
        for _ in range(epochs):
            p, a1 = self._f(X); g = (p-y)/len(X)
            dW2 = a1.T@g; db2 = g.sum(0)
            da1 = (g@self.W2.T)*(1-a1**2)
            dW1 = X.T@da1; db1 = da1.sum(0)
            self.W2-=lr*dW2; self.b2-=lr*db2; self.W1-=lr*dW1; self.b1-=lr*db1

def feat(innov, el, nsat, S):
    n = innov/(np.sqrt(S)+1e-6)
    return np.array([n, abs(n), el])

def calibrate_net():
    X, y = [], []
    for _ in range(6000):
        el = RNG.uniform(0.26, 1.4); nsat = RNG.integers(4,7)
        S = RNG.uniform(0.02, 0.30)
        if RNG.random() < 0.5:
            innov = RNG.normal(0, np.sqrt(S)); lab = 0
        else:
            innov = RNG.uniform(2.5,6.0)*RNG.choice([-1,1]) + RNG.normal(0, np.sqrt(S)); lab = 1
        X.append(feat(innov, el, nsat, S)); y.append(lab)
    net = IntegrityNet(); net.fit(np.array(X), np.array(y))
    return net

def est_vio_only(vio_vel, p0):
    pos = np.zeros((N,3)); p = p0.copy()
    for k in range(N):
        p = p + vio_vel[k]*DT; pos[k] = p
    return pos

def est_loosely(vio_vel, z, los, vis, p0):
    pos = np.zeros((N,3)); p = p0.copy()
    for k in range(N):
        v = vio_vel[k].copy(); idx = np.where(vis[k])[0]
        if len(idx) >= 4:
            A = np.hstack([-los[k,idx], np.ones((len(idx),1))])
            sol,*_ = np.linalg.lstsq(A, z[k,idx], rcond=None)
            v = 0.5*v + 0.5*sol[:3]
        p = p + v*DT; pos[k] = p
    return pos

def est_lodestar(vio_vel, z, los, vis, net, p0):
    pos = np.zeros((N,3)); p = p0.copy()
    v = vio_vel[0].copy(); clk = 0.0
    P = np.eye(4)*0.1
    Q = np.diag([0.03,0.03,0.03,0.02])*DT
    Rm = LEO_SIG**2
    acc = np.zeros(z.shape, bool); rt = np.zeros(N)
    for k in range(N):
        t0 = time.perf_counter()
        v = 0.85*v + 0.15*vio_vel[k]
        P = P + Q
        idx = np.where(vis[k])[0]
        for j in idx:
            u = los[k,j]
            H = np.array([-u[0],-u[1],-u[2],1.0])
            h = H @ np.array([v[0],v[1],v[2],clk])
            innov = z[k,j] - h
            S = H@P@H + Rm
            el = np.arcsin(np.clip(u[2],-1,1))
            if k < 50:
                gate = 1.0
            else:
                gate = 1.0 - net.predict(feat(innov, el, len(idx), S))[0]
            if gate > 0.5:
                K = (P@H)/S
                dx = K*innov
                v = v + dx[:3]; clk += dx[3]
                P = P - np.outer(K,H)@P
                acc[k,j] = True
        p = p + v*DT; pos[k] = p
        rt[k] = (time.perf_counter()-t0)*1e3
    return pos, acc, rt

def ate(e,g): return np.sqrt(np.mean(np.sum((e-g)**2,axis=1)))
def fdrift(e,g): return np.linalg.norm(e[-1]-g[-1])
def spoof_prf(acc, spoof, vis):
    sv = spoof & vis; cv = (~spoof) & vis; rej = (~acc) & vis
    tp = np.sum(rej&sv); fp = np.sum(rej&cv); fn = np.sum((~rej)&sv)
    pr = tp/(tp+fp+1e-9); rc = tp/(tp+fn+1e-9); f1 = 2*pr*rc/(pr+rc+1e-9)
    return dict(precision=pr, recall=rc, f1=f1, tp=int(tp), fp=int(fp), fn=int(fn))

def main():
    t, pos, vel = ground_truth()
    sats = build_constellation(6)
    los, rr, sat_term, vis = leo_geometry(t, pos, vel, sats)
    vio_vel, z, clk, spoof = synth(vel, los, rr, sat_term, vis)
    net = calibrate_net()
    e_vio = est_vio_only(vio_vel, pos[0])
    e_loose = est_loosely(vio_vel, z, los, vis, pos[0])
    e_lode, acc, rt = est_lodestar(vio_vel, z, los, vis, net, pos[0])
    class AllClean:
        def predict(self, X): return np.zeros(len(X))
    e_noint, acc2, _ = est_lodestar(vio_vel, z, los, vis, AllClean(), pos[0])
    res = {
        "ATE_m": {"VIO_only": round(ate(e_vio,pos),3),
            "Loosely_coupled": round(ate(e_loose,pos),3),
            "LODESTAR_no_integrity": round(ate(e_noint,pos),3),
            "LODESTAR": round(ate(e_lode,pos),3)},
        "final_drift_m": {"VIO_only": round(fdrift(e_vio,pos),3),
            "Loosely_coupled": round(fdrift(e_loose,pos),3),
            "LODESTAR_no_integrity": round(fdrift(e_noint,pos),3),
            "LODESTAR": round(fdrift(e_lode,pos),3)},
        "spoof_detection_LODESTAR": {k:(round(v,4) if isinstance(v,float) else v)
                                     for k,v in spoof_prf(acc,spoof,vis).items()},
        "runtime_ms_per_update": {"mean": round(float(np.mean(rt)),4),
            "p95": round(float(np.percentile(rt,95)),4),
            "max": round(float(np.max(rt)),4)},
        "config": {"steps":N,"rate_Hz":1/DT,"duration_s":T_END,"n_leo":6,
                   "spoof_windows_idx":[[800,1400],[2100,2500]]},
    }
    with open(os.path.join(OUT,"results.json"),"w") as f: json.dump(res,f,indent=2)
    np.savez(os.path.join(OUT,"traj.npz"), gt=pos, vio=e_vio, loose=e_loose,
             lode=e_lode, noint=e_noint, t=t, acc=acc, spoof=spoof, vis=vis, rt=rt)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
