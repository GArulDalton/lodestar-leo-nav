"""Multi-seed evaluation + robustness sweeps for LODESTAR."""
import numpy as np, json, importlib
import sim

def run_once(seed, spoof_amp=(3.0,6.0), leo_sig=0.15, n_leo=6):
    sim.RNG = np.random.default_rng(seed)
    sim.LEO_SIG = leo_sig
    t, pos, vel = sim.ground_truth()
    sats = sim.build_constellation(n_leo)
    los, rr, sat_term, vis = sim.leo_geometry(t, pos, vel, sats)
    # synth with custom spoof amplitude
    vio_bias = np.cumsum(sim.RNG.normal(0,0.0045,size=(sim.N,3)),axis=0)
    vio_vel = vel + vio_bias + sim.RNG.normal(0,0.05,size=(sim.N,3))
    clk = np.cumsum(sim.RNG.normal(0,0.02,size=sim.N))
    leo_rr = rr + clk[:,None] + sim.RNG.normal(0,leo_sig,size=rr.shape)
    z = leo_rr - sat_term
    spoof = np.zeros(rr.shape, bool)
    for (a,b) in [(800,1400),(2100,2500)]:
        for j in sim.RNG.choice(rr.shape[1],size=2,replace=False):
            spoof[a:b,j]=True
            z[a:b,j]+=sim.RNG.uniform(*spoof_amp)*sim.RNG.choice([-1,1])
    net = sim.calibrate_net()
    e_vio = sim.est_vio_only(vio_vel,pos[0])
    e_loose = sim.est_loosely(vio_vel,z,los,vis,pos[0])
    e_lode,acc,rt = sim.est_lodestar(vio_vel,z,los,vis,net,pos[0])
    class AC:
        def predict(self,X): return np.zeros(len(X))
    e_noint,_,_ = sim.est_lodestar(vio_vel,z,los,vis,AC(),pos[0])
    prf = sim.spoof_prf(acc,spoof,vis)
    return dict(
        ate_vio=sim.ate(e_vio,pos), ate_loose=sim.ate(e_loose,pos),
        ate_noint=sim.ate(e_noint,pos), ate_lode=sim.ate(e_lode,pos),
        fd_vio=sim.fdrift(e_vio,pos), fd_lode=sim.fdrift(e_lode,pos),
        prec=prf['precision'], rec=prf['recall'], f1=prf['f1'],
        rt_mean=float(np.mean(rt)), rt_p95=float(np.percentile(rt,95)))

# ---- 10 seeds ----
seeds = range(100,110)
rows = [run_once(s) for s in seeds]
def ms(key): 
    a=np.array([r[key] for r in rows]); return (float(a.mean()), float(a.std()))
summary = {k: {"mean": round(ms(k)[0],3), "std": round(ms(k)[1],3)}
           for k in rows[0].keys()}
print("=== 10-seed summary ===")
print(json.dumps(summary, indent=2))

# ---- spoof-amplitude sweep (single seed baseline) ----
sweep = []
for amp in [1.0,2.0,3.0,4.0,6.0,8.0]:
    r = run_once(100, spoof_amp=(amp,amp+0.5))
    sweep.append(dict(amp=amp, ate_lode=round(r['ate_lode'],2),
                      ate_noint=round(r['ate_noint'],2), f1=round(r['f1'],3)))
print("=== spoof amplitude sweep ===")
print(json.dumps(sweep, indent=2))

# ---- LEO count sweep ----
leo_sweep=[]
for n in [4,5,6,8,10]:
    r = run_once(100, n_leo=n)
    leo_sweep.append(dict(n_leo=n, ate_lode=round(r['ate_lode'],2), f1=round(r['f1'],3)))
print("=== LEO count sweep ===")
print(json.dumps(leo_sweep, indent=2))

json.dump(dict(summary=summary, spoof_sweep=sweep, leo_sweep=leo_sweep),
          open('experiments.json','w'), indent=2)
