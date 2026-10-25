import numpy as np, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":0.3,
                     "figure.dpi":150,"savefig.bbox":"tight"})
d = np.load("traj.npz")
gt,vio,loose,lode,noint = d["gt"],d["vio"],d["loose"],d["lode"],d["noint"]
t,acc,spoof,vis,rt = d["t"],d["acc"],d["spoof"],d["vis"],d["rt"]
exp = json.load(open("experiments.json"))

# Fig 1: trajectory (top-down XY)
fig,ax = plt.subplots(figsize=(6,5))
ax.plot(gt[:,0],gt[:,1],'k-',lw=2.2,label="Ground truth")
ax.plot(vio[:,0],vio[:,1],'--',color="#1f77b4",lw=1.3,label="VIO only")
ax.plot(lode[:,0],lode[:,1],'-',color="#d62728",lw=1.4,label="LODESTAR")
ax.scatter(*gt[0,:2],c="green",s=60,zorder=5,label="Start")
ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
ax.set_title("Estimated trajectories under GPS denial")
ax.legend(loc="upper right",fontsize=9); ax.set_aspect("equal","box")
fig.savefig("fig_traj.png"); plt.close(fig)

# Fig 2: position error over time, spoof windows shaded
def err(e): return np.linalg.norm(e-gt,axis=1)
fig,ax = plt.subplots(figsize=(7,4))
ax.plot(t,err(vio),color="#1f77b4",lw=1.2,label="VIO only")
ax.plot(t,err(noint),color="#7f7f7f",lw=1.2,label="LODESTAR w/o integrity")
ax.plot(t,err(lode),color="#d62728",lw=1.5,label="LODESTAR")
for (a,b) in [(800,1400),(2100,2500)]:
    ax.axvspan(a*0.1,b*0.1,color="orange",alpha=0.15)
ax.text(110,ax.get_ylim()[1]*0.9,"spoofing",color="darkorange",fontsize=9)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Position error (m)")
ax.set_title("Position error over the mission (shaded = LEO spoofing)")
ax.legend(fontsize=9)
fig.savefig("fig_error_time.png"); plt.close(fig)

# Fig 3: ATE bar chart with error bars (10-seed)
s = exp["summary"]
methods = ["VIO only","Loose coupling","No integrity","LODESTAR"]
keys = ["ate_vio","ate_loose","ate_noint","ate_lode"]
means = [s[k]["mean"] for k in keys]; stds=[s[k]["std"] for k in keys]
fig,ax = plt.subplots(figsize=(6,4))
cols=["#1f77b4","#9467bd","#7f7f7f","#d62728"]
ax.bar(methods,means,yerr=stds,capsize=4,color=cols,alpha=0.85)
ax.set_ylabel("ATE (m)"); ax.set_yscale("log")
ax.set_title("Absolute trajectory error (10 seeds, mean $\\pm$ std)")
plt.setp(ax.get_xticklabels(),rotation=15,ha="right")
fig.savefig("fig_ate_bar.png"); plt.close(fig)

# Fig 4: spoof amplitude sweep
sw = exp["spoof_sweep"]
amp=[r["amp"] for r in sw]
fig,ax = plt.subplots(figsize=(6,4))
ax.plot(amp,[r["ate_noint"] for r in sw],'o--',color="#7f7f7f",label="No integrity")
ax.plot(amp,[r["ate_lode"] for r in sw],'s-',color="#d62728",label="LODESTAR")
ax2=ax.twinx()
ax2.plot(amp,[r["f1"] for r in sw],'^:',color="#2ca02c",label="Detection F1")
ax2.set_ylabel("Detection F1",color="#2ca02c"); ax2.set_ylim(0,1.05); ax2.grid(False)
ax.set_xlabel("Spoofing bias magnitude (m/s)"); ax.set_ylabel("ATE (m)")
ax.set_title("Robustness to spoofing intensity")
ax.legend(loc="upper left",fontsize=9)
fig.savefig("fig_spoof_sweep.png"); plt.close(fig)

# Fig 5: LEO count sweep + runtime inset
ls = exp["leo_sweep"]
fig,ax = plt.subplots(figsize=(6,4))
ax.plot([r["n_leo"] for r in ls],[r["ate_lode"] for r in ls],'o-',color="#d62728")
ax.set_xlabel("Number of LEO satellites in view"); ax.set_ylabel("LODESTAR ATE (m)")
ax.set_title("Effect of LEO constellation size")
fig.savefig("fig_leo_sweep.png"); plt.close(fig)

# Fig 6: runtime histogram
fig,ax = plt.subplots(figsize=(6,3.6))
ax.hist(rt,bins=40,color="#17becf",alpha=0.85)
ax.axvline(100,color="red",ls="--",label="10 Hz real-time budget (100 ms)")
ax.axvline(np.mean(rt),color="black",ls=":",label=f"mean {np.mean(rt):.2f} ms")
ax.set_xlabel("Per-update compute time (ms)"); ax.set_ylabel("Count")
ax.set_title("Real-time per-update latency (CPU)")
ax.legend(fontsize=8); ax.set_xlim(0,1.5)
fig.savefig("fig_runtime.png"); plt.close(fig)

# Fig 7: system block diagram
fig,ax = plt.subplots(figsize=(9,4.6)); ax.axis("off")
ax.set_xlim(0,10); ax.set_ylim(0,6)
def box(x,y,w,h,txt,fc):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.03,rounding_size=0.12",
                 fc=fc,ec="black",lw=1.3))
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=9.5,wrap=True)
def arrow(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",
                 mutation_scale=14,lw=1.3,color="black"))
box(0.2,4.3,2.1,1.0,"IMU + Camera\n(Visual-Inertial\nOdometry)","#cfe8ff")
box(0.2,0.7,2.1,1.0,"LEO downlink\nDoppler\n(pseudorange-rate)","#ffe0b3")
box(3.0,0.7,2.0,1.0,"Ephemeris\nde-rotation","#ffe0b3")
box(3.0,2.5,2.0,1.1,"Learned Integrity\nModule\n(spoof/multipath\ngate)","#e3d1ff")
box(5.7,2.3,2.3,1.5,"Tightly-coupled\nError-state EKF\n[velocity, clock]","#d6f5d6")
box(8.3,2.3,1.5,1.5,"Navigation\nstate\n(pose, vel)","#d6f5d6")
arrow(2.3,4.8,6.0,3.8)                      # VIO -> EKF predict
arrow(2.3,1.2,3.0,1.2)                      # LEO -> de-rotate
arrow(5.0,1.2,6.0,2.5)                      # de-rotate -> EKF
arrow(4.0,2.5,4.0,1.7)                      # integrity reads residual (down)
arrow(5.0,3.05,5.7,3.05)                    # integrity -> EKF gate
arrow(6.85,2.9,6.85,2.0); arrow(6.0,1.6,4.7,2.5)  # EKF residual -> integrity (feedback)
arrow(8.0,3.05,8.3,3.05)                    # EKF -> state
ax.text(5.0,5.6,"LODESTAR architecture",ha="center",fontsize=12,weight="bold")
fig.savefig("fig_block.png"); plt.close(fig)

print("figures written:", [f for f in __import__('os').listdir('.') if f.endswith('.png')])
