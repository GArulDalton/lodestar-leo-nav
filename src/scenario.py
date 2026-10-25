import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle, FancyBboxPatch
plt.rcParams.update({"figure.dpi":150,"savefig.bbox":"tight"})

fig,ax = plt.subplots(figsize=(9,5.2)); ax.axis("off")
ax.set_xlim(0,10); ax.set_ylim(0,6.5)

# sky band
ax.add_patch(Rectangle((0,3.4),10,3.1,fc="#eaf2fb",ec="none"))
ax.add_patch(Rectangle((0,0),10,1.1,fc="#efe9df",ec="none"))  # ground

# LEO satellites
sat_xy = [(1.6,5.3),(4.2,5.7),(6.8,5.4)]
for i,(sx,sy) in enumerate(sat_xy):
    ax.add_patch(FancyBboxPatch((sx-0.28,sy-0.16),0.56,0.32,
        boxstyle="round,pad=0.02",fc="#ffd27f",ec="black",lw=1))
    ax.plot([sx-0.45,sx-0.3],[sy,sy],color="black",lw=3)
    ax.plot([sx+0.3,sx+0.45],[sy,sy],color="black",lw=3)
    ax.text(sx,sy+0.33,f"LEO {i+1}",ha="center",fontsize=8)
ax.annotate("", xy=(4.9,5.6), xytext=(3.6,5.6),
            arrowprops=dict(arrowstyle="->",color="#888"))
ax.text(4.25,5.78,"7.5 km/s",fontsize=7,color="#888",ha="center")

# vehicle
vx,vy = 4.8,2.2
ax.add_patch(FancyBboxPatch((vx-0.55,vy-0.28),1.1,0.56,
    boxstyle="round,pad=0.02",fc="#cfe8ff",ec="black",lw=1.4))
ax.text(vx,vy,"Aerospace\nvehicle",ha="center",va="center",fontsize=8.5)
ax.text(vx,vy-0.6,"VIO + EKF + integrity",ha="center",fontsize=7.5,color="#1f4e79")

# clean Doppler downlinks (green)
for sx,sy in sat_xy:
    ax.add_patch(FancyArrowPatch((sx,sy-0.18),(vx,vy+0.3),
        arrowstyle="-|>",mutation_scale=10,color="#2ca02c",lw=1.1,alpha=0.8))
ax.text(2.3,3.9,"opportunistic\nDoppler",fontsize=7.5,color="#2ca02c")

# GNSS denied (red X at top right)
ax.add_patch(Circle((9.0,5.6),0.33,fc="#f4cccc",ec="#cc0000",lw=1.3))
ax.text(9.0,5.6,"GNSS",ha="center",va="center",fontsize=6.5,color="#cc0000")
ax.plot([8.72,9.28],[5.35,5.85],color="#cc0000",lw=2)
ax.plot([8.72,9.28],[5.85,5.35],color="#cc0000",lw=2)
ax.text(9.0,5.08,"jammed",fontsize=7,color="#cc0000",ha="center")

# spoofer on ground (red dashed)
spx,spy = 7.4,0.55
ax.add_patch(FancyBboxPatch((spx-0.5,spy-0.2),1.0,0.42,
    boxstyle="round,pad=0.02",fc="#f4cccc",ec="#cc0000",lw=1.2))
ax.text(spx,spy,"Spoofer",ha="center",va="center",fontsize=8,color="#990000")
ax.add_patch(FancyArrowPatch((spx,spy+0.22),(vx+0.3,vy-0.3),
    arrowstyle="-|>",mutation_scale=10,color="#cc0000",lw=1.3,ls="--"))
ax.text(6.0,1.15,"false Doppler\n(rejected by integrity module)",fontsize=7.5,color="#cc0000")

ax.text(5.0,6.3,"GNSS-denied operation with LEO signals of opportunity",
        ha="center",fontsize=11,weight="bold")
fig.savefig("fig_scenario.png"); plt.close(fig)
print("scenario diagram written")
