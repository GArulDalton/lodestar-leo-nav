import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"figure.dpi":150,"savefig.bbox":"tight"})

fig,ax = plt.subplots(figsize=(9.5,5)); ax.axis("off")
ax.set_xlim(0,10); ax.set_ylim(0,6.2)
def box(x,y,w,h,txt,fc):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.03,rounding_size=0.12",
                 fc=fc,ec="black",lw=1.3))
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=9.5)
def arrow(x1,y1,x2,y2,color="black",style="-|>"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,
                 mutation_scale=14,lw=1.3,color=color))

box(0.2,4.5,2.2,1.0,"IMU + Camera\nVisual-Inertial\nOdometry","#cfe8ff")
box(0.2,0.6,2.2,1.0,"LEO downlink\nDoppler\n(pseudorange-rate)","#ffe0b3")
box(3.0,0.6,1.9,1.0,"Ephemeris\nde-rotation","#ffe0b3")
box(3.0,2.55,2.2,1.1,"Learned Integrity\nModule\n(spoof / multipath gate)","#e3d1ff")
box(6.0,2.4,2.2,1.5,"Tightly-coupled\nError-state EKF\n[velocity, clock]","#d6f5d6")
box(8.5,2.4,1.3,1.5,"Navigation\nstate\n(pose, vel)","#d6f5d6")

arrow(2.4,5.0,6.0,3.6)                 # VIO -> EKF (prediction)
ax.text(4.0,4.75,"state propagation",fontsize=8,rotation=-16,color="#1f4e79")
arrow(2.4,1.1,3.0,1.1)                 # LEO -> de-rotate
arrow(4.9,1.1,5.4,1.1); arrow(5.4,1.1,5.4,2.55); arrow(5.4,2.55,6.0,2.9)  # de-rot -> EKF
arrow(4.1,2.55,4.1,1.6)               # integrity gates de-rotated meas
ax.text(4.2,2.05,"gate",fontsize=8,color="#5b2d8e")
arrow(6.0,3.2,5.2,3.1)                # EKF innovation -> integrity (feedback)
ax.text(5.15,3.45,"innovation",fontsize=8,color="#555")
arrow(8.2,3.15,8.5,3.15)             # EKF -> state
arrow(9.15,2.4,9.15,1.55); arrow(9.15,1.55,0.6,1.55,color="#999",style="-|>")  # fb (faint)
ax.text(4.8,1.72,"feedback to VIO scale / bias",fontsize=7.5,color="#999")
ax.text(5.0,5.9,"LODESTAR architecture",ha="center",fontsize=12,weight="bold")
fig.savefig("fig_block.png"); plt.close(fig)
print("ok")
