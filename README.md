# LODESTAR

**Integrity Aware LEO Doppler Navigation for GNSS Denied Aerospace Vehicles**

G. Arul Dalton, Department of Computer Science and Engineering, Saveetha Engineering College, Chennai, Tamil Nadu, India.
Corresponding author: aruldalton@saveetha.ac.in · ORCID [0000-0002-5981-7176](https://orcid.org/0000-0002-5981-7176)

LODESTAR bounds the unbounded drift of visual-inertial odometry (VIO) in GNSS-denied
flight by fusing it with **opportunistic Doppler (pseudorange-rate) measurements from
proliferated LEO communication satellites**, inside a tightly-coupled error-state Kalman
filter. Because signals of opportunity are unauthenticated, a **lightweight learned
integrity module** screens every measurement from its filter innovation and rejects
spoofed or multipath-corrupted observations before they bias the state.

Prepared for submission to the *Journal of the Astronautical Sciences* (Springer).
This repository contains the full, reproducible, CPU-only experiment code.

## Method at a glance

```
IMU + Camera (VIO) ──state propagation──┐
                                        ▼
LEO downlink Doppler ─ ephemeris ─ de-rotation ─► Error-state EKF ─► nav state
                             │                        ▲   [v, clock]
                   Learned Integrity Module ──gate────┘
                   (reads filter innovation)
```

See the paper's Algorithm 1 for the per-step update. Key steps:
- **De-rotation.** `z = r − v_sat·u = −v_rx·u + clk + n` turns raw LEO range-rate into a
  linear velocity observation.
- **Estimator.** Tightly-coupled error-state EKF over `[vx, vy, vz, clk]`; satellites
  processed sequentially so the state is informed with even a single visible satellite.
- **Integrity.** A one-hidden-layer network maps `[innovation/√S, |innovation/√S|,
  elevation]` to a spoof probability; measurements are gated after a short burn-in.

## Results (10 random seeds, mean ± std)

| Method | ATE (m) | Final drift (m) |
|---|---|---|
| VIO only | 36.60 ± 12.11 | 73.20 ± 27.72 |
| Loose coupling | 614.24 ± 949.37 | — |
| LODESTAR w/o integrity | 269.78 ± 107.95 | — |
| **LODESTAR (full)** | **23.15 ± 10.51** | **45.86 ± 28.45** |

- Spoofing detection: **precision 0.82, recall 0.98, F1 0.89**
- Per-update latency: **0.13 ms mean, 0.19 ms p95** on one CPU core (≈800× inside a 10 Hz budget)

Raw metrics are in `results/`; generated plots and the two block diagrams are in `figures/`.

## Reproduce

```bash
pip install -r requirements.txt
python src/sim.py            # single run -> results.json, traj.npz
python src/experiments.py    # 10-seed table + spoof & LEO sweeps -> experiments.json
python src/figures.py        # result figures
python src/block.py          # architecture block diagram (Fig 2 in the paper)
python src/scenario.py       # operational scenario diagram (Fig 1 in the paper)
```

Everything runs on CPU in a few seconds; the random seed is fixed for reproducibility.

## Note on scope

The evaluation is a controlled simulation so that ground truth, spoofing, and satellite
geometry are known exactly and each component can be isolated. The VIO front end is
modelled as a stochastic velocity source rather than run on real imagery; LEO geometry
uses a local-frame placeholder. See the paper's *Limitations* section.

## Citation

```bibtex
@article{daltonLODESTAR2026,
  title   = {Integrity Aware LEO Doppler Navigation for GNSS Denied Aerospace Vehicles},
  author  = {Arul Dalton, G.},
  journal = {Journal of the Astronautical Sciences},
  year    = {2026},
  note    = {Under review}
}
```

## License

MIT — see [LICENSE](LICENSE).
