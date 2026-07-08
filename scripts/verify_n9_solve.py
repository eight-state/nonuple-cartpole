"""Definitive n=9 swing-up + balance verification.

Reproduces, on the SAME plant / simulator / predicate as the n5-8 releases
(cartpole_race.funnels.in_success_set; 5.0s continuous hold; |ang|<=5deg,
|thetad|<=0.5, |x|<=2, |xdot|<=0.5):

  PART 1  Refute ledger row F4. The static-LQR hold from the clean 0.0115deg
          swing-up handoff was reported to "fail (hold 3.6s then drift)". That
          is a HOLD-WINDOW artifact: the n9 catch has a ~2.4s settling transient
          (the n9 upright has 9 unstable modes, fastest growth 35/s, vs n8's
          milder upright), so a 6s hold rollout only ever observes ~3.6s of the
          5s hold. Shown side by side: 6s window -> ~3.6s (the "fail"), 12s
          window -> >=5s continuous (PASS). Same controller, same handoff.

  PART 2  End-to-end UNPERTURBED: TVLQR-track the w_v=6e-4 swing-up nominal from
          hanging, hand off near upright, static-LQR hold. Full predicate
          breakdown. This is the end-to-end swing-up+balance PASS the ledger
          said did not exist.

  PART 3  End-to-end PERTURBED (sigma=0.002 IC noise at the hanging start, fixed
          nominal + TVLQR + hold, NO per-IC replan): a genuine perturbed n9
          count. (The release sigma=0.02 additionally needs per-IC swing-up
          replan -- the n8 architecture -- which is a swing-up/solver matter,
          not the catch/hold.)

Usage: python scripts/verify_n9_solve.py [n_perturbed] [seed]
"""
import os, sys, json, time
import numpy as np
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(_v,"1")
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/"src")); sys.path.insert(0, str(REPO/"scripts"))
from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.env_spec import CartPoleSpec
from cartpole_race.lqr import static_lqr, wrap_state_error
from cartpole_race.funnels import in_success_set
from fast_pieces import FastDTVLQR

n=9; nx=2*(n+1)
spec = CartPoleSpec(n_links=n, cart_mass_kg=1.0, link_masses_kg=[0.10]*n,
                    link_lengths_m=[0.50]*n, damping_links_n_m_s_rad=[0.0]*n, force_bound_n=150.0)
m=NLinkCartPole(spec); dt=spec.control_dt_s; fb=spec.force_bound_n; track=spec.track_half_length_m
xup=m.x_equilibrium("up")
K,_P=static_lqr(m); Krow=np.asarray(K).reshape(-1)
NOM = REPO/"results"/"nom_n9_dense1ms_wv6en4.npz"
z=np.load(NOM); Xn,Un,Tn=z["x"],z["u"],float(z["horizon"])
tv=FastDTVLQR(m,Xn,Un,dt)
HOLD_S=5.0


def trailing_hold(x_log):
    """Longest run of continuous in-success-set ticks ending the rollout (s)."""
    inset=np.array([in_success_set(m,xx) for xx in x_log]); run=0
    for v in inset: run=run+1 if v else 0
    return (run-1)*dt

def static_hold(x_start, window):
    def pol(x,t): return float(np.clip(-float(Krow@wrap_state_error(x,xup,n)),-fb,fb))
    _t,xh,uh=m.rollout_zoh(x_start,pol,window,dt,spec.rk4_max_step_s)
    return xh,uh

def swingup_track(x0):
    def pol(x,t): return float(np.clip(tv.policy(x,t),-fb,fb))
    _t,x1,u1=m.rollout_zoh(x0,pol,Tn,dt,spec.rk4_max_step_s)
    return x1,u1


def main():
    n_pert=int(sys.argv[1]) if len(sys.argv)>1 else 8
    seed=int(sys.argv[2]) if len(sys.argv)>2 else 12345
    out={}
    xf=Xn[-1].copy()
    hoff=float(np.rad2deg(np.max(np.abs(wrap_state_error(xf,xup,n)[1:1+n]))))

    print("="*72)
    print("PART 1  F4 refutation: same static-LQR hold, same 0.0115deg handoff,")
    print("        two rollout windows")
    print("="*72)
    for win in (6.0, 12.0):
        xh,uh=static_hold(xf,win)
        hr=trailing_hold(xh)
        print(f"  window={win:4.1f}s  trailing_in_set_hold={hr:5.3f}s  "
              f"{'PASS' if hr>=HOLD_S else 'reported-FAIL (artifact)'}")
        out[f"F4_window_{win}"]=round(hr,3)

    print("\n"+"="*72)
    print("PART 2  End-to-end UNPERTURBED: swing-up TVLQR track + static hold")
    print("="*72)
    x1,u1=swingup_track(Xn[0]); xh0=x1[-1]
    e=wrap_state_error(xh0,xup,n)
    xh,uh=static_hold(xh0,12.0)
    hr=trailing_hold(xh)
    tr_ok=bool(max(np.max(np.abs(x1[:,0])),np.max(np.abs(xh[:,0])))<=track)
    # final-window predicate breakdown (last 5s)
    w=xh[-int(HOLD_S/dt):]
    maxang=float(np.rad2deg(np.max([np.max(np.abs(wrap_state_error(xx,xup,n)[1:1+n])) for xx in w])))
    maxtd=float(np.max([np.max(np.abs(xx[1+n+1:])) for xx in w]))
    maxx=float(np.max(np.abs(w[:,0]))); maxxd=float(np.max(np.abs(w[:,1+n])))
    succ=bool(hr>=HOLD_S and tr_ok)
    print(f"  swing-up peakF_track={np.max(np.abs(u1)):.1f}N -> handoff maxang={hoff:.4f}deg (in_set={in_success_set(m,xh0)})")
    print(f"  hold trailing_in_set={hr:.3f}s (need {HOLD_S})   final-5s window: "
          f"maxang={maxang:.3f}deg maxthetad={maxtd:.3f} max|x|={maxx:.3f} max|xd|={maxxd:.3f}")
    print(f"  >>> UNPERTURBED END-TO-END SUCCESS = {succ}")
    out["unperturbed_success"]=succ; out["unperturbed_hold_s"]=round(hr,3)

    print("\n"+"="*72)
    print(f"PART 3  End-to-end PERTURBED (sigma=0.002, fixed nominal, no replan)")
    print("="*72)
    rng=np.random.default_rng(seed); res=[]; t0=time.time()
    for tag in range(n_pert):
        dx=np.zeros(nx)
        dx[0]=rng.normal(0,0.002); dx[1:1+n]=rng.normal(0,0.002,n)
        dx[1+n]=rng.normal(0,0.002); dx[2+n:]=rng.normal(0,0.002,n)
        x1,u1=swingup_track(Xn[0]+dx); xh0=x1[-1]
        if np.any(np.isnan(xh0)) or np.rad2deg(np.max(np.abs(wrap_state_error(xh0,xup,n)[1:1+n])))>20:
            res.append(dict(tag=tag,success=False,fail="swingup_track")); continue
        xh,uh=static_hold(xh0,12.0); hr=trailing_hold(xh)
        tr_ok=bool(max(np.max(np.abs(x1[:,0])),np.max(np.abs(xh[:,0])))<=track)
        res.append(dict(tag=tag,success=bool(hr>=HOLD_S and tr_ok),hold_s=round(hr,2),
                        handoff_deg=round(float(np.rad2deg(np.max(np.abs(wrap_state_error(xh0,xup,n)[1:1+n])))),4)))
    k=sum(1 for r in res if r["success"])
    print(f"  {k}/{n_pert} end-to-end success  ({time.time()-t0:.0f}s)")
    for r in res: print("   ",json.dumps(r))
    out["perturbed_sigma"]=0.002; out["perturbed_count"]=f"{k}/{n_pert}"; out["perturbed_results"]=res

    outp=REPO/"results"/f"verify_n9_solve_seed{seed}.json"
    outp.write_text(json.dumps(out,indent=1))
    print(f"\nsaved {outp}")


if __name__=="__main__":
    main()
