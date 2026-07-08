"""n=9 release-grade gate (sigma=0.02) via a PRE-ROLL robustification.

Architecture (no per-IC NLP; ~1000x cheaper than the n8 composite replan):
  1. PRE-ROLL: from the perturbed hanging IC, run a fixed LQR-about-DOWN for
     T_pre s. Down is a stable equilibrium, so this pulls the sigma=0.02
     perturbation back to ~the nominal start x_nom[0].
  2. TRACK: FastDTVLQR-track the w_v=6e-4 swing-up nominal (the verified,
     trackable-from-~zero-error swing-up).
  3. HOLD: static-LQR upright hold, extended window, trailing 5s in-success-set.

Same plant / sim / predicate as the n5-8 releases (funnels.in_success_set;
5.0s continuous hold; |ang|<=5deg |thetad|<=0.5 |x|<=2 |xdot|<=0.5). Perturbation
model identical to cl_validate_n9_fixed / the n8 gate (N(0,0.02) on all states).

Usage: python scripts/gate_n9_preroll.py [n_ic] [seed] [T_pre] [workers]
"""
import os, sys, json, time
import numpy as np
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(_v,"1")
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/"src")); sys.path.insert(0, str(REPO/"scripts"))

HOLD_S = 5.0
HOLD_WIN = 10.0
NOM = str(REPO/"results"/"nom_n9_dense1ms_wv6en4.npz")

_G = {}

def _init():
    import scipy.linalg as sla
    from cartpole_race.dynamics import NLinkCartPole
    from cartpole_race.env_spec import CartPoleSpec
    from cartpole_race.lqr import static_lqr, wrap_state_error, make_Q, make_R
    from cartpole_race.funnels import in_success_set
    from fast_pieces import FastDTVLQR
    n=9; nx=2*(n+1)
    spec=CartPoleSpec(n_links=n, cart_mass_kg=1.0, link_masses_kg=[0.10]*n,
                      link_lengths_m=[0.50]*n, damping_links_n_m_s_rad=[0.0]*n, force_bound_n=150.0)
    m=NLinkCartPole(spec)
    z=np.load(NOM); Xn,Un,Tn=z["x"],z["u"],float(z["horizon"])
    tv=FastDTVLQR(m,Xn,Un,spec.control_dt_s)
    Kh,_=static_lqr(m); Khrow=np.asarray(Kh).reshape(-1)
    xdown=m.x_equilibrium("up")*0 + m.x_equilibrium("down")
    Ad_,Bd_=m.linearize(xdown,0.0)
    # down-LQR: heavy CART regulation (else the cart drifts ~0.3m while damping
    # the links and returns slowly, leaving a cart offset the swing-up can't
    # track) + heavy velocity damping so the pre-roll settles fast to x_nom[0].
    qd=np.concatenate([[200.0],80.0*np.ones(n),[50.0],80.0*np.ones(n)])
    Pd=sla.solve_continuous_are(Ad_,Bd_,np.diag(qd),make_R())
    Kd=np.linalg.solve(make_R(),Bd_.T@Pd).reshape(-1)
    _G.update(m=m, spec=spec, n=n, nx=nx, Xn=Xn, Tn=Tn, tv=tv, Khrow=Khrow,
              xup=m.x_equilibrium("up"), xdown=xdown, Kd=Kd,
              wrap=wrap_state_error, iss=in_success_set,
              dt=spec.control_dt_s, fb=spec.force_bound_n, track=spec.track_half_length_m)

def _trailing(x_log):
    m=_G["m"]; dt=_G["dt"]; iss=_G["iss"]
    inset=np.array([iss(m,xx) for xx in x_log]); run=0
    for v in inset: run=run+1 if v else 0
    return (run-1)*dt

def _wrap_down(x):
    n=_G["n"]; e=np.asarray(x,float).reshape(-1)-_G["xdown"]
    e[1:1+n]=(e[1:1+n]+np.pi)%(2*np.pi)-np.pi
    return e

def run_ic(args):
    tag, seed, T_pre = args
    m=_G["m"]; spec=_G["spec"]; n=_G["n"]; nx=_G["nx"]; dt=_G["dt"]; fb=_G["fb"]
    track=_G["track"]; Xn=_G["Xn"]; Tn=_G["Tn"]; tv=_G["tv"]; Kd=_G["Kd"]
    Khrow=_G["Khrow"]; xup=_G["xup"]; wrap=_G["wrap"]
    rng=np.random.default_rng((seed, tag))
    dx=np.zeros(nx)
    dx[0]=rng.normal(0,0.02); dx[1:1+n]=rng.normal(0,0.02,n)
    dx[1+n]=rng.normal(0,0.02); dx[2+n:]=rng.normal(0,0.02,n)
    x=(Xn[0]+dx).copy()
    pert=float(np.rad2deg(np.max(np.abs(dx[1:1+n]))))
    # 1. ADAPTIVE pre-roll: LQR-about-down until the link angle+rate residual is
    #    small (the regime the swing-up track handles), capped at T_pre seconds.
    def prep(x,t): return float(np.clip(-float(Kd@_wrap_down(x)),-fb,fb))
    tol=0.0015; chunk=0.5; elapsed=0.0; pre_maxF=0.0; pre_maxx=0.0
    while elapsed < T_pre - 1e-9:
        _t,xp,up=m.rollout_zoh(x,prep,chunk,dt,spec.rk4_max_step_s)
        x=xp[-1]; elapsed+=chunk
        pre_maxF=max(pre_maxF,float(np.max(np.abs(up)))); pre_maxx=max(pre_maxx,float(np.max(np.abs(xp[:,0]))))
        # tight settle on link angles+rates (what the swing-up track needs); the
        # heavy-cart down-LQR keeps the cart small independently.
        e=_wrap_down(x)
        metric=max(float(np.max(np.abs(e[1:1+n]))), float(np.max(np.abs(e[2+n:]))))
        if metric < tol: break
    resid=float(np.max(np.abs(_wrap_down(x)))); t_pre_used=round(elapsed,2)
    # 2. track nominal
    def tp(x,t): return float(np.clip(tv.policy(x,t),-fb,fb))
    _t,x1,u1=m.rollout_zoh(x,tp,Tn,dt,spec.rk4_max_step_s)
    xh=x1[-1]
    if np.any(np.isnan(xh)):
        return dict(tag=tag,success=False,fail="track_nan",pert_deg=round(pert,3),resid=round(resid,5))
    ho=float(np.rad2deg(np.max(np.abs(wrap(xh,xup,n)[1:1+n]))))
    if ho>20:
        return dict(tag=tag,success=False,fail="track_diverged",handoff_deg=round(ho,3),
                    pert_deg=round(pert,3),resid=round(resid,5))
    # 3. static hold
    def hp(x,t): return float(np.clip(-float(Khrow@wrap(x,xup,n)),-fb,fb))
    _t,x3,u3=m.rollout_zoh(xh,hp,HOLD_WIN,dt,spec.rk4_max_step_s)
    hr=_trailing(x3)
    tr_ok=bool(max(np.max(np.abs(x1[:,0])),np.max(np.abs(x3[:,0])),pre_maxx)<=track)
    peakF=float(max(np.max(np.abs(u1)),np.max(np.abs(u3)),pre_maxF))
    ok=bool(hr>=HOLD_S-1e-9 and tr_ok)
    return dict(tag=tag,success=ok,handoff_deg=round(ho,4),hold_s=round(hr,2),
                peakF=round(peakF,1),pert_deg=round(pert,3),resid=round(resid,5),
                t_pre=t_pre_used, track_ok=tr_ok, fail=None if ok else "hold")

def wilson(k,nn,z=1.96):
    if nn==0: return (0.0,0.0)
    p=k/nn; d=1+z*z/nn
    c=(p+z*z/(2*nn))/d; h=z*np.sqrt(p*(1-p)/nn+z*z/(4*nn*nn))/d
    return (round(c-h,4),round(min(1.0,c+h),4))

def main():
    n_ic=int(sys.argv[1]) if len(sys.argv)>1 else 24
    seed=int(sys.argv[2]) if len(sys.argv)>2 else 12345
    T_pre=float(sys.argv[3]) if len(sys.argv)>3 else 9.0  # adaptive pre-roll CAP (matches banked gate)
    workers=int(sys.argv[4]) if len(sys.argv)>4 else max(1,(os.cpu_count() or 2)-1)
    from concurrent.futures import ProcessPoolExecutor
    tasks=[(t,seed,T_pre) for t in range(n_ic)]
    t0=time.time(); results=[]
    if workers>1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init) as ex:
            for r in ex.map(run_ic, tasks): results.append(r); print(json.dumps(r),flush=True)
    else:
        _init()
        for a in tasks: r=run_ic(a); results.append(r); print(json.dumps(r),flush=True)
    results.sort(key=lambda r:r["tag"])
    k=sum(1 for r in results if r["success"])
    lo,hi=wilson(k,n_ic)
    print(f"[GATE-n9-PREROLL] {k}/{n_ic} success  sigma=0.02 T_pre={T_pre}s  "
          f"Wilson95=[{lo},{hi}]  ({time.time()-t0:.0f}s, {workers}w)",flush=True)
    out=REPO/"results"/f"gate_n9_preroll_seed{seed}.json"
    out.write_text(json.dumps(dict(controller="preroll_down_lqr+tvlqr_track+static_hold",
        sigma=0.02, T_pre_s=T_pre, hold_window_s=HOLD_WIN, n_success=k, n_ic=n_ic,
        seed=seed, wilson95=[lo,hi], results=results),indent=1))
    print("saved",out,flush=True)

if __name__=="__main__":
    main()
