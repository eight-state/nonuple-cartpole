import os, sys, time
from pathlib import Path
import numpy as np
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(v,"1")
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src")); sys.path.insert(0,str(Path(__file__).resolve().parent))
from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.env_spec import CartPoleSpec
from cartpole_race.collocation import solve_trajopt
from cartpole_race.lqr import StaticLQRPolicy, static_lqr
from cartpole_race.rollout import static_hold_rollout
from _dtvlqr import DiscreteTVLQR

def main():
    w_v=float(sys.argv[1]); tag=sys.argv[2] if len(sys.argv)>2 else f"{w_v:.0e}"
    h_ms=float(sys.argv[3]) if len(sys.argv)>3 else 4.0
    n=9; T=10.0; N=int(round(T/(h_ms*1e-3))); stride=int(round(h_ms))
    spec=CartPoleSpec(n_links=n,cart_mass_kg=1.0,link_masses_kg=[0.10]*n,link_lengths_m=[0.50]*n,damping_links_n_m_s_rad=[0.0]*n,force_bound_n=150.0)
    m=NLinkCartPole(spec)
    d=np.load("runs/r2/nom_n9_gluck.npz",allow_pickle=True); X9=d["states"]; U9=d["forces"]
    sw=int(round((T/N)/0.001)); Xw=X9[::sw][:N+1].copy(); Uw=U9[:N*sw].reshape(N,sw).mean(axis=1); Xw[0]=m.x_equilibrium("down")
    print(f"[WV {tag}] w_v={w_v} N={N} ({h_ms}ms)",flush=True); t0=time.time()
    res=solve_trajopt(m,m.x_equilibrium("down"),horizon_s=T,n_nodes=N,terminal_tol_rad=2e-4,force_bound=100.0,w_u=1e-4,w_v=w_v,x_init_guess=Xw,u_init_guess=Uw,zoh_consistent=False,max_iter=5000,print_level=5)
    pv=float(np.max(np.abs(res.x[:,1+n:])))
    print(f"[WV {tag}] {res.solver_status} defect={res.max_defect:.3e} peakF={np.abs(res.u).max():.1f}N peakVEL={pv:.1f}rad/s {time.time()-t0:.0f}s",flush=True)
    np.savez(f"runs/r2/nom_n9_4ms_wv{tag}.npz",x=res.x,u=res.u,horizon=T,n=n,force=150.0,n_nodes=N)
    if not res.success: print(f"[WV {tag}] NLP not converged",flush=True); return
    n_sub=max(1,int(np.ceil(spec.control_dt_s/spec.rk4_max_step_s))); dt_sub=spec.control_dt_s/n_sub
    Xd=[res.x[0]]; Ud=[]
    for k in range(N):
        xx=res.x[k].astype(float).copy()
        for _ in range(stride):
            for _ in range(n_sub): xx=m.rk4_step(xx,float(res.u[k]),dt_sub)
            Xd.append(xx.copy()); Ud.append(float(res.u[k]))
    Xd=np.array(Xd); Ud=np.array(Ud); np.savez(f"runs/r2/nom_n9_dense1ms_wv{tag}.npz",x=Xd,u=Ud,horizon=T,n=n,force=150.0)
    tv=DiscreteTVLQR(m,Xd,Ud,spec.control_dt_s); print(f"[WV {tag}] DTVLQR rho={tv.monodromy():.4g}",flush=True)
    K,P=static_lqr(m); sp_=StaticLQRPolicy(m,K); sp_.P=P
    x0=m.x_equilibrium("down")
    t1,x1,u1=m.rollout_zoh(x0,lambda x,t: float(np.clip(tv.policy(x,t),-150,150)),T,spec.control_dt_s,spec.rk4_max_step_s)
    xup=m.x_equilibrium("up"); xh=x1[-1]
    hdev=np.rad2deg(np.max(np.abs(((xh[1:1+n]-xup[1:1+n]+np.pi)%(2*np.pi))-np.pi)))
    print(f"[WV {tag}] CL handoff dev {hdev:.5f} deg peakF {np.abs(u1).max():.1f} N",flush=True)
    succ,info=static_hold_rollout(m,xh,sp_,hold_time_s=5.0)
    print(f"[WV {tag}] HOLD success={succ} maxF={info.get('max_force'):.1f}",flush=True)
    if succ: print(f"\n*** WV {tag}: n=9 UNPERTURBED CLOSED-LOOP PASS (peakVEL={pv:.1f}) ***",flush=True)

if __name__=="__main__": main()
