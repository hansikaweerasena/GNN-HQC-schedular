import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit import QuantumCircuit
import sc_ti_fidelity_demo as M

def end_coherence(layers):
    qc=QuantumCircuit(M.N)
    for L in layers:
        for op in L: M.apply_gate(qc,op)
    R=DensityMatrix(Statevector.from_instruction(qc)).data
    tot=0.0
    for i in range(2**M.N):
        for j in range(i+1,2**M.N):
            tot+=abs(R[i,j])
    return 2*tot  # total off-diagonal coherence mass in the ideal output

rows=[]
for c in range(M.N_CIRCUITS):
    rng=np.random.RandomState(M.SEED0+c)
    layers=M.gen_layers(M.N,M.DEPTH,rng)
    coh=end_coherence(layers)
    for nm,sfn in [("B1",lambda L:M.schedule_b1(L,M.N)),
                   ("B2s",lambda L:M.schedule_b2(L,M.N,True)),
                   ("B2n",lambda L:M.schedule_b2(L,M.N,False))]:
        sched=sfn(layers)
        fid,_=M.aer_fidelity(layers,sched,include_idle=True)
        efcl=M.efcl_proxy(layers,sched)
        T=len(layers)
        rows.append(dict(circ=c,sched=nm,coh=coh,fid=fid,efcl=efcl,
                         efcl_total=efcl*T, neglogfid=-np.log(max(fid,1e-12))))

import json
efcl=np.array([r["efcl"] for r in rows])
fid=np.array([r["fid"] for r in rows])
efcl_tot=np.array([r["efcl_total"] for r in rows])
nlf=np.array([r["neglogfid"] for r in rows])
coh=np.array([r["coh"] for r in rows])

print("=== ALL 30 POINTS ===")
print(f"Spearman(EFCL, Aer_fid)          = {spearmanr(efcl,fid)[0]:+.3f}  (expect strong NEGATIVE)")
print(f"Kendall (EFCL, Aer_fid)          = {kendalltau(efcl,fid)[0]:+.3f}")
print(f"Pearson (EFCL_total, -ln fid)    = {pearsonr(efcl_tot,nlf)[0]:+.3f}  (linearized; expect POSITIVE)")
print(f"Pearson (EFCL, Aer_fid) [raw]    = {pearsonr(efcl,fid)[0]:+.3f}  (raw, understates due to curvature)")

# coherence-carrying subset
mask=coh>1e-6
print(f"\n=== COHERENCE-CARRYING SUBSET ({mask.sum()} pts; drops basis-state circuits) ===")
print(f"Spearman(EFCL, Aer_fid)          = {spearmanr(efcl[mask],fid[mask])[0]:+.3f}")
print(f"Pearson (EFCL_total, -ln fid)    = {pearsonr(efcl_tot[mask],nlf[mask])[0]:+.3f}")

# within-circuit ranking agreement (does EFCL pick the same best schedule as Aer?)
agree=0; tot=0
for c in range(M.N_CIRCUITS):
    grp=[r for r in rows if r["circ"]==c]
    if len({r["fid"] for r in grp})<2: 
        continue  # flat (basis-state) -> undefined ranking
    tot+=1
    best_efcl=min(grp,key=lambda r:r["efcl"])["sched"]
    best_fid =max(grp,key=lambda r:r["fid"])["sched"]
    agree+= (best_efcl==best_fid)
print(f"\nWithin-circuit best-schedule agreement (non-flat circuits): {agree}/{tot}")

# which circuits are basis-state (flat)?
flat=[c for c in range(M.N_CIRCUITS) if end_coherence(M.gen_layers(M.N,M.DEPTH,np.random.RandomState(M.SEED0+c)))<1e-6]
print(f"basis-state (decoherence-blind) circuits: {flat}")

np.save("corr_rows.npy", rows, allow_pickle=True)
