import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, kendalltau
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit import QuantumCircuit
import sc_ti_fidelity_demo as M

def end_coherence(layers):
    qc=QuantumCircuit(M.N)
    for L in layers:
        for op in L: M.apply_gate(qc,op)
    R=DensityMatrix(Statevector.from_instruction(qc)).data
    return 2*sum(abs(R[i,j]) for i in range(2**M.N) for j in range(i+1,2**M.N))

SCH=[("B1",lambda L:M.schedule_b1(L,M.N)),
     ("B2s",lambda L:M.schedule_b2(L,M.N,True)),
     ("B2n",lambda L:M.schedule_b2(L,M.N,False))]
rows=[]
for c in range(M.N_CIRCUITS):
    layers=M.gen_layers(M.N,M.DEPTH,np.random.RandomState(M.SEED0+c))
    coh=end_coherence(layers); T=len(layers)
    for nm,sfn in SCH:
        s=sfn(layers); fid,_=M.aer_fidelity(layers,s,include_idle=True)
        efcl=M.efcl_proxy(layers,s)
        rows.append((c,nm,coh,fid,efcl,efcl*T,-np.log(max(fid,1e-12))))

c_=np.array([r[0] for r in rows]); coh=np.array([r[2] for r in rows])
fid=np.array([r[3] for r in rows]); efcl=np.array([r[4] for r in rows])
tot=np.array([r[5] for r in rows]); nlf=np.array([r[6] for r in rows])
carry=coh>1e-6

sp_all=spearmanr(efcl,fid)[0]; sp_c=spearmanr(efcl[carry],fid[carry])[0]
kd_all=kendalltau(efcl,fid)[0]
pe_c=pearsonr(tot[carry],nlf[carry])[0]

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5.2))

# panel 1: EFCL vs Aer fidelity (rank alignment)
ax1.scatter(efcl[~carry],fid[~carry],s=90,c="#b0b0b0",edgecolor="k",
            label="basis-state circuits (0,3,5)\nAer blind to decoherence")
ax1.scatter(efcl[carry],fid[carry],s=90,c="#e45756",edgecolor="k",
            label="coherence-carrying circuits")
ax1.set_xlabel("EFCL  (per-layer cost, lower = better)")
ax1.set_ylabel("Aer state fidelity (higher = better)")
ax1.set_title(f"Rank alignment\nSpearman all={sp_all:+.2f} | coherent={sp_c:+.2f} | Kendall all={kd_all:+.2f}")
ax1.legend(fontsize=8,loc="upper right"); ax1.set_ylim(0,1.02)

# panel 2: linearized -- EFCL*L vs -ln fidelity (quantitative), coherent subset
x=tot[carry]; y=nlf[carry]
ax2.scatter(x,y,s=90,c="#e45756",edgecolor="k")
b,a=np.polyfit(x,y,1)
xs=np.linspace(x.min(),x.max(),50)
ax2.plot(xs,b*xs+a,color="k",ls="--",lw=1.5,label=f"fit (Pearson r={pe_c:+.2f})")
ax2.set_xlabel("EFCL x layers  (total -log-survival cost)")
ax2.set_ylabel("-ln(Aer fidelity)  (measured infidelity)")
ax2.set_title("Quantitative alignment (log-linearized)\ncoherence-carrying circuits only")
ax2.legend(fontsize=9,loc="upper left")

fig.suptitle("EFCL vs independent Aer judge -- 10 circuits x 3 schedules (SC+TI)",fontsize=13)
fig.tight_layout(); fig.savefig("fig_correlation.png",dpi=150,bbox_inches="tight")
print("wrote fig_correlation.png")
print(f"Spearman all={sp_all:+.3f}  coherent={sp_c:+.3f}  Kendall all={kd_all:+.3f}  Pearson(lin,coherent)={pe_c:+.3f}")
