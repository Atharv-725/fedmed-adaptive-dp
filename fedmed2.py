import os,json,random,warnings
import numpy as np
import pandas as pd
import torch,torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader,TensorDataset
from collections import OrderedDict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score
from opacus.accountants import RDPAccountant
warnings.filterwarnings("ignore")

NUM_CLIENTS=10;CLIENT_FRACTION=0.6;ROUNDS=20;LOCAL_EPOCHS=5
BATCH_SIZE=32;LR=1e-3;BASE_NOISE_MULT=1.0;MAX_GRAD_NORM=1.0
EPSILON_MAX=10.0;DELTA=1e-5;OUTPUT_DIR="results";SEED=42

def set_seed(s=SEED):
    random.seed(s);np.random.seed(s);torch.manual_seed(s)

def load_data():
    url="https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    cols=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"]
    try:
        df=pd.read_csv(url,names=cols,na_values="?")
        print("[Data] Downloaded from UCI.")
    except:
        print("[Data] Using synthetic data.")
        np.random.seed(SEED);n=1000
        df=pd.DataFrame({"age":np.random.randint(30,80,n),"sex":np.random.randint(0,2,n),"cp":np.random.randint(0,4,n),"trestbps":np.random.randint(90,180,n),"chol":np.random.randint(150,400,n),"fbs":np.random.randint(0,2,n),"restecg":np.random.randint(0,3,n),"thalach":np.random.randint(70,200,n),"exang":np.random.randint(0,2,n),"oldpeak":np.random.uniform(0,6,n).round(1),"slope":np.random.randint(0,3,n),"ca":np.random.randint(0,4,n),"thal":np.random.choice([3.0,6.0,7.0],n),"target":np.random.randint(0,2,n)})
    df=df.dropna();df["target"]=(df["target"]>0).astype(int)
    X=StandardScaler().fit_transform(df.drop("target",axis=1).values.astype(np.float32))
    return X,df["target"].values.astype(np.int64)

class MedNet(nn.Module):
    def __init__(self,d=13,h=64,c=2):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(0.3),nn.Linear(h,h//2),nn.ReLU(),nn.Linear(h//2,c))
    def forward(self,x):return self.net(x)

def get_params(m): return [v.cpu().numpy() for v in m.state_dict().values()]
def set_params(m,p):
    sd=OrderedDict({k:torch.tensor(v) for k,v in zip(m.state_dict().keys(),p)})
    m.load_state_dict(sd,strict=True)

def dp_step(model,opt,Xb,yb,sigma,C,crit):
    grads=[]
    for i in range(len(Xb)):
        opt.zero_grad()
        crit(model(Xb[i:i+1]),yb[i:i+1]).backward()
        gn=torch.sqrt(sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None))
        cf=min(1.0,C/(gn.item()+1e-8))
        grads.append([p.grad.clone()*cf for p in model.parameters() if p.grad is not None])
    pw=[p for p in model.parameters() if p.requires_grad]
    opt.zero_grad()
    for i,p in enumerate(pw):
        agg=torch.stack([g[i] for g in grads]).mean(0)
        p.grad=agg+torch.randn_like(agg)*sigma*C/len(Xb)
    opt.step()

def train_client(gp,Xc,yc,sigma,d):
    m=MedNet(d);set_params(m,gp);m.train()
    Xt=torch.tensor(Xc,dtype=torch.float32);yt=torch.tensor(yc,dtype=torch.long)
    opt=Adam(m.parameters(),lr=LR);crit=nn.CrossEntropyLoss()
    for _ in range(LOCAL_EPOCHS):
        for Xb,yb in DataLoader(TensorDataset(Xt,yt),batch_size=BATCH_SIZE,shuffle=True):
            dp_step(m,opt,Xb,yb,sigma,MAX_GRAD_NORM,crit)
    m.eval()
    with torch.no_grad():
        logits=m(Xt);preds=logits.argmax(1).numpy()
    return get_params(m),len(Xc),crit(logits,yt).item(),accuracy_score(yc,preds)

def evaluate(gp,Xt,yt_np,d):
    m=MedNet(d);set_params(m,gp);m.eval()
    Xtt=torch.tensor(Xt,dtype=torch.float32);ytt=torch.tensor(yt_np,dtype=torch.long)
    with torch.no_grad():
        logits=m(Xtt);probs=torch.softmax(logits,1)[:,1].numpy();preds=logits.argmax(1).numpy()
    try: auc=roc_auc_score(yt_np,probs)
    except: auc=0.0
    return nn.CrossEntropyLoss()(logits,ytt).item(),accuracy_score(yt_np,preds),f1_score(yt_np,preds,zero_division=0),auc

def main():
    set_seed()
    print("\n"+"="*60)
    print("  FedMed: Federated Learning with Adaptive DP")
    print("="*60+"\n")
    X,y=load_data()
    d=X.shape[1]
    print(f"[Data] Samples:{len(X)} Features:{d} Pos:{y.mean():.2f}")
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=SEED,stratify=y)
    idx=np.random.permutation(len(Xtr));splits=np.array_split(idx,NUM_CLIENTS)
    clients=[(Xtr[s],ytr[s]) for s in splits]
    print(f"[Data] Client sizes:{[len(c[0]) for c in clients]}")
    scores=np.array([np.mean(np.var(Xc,axis=0)) for Xc,_ in clients])
    mn,mx=scores.min(),scores.max()
    sigmas=1.0+0.5*(scores-mn)/(mx-mn+1e-8)
    print(f"\n[DP] Sensitivity:{np.round(scores,3)}")
    print(f"[DP] Noise mults:{np.round(sigmas,3)}\n")
    gp=get_params(MedNet(d))
    acc_h,f1_h,auc_h,eps_h=[],[],[],[]
    acc=RDPAccountant()
    sr=min(0.9,BATCH_SIZE/(len(Xtr)//NUM_CLIENTS))
    spr=(len(Xtr)//NUM_CLIENTS//BATCH_SIZE)*LOCAL_EPOCHS
    print(f"[Training] {ROUNDS} rounds...\n")
    for rnd in range(1,ROUNDS+1):
        sel=random.sample(range(NUM_CLIENTS),max(1,int(NUM_CLIENTS*CLIENT_FRACTION)))
        ap,ns=[],[]
        for cid in sel:
            Xc,yc=clients[cid]
            p,n,l,a=train_client(gp,Xc,yc,sigmas[cid],d)
            print(f"  [Client {cid:02d}] loss={l:.4f} acc={a:.4f} sigma={sigmas[cid]:.3f}")
            ap.append(p);ns.append(n)
        total=sum(ns)
        gp=[sum((ns[i]/total)*ap[i][j] for i in range(len(ap))) for j in range(len(gp))]
        tl,ta,tf,tauc=evaluate(gp,Xte,yte,d)
        acc_h.append(ta);f1_h.append(tf);auc_h.append(tauc)
        acc.step(noise_multiplier=BASE_NOISE_MULT, sample_rate=sr)
        eps=acc.get_epsilon(delta=DELTA);eps_h.append(eps)
        print(f"\n{'='*55}")
        print(f"  [Round {rnd:02d}/{ROUNDS}] Acc={ta:.4f} F1={tf:.4f} AUC={tauc:.4f} eps={eps:.4f}")
        print(f"{'='*55}\n")
        if eps>=EPSILON_MAX:print("[DP] Budget exhausted.");break
    print(f"\n  FINAL: Acc={acc_h[-1]*100:.2f}% F1={f1_h[-1]:.4f} AUC={auc_h[-1]:.4f} eps={eps_h[-1]:.4f}\n")
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    json.dump({"final_accuracy":acc_h[-1],"final_f1":f1_h[-1],"final_auc":auc_h[-1],"final_epsilon":eps_h[-1],"accuracy_history":acc_h,"f1_history":f1_h,"auc_history":auc_h,"epsilon_history":eps_h,"sensitivity_scores":scores.tolist(),"noise_multipliers":sigmas.tolist()},open(f"{OUTPUT_DIR}/results.json","w"),indent=2)
    rounds=list(range(1,len(acc_h)+1))
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    fig.suptitle("FedMed Results",fontsize=14,fontweight="bold")
    axes[0,0].plot(rounds,acc_h,marker="o",color="#2196F3",linewidth=2);axes[0,0].set_title("Accuracy");axes[0,0].grid(alpha=0.3)
    axes[0,1].plot(rounds,auc_h,marker="s",color="#4CAF50",linewidth=2);axes[0,1].set_title("AUC-ROC");axes[0,1].grid(alpha=0.3)
    axes[1,0].plot(rounds,eps_h,marker="^",color="#E91E63",linewidth=2);axes[1,0].set_title("Privacy Budget");axes[1,0].grid(alpha=0.3)
    axes[1,1].bar(range(NUM_CLIENTS),sigmas,color="#FF9800",alpha=0.8);axes[1,1].set_title("Noise Multipliers");axes[1,1].grid(alpha=0.3,axis="y")
    plt.tight_layout();plt.savefig(f"{OUTPUT_DIR}/fedmed_results.png",dpi=150);plt.close()
    print(f"Results saved to {OUTPUT_DIR}/")

if __name__=="__main__":
    main()
