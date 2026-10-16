#!/usr/bin/env python3
"""Small, reproducible MovieLens-100K GPT-style recommendation/RL experiment."""
import argparse
import bisect
import copy
import json
import math
import random
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def get_data(root):
    root.mkdir(parents=True, exist_ok=True)
    p = root / "ml-100k" / "u.data"
    if not p.exists():
        z = root / "ml-100k.zip"
        urllib.request.urlretrieve(URL, z)
        with zipfile.ZipFile(z) as f: f.extract("ml-100k/u.data", root)
    rows = []
    for line in p.read_text().splitlines():
        u, i, r, t = map(int, line.split())
        rows.append((t, u, i, r))
    rows.sort(key=lambda x: (x[0], x[1], x[2]))
    n = len(rows)
    times = [x[0] for x in rows]
    # Keep every tied timestamp on the same side of each global boundary.
    a = bisect.bisect_left(times, times[int(.8*n)])
    b = bisect.bisect_left(times, times[int(.9*n)])
    train_items = sorted({x[2] for x in rows[:a]})
    mapping = {v: j+1 for j, v in enumerate(train_items)}
    return rows, a, b, mapping


def examples(rows, a, b, mapping, length):
    histories = defaultdict(list); train=[]; val=[]; test=[]
    j = 0
    while j < len(rows):
        timestamp, u = rows[j][0], rows[j][1]
        k = j+1
        while k < len(rows) and rows[k][0] == timestamp and rows[k][1] == u:
            k += 1
        h = histories[u]
        # All items rated at the same timestamp share the strictly earlier
        # context. Their within-timestamp ordering is not observed.
        prior = tuple(h)
        for _, _, item, rating in rows[j:k]:
            if prior and item in mapping:
                ex = (u, prior, mapping[item], (rating-1)/4)
                (train if j<a else val if j<b else test).append(ex)
        # A fixed order is needed only when representing a past timestamp's
        # unordered set as tokens in a later context.
        h.extend(mapping[item] for _, _, item, _ in rows[j:k] if item in mapping)
        j = k
    return train, val, test


def batch(xs, length, device):
    h = torch.zeros((len(xs),length), dtype=torch.long)
    for j,x in enumerate(xs):
        seq=x[1][-length:]; h[j,-len(seq):]=torch.tensor(seq)
    y=torch.tensor([x[2] for x in xs],dtype=torch.long)
    r=torch.tensor([x[3] for x in xs],dtype=torch.float32)
    return h.to(device),y.to(device),r.to(device)


class GPTRec(nn.Module):
    def __init__(self, n_items, length, dim=64, layers=2, heads=4):
        super().__init__()
        self.length=length
        self.item=nn.Embedding(n_items+1,dim,padding_idx=0)
        self.pos=nn.Embedding(length,dim)
        block=nn.TransformerEncoderLayer(dim,heads,dim*4,.1,batch_first=True,norm_first=True)
        self.blocks=nn.TransformerEncoder(block,layers,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(dim)
        self.out=nn.Linear(dim,n_items+1)
    def forward(self,h):
        x=self.item(h)+self.pos(torch.arange(h.shape[1],device=h.device))[None,:,:]
        mask=torch.triu(torch.ones(h.shape[1],h.shape[1],device=h.device,dtype=torch.bool),1)
        x=self.norm(self.blocks(x,mask=mask,src_key_padding_mask=(h==0)))
        z=self.out(x[:,-1]); z[:,0]=-1e9
        return z


class RewardModel(nn.Module):
    def __init__(self,n_items,dim=48):
        super().__init__()
        self.emb=nn.Embedding(n_items+1,dim,padding_idx=0)
        self.bias=nn.Embedding(n_items+1,1)
        self.net=nn.Sequential(nn.Linear(dim*3,96),nn.ReLU(),nn.Linear(96,1))
    def forward(self,h,item):
        if item.ndim==1: item=item[:,None]
        ctx=self.emb(h).sum(1)/(h.ne(0).sum(1).clamp(min=1)[:,None])
        it=self.emb(item)
        c=ctx[:,None,:].expand_as(it)
        x=torch.cat([c,it,c*it],-1)
        return torch.sigmoid(self.net(x).squeeze(-1)+self.bias(item).squeeze(-1))


def shuffled_batches(xs, bs, rng):
    order=rng.permutation(len(xs))
    for start in range(0,len(order),bs):
        yield [xs[j] for j in order[start:start+bs]]


def fit_base(model, train, val, args, device):
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    rng=np.random.default_rng(args.seed)
    logs=[]
    for epoch in range(args.base_epochs):
        model.train(); losses=[]
        for xs in shuffled_batches(train,args.batch,rng):
            h,y,_=batch(xs,args.length,device)
            loss=F.cross_entropy(model(h),y)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            vl=[]
            for k in range(0,len(val),args.batch):
                h,y,_=batch(val[k:k+args.batch],args.length,device)
                vl.append(F.cross_entropy(model(h),y).item())
        logs.append({'epoch':epoch+1,'train_ce':float(np.mean(losses)),'val_ce':float(np.mean(vl))})
        print('base',logs[-1],flush=True)
    return logs


def fit_reward(model, train, val, args, device):
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    rng=np.random.default_rng(args.seed+1); logs=[]
    for epoch in range(args.reward_epochs):
        model.train(); losses=[]
        for xs in shuffled_batches(train,args.batch,rng):
            h,y,r=batch(xs,args.length,device)
            pred=model(h,y).squeeze(1)
            loss=F.mse_loss(pred,r)
            opt.zero_grad();loss.backward();opt.step();losses.append(loss.item())
        logs.append({'epoch':epoch+1,'train_mse':float(np.mean(losses)),'val':reward_quality(model,val,args,device)})
        print('reward',logs[-1],flush=True)
    return logs


@torch.no_grad()
def reward_quality(model,xs,args,device):
    model.eval(); pp=[]; yy=[]
    for k in range(0,len(xs),args.batch):
        h,y,r=batch(xs[k:k+args.batch],args.length,device)
        pp.extend(model(h,y).squeeze(1).cpu().tolist());yy.extend(r.cpu().tolist())
    p=np.array(pp); y=np.array(yy)
    return {'mae':float(np.abs(p-y).mean()),'rmse':float(np.sqrt(np.mean((p-y)**2))),
            'train_mean_baseline_rmse':float(np.sqrt(np.mean((y-args.train_rating_mean)**2)))}


def mask_seen(logits,h,xs=None):
    logits=logits.clone()
    logits[:,0]=-1e9
    logits.scatter_(1,h,-1e9)
    if xs is not None:
        for row,x in enumerate(xs):
            logits[row,list(set(x[1]))]=-1e9
    logits[:,0]=-1e9
    return logits


def fit_rl(base, rm, train, popularity, beta, novelty, args, device):
    model=copy.deepcopy(base).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.rl_lr,weight_decay=.001)
    rng=np.random.default_rng(args.seed+3)
    selected=rng.choice(len(train),size=min(args.rl_examples,len(train)),replace=False)
    subset=[train[j] for j in selected]
    pop=torch.tensor(popularity,device=device,dtype=torch.float32)
    # Penalize the most popular item by at most 0.25 normalized rating units.
    penalty=.25*torch.log1p(pop)/torch.log1p(pop.max())
    base.eval();rm.eval();model.train(); losses=[]
    for xs in shuffled_batches(subset,args.batch,rng):
        h,_,_=batch(xs,args.length,device)
        with torch.no_grad():
            ref=mask_seen(base(h),h,xs)
            cand=ref.topk(min(64,ref.shape[1]-1),dim=1).indices
            rew=rm(h,cand)
            if novelty: rew=rew-penalty[cand]
            ref_p=F.softmax(ref,dim=1)
        z=mask_seen(model(h),h,xs)
        # Exact expected contextual-bandit reward over a fixed candidate set.
        # Full-catalog KL limits movement into unscored actions.
        cp=F.softmax(z.gather(1,cand),dim=1)
        expected=(cp*rew).sum(1).mean()
        kl=F.kl_div(F.log_softmax(z,dim=1),ref_p,reduction='batchmean')
        loss=-expected+beta*kl
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        losses.append((float(expected.item()),float(kl.item())))
    return model,{'mean_train_expected_reward':float(np.mean([x[0] for x in losses])),
                  'mean_train_kl':float(np.mean([x[1] for x in losses]))}


@torch.no_grad()
def evaluate(model,rm,xs,popularity,args,device):
    model.eval();rm.eval(); hit=[];ndcg=[];positive_hit=[];positive_ndcg=[];recs=[];rewards=[]; pops=[];seen_rates=[]
    user_hit=defaultdict(list);user_ndcg=defaultdict(list)
    for k in range(0,len(xs),args.batch):
        chunk=xs[k:k+args.batch]
        h,y,_=batch(chunk,args.length,device)
        z=mask_seen(model(h),h,chunk)
        top=z.topk(10,dim=1).indices
        ranks=(top==y[:,None]).nonzero()
        ranks_by={int(i):int(j) for i,j in ranks.cpu().tolist()}
        for i,x in enumerate(chunk):
            h_i=int(i in ranks_by)
            n_i=1/math.log2(ranks_by[i]+2) if h_i else 0
            hit.append(h_i);ndcg.append(n_i)
            user_hit[x[0]].append(h_i);user_ndcg[x[0]].append(n_i)
        positive_hit.extend([int(i in ranks_by) for i,x in enumerate(chunk) if x[3]>=.75])
        positive_ndcg.extend([1/math.log2(ranks_by[i]+2) if i in ranks_by else 0
                              for i,x in enumerate(chunk) if x[3]>=.75])
        recs.extend(top.cpu().flatten().tolist())
        rewards.extend(rm(h,top).cpu().flatten().tolist())
        pops.extend([popularity[i] for i in top.cpu().flatten().tolist()])
        seen_rates.extend([int(i in x[1]) for x, row in zip(chunk,top.cpu().tolist()) for i in row])
    counts=Counter(recs)
    return {'n':len(xs),'n_positive':len(positive_hit),
            'recall@10':float(np.mean(hit)),'ndcg@10':float(np.mean(ndcg)),
            'user_balanced_recall@10':float(np.mean([np.mean(v) for v in user_hit.values()])),
            'user_balanced_ndcg@10':float(np.mean([np.mean(v) for v in user_ndcg.values()])),
            'positive_recall@10':float(np.mean(positive_hit)),
            'positive_ndcg@10':float(np.mean(positive_ndcg)),
            'predicted_rating':float(1+4*np.mean(rewards)),
            'catalog_coverage@10':len(counts)/float(len(popularity)-1),
            'top1_item_share':counts.most_common(1)[0][1]/len(recs),
            'top10_items_share':sum(v for _,v in counts.most_common(10))/len(recs),
            'mean_train_popularity':float(np.mean(pops)),
            'seen_rate':float(np.mean(seen_rates))}


@torch.no_grad()
def rollout(model,rm,xs,args,device):
    model.eval();rm.eval(); rng=np.random.default_rng(args.seed+8)
    pick=[xs[i] for i in rng.choice(len(xs),min(200,len(xs)),replace=False)]
    histories=[list(x[1]) for x in pick]; curve=[]; repeat=[]
    for step in range(5):
        synthetic=[(x[0],tuple(h),x[2],x[3]) for x,h in zip(pick,histories)]
        h,_,_=batch(synthetic,args.length,device)
        item=mask_seen(model(h),h,synthetic).argmax(1)
        val=rm(h,item).squeeze(1)
        curve.append(float(1+4*val.mean().item()))
        repeat.append(float(np.mean([int(int(i) in hist) for i,hist in zip(item.cpu(),histories)])))
        for hist,i in zip(histories,item.cpu().tolist()): hist.append(i)
    return {'predicted_rating_by_step':curve,'repeat_rate_by_step':repeat}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',type=Path,default=Path('data'))
    ap.add_argument('--out-dir',type=Path,default=Path('results'))
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--length',type=int,default=30)
    ap.add_argument('--batch',type=int,default=256)
    ap.add_argument('--base-epochs',type=int,default=3)
    ap.add_argument('--reward-epochs',type=int,default=5)
    ap.add_argument('--rl-examples',type=int,default=6000)
    ap.add_argument('--lr',type=float,default=.001)
    ap.add_argument('--rl-lr',type=float,default=.0001)
    ap.add_argument('--device',default='auto')
    args=ap.parse_args();seed_all(args.seed)
    device=('mps' if torch.backends.mps.is_available() else 'cpu') if args.device=='auto' else args.device
    args.out_dir.mkdir(parents=True,exist_ok=True)
    rows,a,b,mapping=get_data(args.data_dir)
    train,val,test=examples(rows,a,b,mapping,args.length)
    args.train_rating_mean=float(np.mean([x[3] for x in train]))
    # One starting context per user for the synthetic rollout.
    first={}
    for x in test: first.setdefault(x[0],x)
    rollout_starts=list(first.values())
    evaluation=test
    pop=np.zeros(len(mapping)+1,dtype=np.int64)
    for x in train: pop[x[2]]+=1
    base=GPTRec(len(mapping),args.length).to(device)
    rm=RewardModel(len(mapping)).to(device)
    base_log=fit_base(base,train,val,args,device)
    rm_log=fit_reward(rm,train,val,args,device)
    result={'config':vars(args)|{'data_dir':str(args.data_dir),'out_dir':str(args.out_dir),'device':str(device)},
            'data':{'rows':len(rows),'train_cutoff_timestamp':rows[a][0],
                    'val_cutoff_timestamp':rows[b][0],'train_examples':len(train),
                    'val_examples':len(val),'test_examples':len(test),'test_users':len(rollout_starts),
                    'train_events':a,'val_events':b-a,'test_events':len(rows)-b,
                    'strict_cutoff_check':bool(rows[a-1][0]<rows[a][0] and rows[b-1][0]<rows[b][0]),
                    'train_items':len(mapping)},'base_training':base_log,'reward_training':rm_log,
            'reward_test':reward_quality(rm,test,args,device),'policies':{}}
    result['policies']['base']={'metrics':evaluate(base,rm,evaluation,pop,args,device),
                                'rollout':rollout(base,rm,rollout_starts,args,device)}
    torch.save({'model':base.state_dict(),'items':mapping,'length':args.length},args.out_dir/'base.pt')
    torch.save(rm.state_dict(),args.out_dir/'reward.pt')
    for name,beta,novelty in [('rating_weak_kl',.03,False),('rating_strong_kl',.3,False),('rating_novelty',.03,True)]:
        policy,rl_log=fit_rl(base,rm,train,pop,beta,novelty,args,device)
        result['policies'][name]={'rl_training':rl_log,
            'metrics':evaluate(policy,rm,evaluation,pop,args,device),
            'rollout':rollout(policy,rm,rollout_starts,args,device)}
        torch.save(policy.state_dict(),args.out_dir/f'{name}.pt')
        print(name,result['policies'][name]['metrics'],flush=True)
    (args.out_dir/'metrics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print('saved',args.out_dir/'metrics.json',flush=True)

if __name__=='__main__': main()
