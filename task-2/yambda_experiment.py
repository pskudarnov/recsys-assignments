#!/usr/bin/env python3
"""Yambda-50M: temporal GPT recommender, feedback models, and 5-step RL."""
import argparse
import copy
import hashlib
import json
import math
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F

from experiment import GPTRec

URL = "https://huggingface.co/datasets/yandex/yambda/resolve/dd6f3a19eef5866e346c3270e098baa641a44948/flat/50m/listens.parquet?download=true"
SHA256 = "eed9cbd094af1e189507d2f8132a0dc9653b90e65480125c7cdccd601e0592d1"
DAY = 24 * 60 * 60 // 5  # Yambda timestamps are in five-second units.


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def get_parquet(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        subprocess.run(["curl", "-L", "--fail", "--silent", "--show-error", URL,
                        "-o", str(path)], check=True)
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    if h.hexdigest()!=SHA256: raise ValueError("Parquet SHA256 differs from pinned source")
    return path


def load_cohort(path, uid_mod=1000, uid_remainder=0):
    columns=['uid','item_id','timestamp','is_organic','played_ratio_pct','track_length_seconds']
    pf=pq.ParquetFile(path); parts=[]
    for block in pf.iter_batches(batch_size=500_000,columns=columns):
        df=block.to_pandas()
        keep=df.uid.mod(uid_mod).eq(uid_remainder)
        if keep.any():parts.append(df.loc[keep])
    cohort=pd.concat(parts,ignore_index=True)
    cohort=cohort.sort_values(['uid','timestamp','item_id'],kind='mergesort').reset_index(drop=True)
    return cohort


def prepare(df,args):
    times=df.timestamp.to_numpy(dtype=np.int64)
    cutoff1=int(np.quantile(times,.8)); cutoff2=int(np.quantile(times,.9))
    period_ends=(cutoff1,cutoff2,int(times.max())+1)
    train_rows=df[df.timestamp<cutoff1]
    counts=train_rows.item_id.value_counts()
    # Track length is item metadata; use its first training-period value.
    observed_lengths=train_rows.drop_duplicates('item_id').set_index('item_id').track_length_seconds
    top=counts.head(args.catalog).index.tolist()
    mapping={int(item):j+1 for j,item in enumerate(top)}
    pop=np.zeros(len(mapping)+1,dtype=np.int64)
    lengths=np.full(len(mapping)+1,180.0,dtype=np.float32)
    for orig,j in mapping.items():
        pop[j]=int(counts.loc[orig])
        lengths[j]=float(observed_lengths.loc[orig])
    lengths=np.clip(lengths,1,3600)
    seqs={};first_pos={};records=[[],[],[]]
    for uid,g in df.groupby('uid',sort=True):
        uid=int(uid)
        ts=g.timestamp.to_numpy(dtype=np.int64)
        ids=g.item_id.to_numpy(dtype=np.int64)
        watch=np.clip(g.played_ratio_pct.to_numpy(dtype=np.float32),0,100)/100
        organic=g.is_organic.to_numpy(dtype=np.int8)
        j=np.searchsorted(ts,ts+DAY,side='left')
        future=ts[np.minimum(j,len(ts)-1)]
        returned=(j<len(ts)) & (future<=ts+7*DAY)
        seq=[];seen={};k=0
        while k<len(ts):
            end=k+1
            while end<len(ts) and ts[end]==ts[k]:end+=1
            pos=len(seq)
            for ix in range(k,end):
                item=mapping.get(int(ids[ix]))
                if not item or not pos:continue
                t=int(ts[ix]); split=0 if t<cutoff1 else 1 if t<cutoff2 else 2
                period_end=period_ends[split]
                valid=t+7*DAY<period_end
                records[split].append((uid,pos,item,float(watch[ix]),
                                       int(returned[ix]),int(organic[ix]),t,bool(valid)))
            for ix in range(k,end):
                item=mapping.get(int(ids[ix]))
                if item:
                    seen.setdefault(item,len(seq))
                    seq.append(item)
            k=end
        seqs[uid]=seq;first_pos[uid]=seen
    rng=np.random.default_rng(args.seed)
    train=records[0]
    if len(train)>args.train_cap:
        keep=rng.choice(len(train),args.train_cap,replace=False)
        train=[train[i] for i in sorted(keep)]
    val=records[1]
    if len(val)>args.eval_cap:
        keep=rng.choice(len(val),args.eval_cap,replace=False)
        val=[val[i] for i in sorted(keep)]
    test_by_user=defaultdict(list)
    for x in records[2]:test_by_user[x[0]].append(x)
    test=[]
    for uid,ev in test_by_user.items():
        if len(ev)>args.test_per_user:
            keep=rng.choice(len(ev),args.test_per_user,replace=False)
            ev=[ev[i] for i in sorted(keep)]
        test.extend(ev)
    test.sort(key=lambda x:(x[0],x[6],x[2]))
    stats={'raw_cohort_events':len(df),'users':int(df.uid.nunique()),
           'catalog':len(mapping),'cutoff_train':cutoff1,'cutoff_test':cutoff2,
           'available_examples':[len(x) for x in records],
           'selected_examples':[len(train),len(val),len(test)],
           'test_users':len(test_by_user),
           'same_uid_timestamp_extra_events':int(df.duplicated(['uid','timestamp']).sum()),
           'organic_fraction':float(df.is_organic.mean())}
    return mapping,seqs,first_pos,train,records[0],val,test,records[2],pop,lengths,stats


def batch(xs,seqs,length,device):
    h=torch.zeros((len(xs),length),dtype=torch.long)
    for row,x in enumerate(xs):
        seq=seqs[x[0]][max(0,x[1]-length):x[1]]
        h[row,-len(seq):]=torch.tensor(seq,dtype=torch.long)
    item=torch.tensor([x[2] for x in xs],dtype=torch.long)
    watch=torch.tensor([x[3] for x in xs],dtype=torch.float32)
    returned=torch.tensor([x[4] for x in xs],dtype=torch.float32)
    users=torch.tensor([x[0]//1000 for x in xs],dtype=torch.long)
    return h.to(device),item.to(device),watch.to(device),returned.to(device),users.to(device)


class FeedbackModel(nn.Module):
    def __init__(self,n_items,n_users,lengths,dim=48):
        super().__init__()
        self.emb=nn.Embedding(n_items+1,dim,padding_idx=0)
        self.user=nn.Embedding(n_users+1,16)
        self.bias=nn.Embedding(n_items+1,2)
        self.net=nn.Sequential(nn.Linear(dim*3+17,128),nn.GELU(),nn.Linear(128,2))
        self.register_buffer('length_feature',torch.tensor(np.log1p(lengths)/np.log1p(600),dtype=torch.float32))
        self.register_buffer('calibration_alpha',torch.ones(2))
        self.register_buffer('calibration_mean',torch.zeros(2))
    def forward(self,h,item,users):
        if item.ndim==1:item=item[:,None]
        ctx=self.emb(h).sum(1)/h.ne(0).sum(1).clamp(min=1)[:,None]
        it=self.emb(item);c=ctx[:,None,:].expand_as(it)
        u=self.user(users)[:,None,:].expand(item.shape[0],item.shape[1],-1)
        length=self.length_feature[item][:,:,None]
        x=torch.cat([c,it,c*it,u,length],dim=-1)
        raw=torch.sigmoid(self.net(x)+self.bias(item))
        return self.calibration_alpha*raw+(1-self.calibration_alpha)*self.calibration_mean


def iter_shuffled(xs,batch_size,rng):
    order=rng.permutation(len(xs))
    for start in range(0,len(order),batch_size):
        yield [xs[i] for i in order[start:start+batch_size]]


def auc_binary(y,p):
    y=np.asarray(y);p=np.asarray(p)
    n1=int(y.sum());n0=len(y)-n1
    if not n1 or not n0:return None
    order=np.argsort(p,kind='mergesort')
    ranks=np.empty(len(p),dtype=np.float64);ranks[order]=np.arange(1,len(p)+1)
    return float((ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0))


@torch.no_grad()
def feedback_quality(rm,xs,seqs,args,device,watch_mean,return_mean):
    rm.eval();actual_w=[];actual_r=[];pred_w=[];pred_r=[]
    for start in range(0,len(xs),args.batch):
        h,item,w,r,users=batch(xs[start:start+args.batch],seqs,args.length,device)
        pred=rm(h,item,users).squeeze(1).cpu().numpy()
        pred_w.extend(pred[:,0]);pred_r.extend(pred[:,1]);
        actual_w.extend(w.cpu().numpy());actual_r.extend(r.cpu().numpy())
    w=np.array(actual_w);r=np.array(actual_r);pw=np.array(pred_w);pr=np.array(pred_r)
    return {'n':len(xs),'watch_rmse':float(np.sqrt(np.mean((pw-w)**2))),
            'watch_mae':float(np.mean(abs(pw-w))),
            'watch_constant_rmse':float(np.sqrt(np.mean((watch_mean-w)**2))),
            'return_auc':auc_binary(r,pr),
            'return_brier':float(np.mean((pr-r)**2)),
            'return_constant_brier':float(np.mean((return_mean-r)**2)),
            'return_rate':float(r.mean())}


def fit_base(model,train,val,seqs,args,device):
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    rng=np.random.default_rng(args.seed);logs=[]
    for epoch in range(args.base_epochs):
        model.train();losses=[]
        for xs in iter_shuffled(train,args.batch,rng):
            h,item,_,_,_=batch(xs,seqs,args.length,device)
            loss=F.cross_entropy(model(h),item)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
            losses.append(float(loss.detach().item()))
        model.eval();vl=[]
        with torch.no_grad():
            for start in range(0,len(val),args.batch):
                h,item,_,_,_=batch(val[start:start+args.batch],seqs,args.length,device)
                vl.append(float(F.cross_entropy(model(h),item)))
        logs.append({'epoch':epoch+1,'train_ce':float(np.mean(losses)),
                     'val_ce':float(np.mean(vl))})
        print('base',logs[-1],flush=True)
    return logs


def fit_feedback(rm,train,val,seqs,args,device):
    reward_train=[x for x in train if x[5]==0 and x[7]]
    reward_val=[x for x in val if x[5]==0 and x[7]]
    if len(reward_train)>args.reward_cap:
        rng=np.random.default_rng(args.seed+1)
        reward_train=[reward_train[i] for i in sorted(rng.choice(len(reward_train),args.reward_cap,replace=False))]
    wm=float(np.mean([x[3] for x in reward_train]));rmn=float(np.mean([x[4] for x in reward_train]))
    opt=torch.optim.AdamW(rm.parameters(),lr=.001,weight_decay=.001)
    rng=np.random.default_rng(args.seed+2);logs=[];best_score=float('inf');best_state=None
    for epoch in range(args.reward_epochs):
        rm.train();losses=[]
        for xs in iter_shuffled(reward_train,args.batch,rng):
            h,item,w,r,users=batch(xs,seqs,args.length,device)
            pred=rm(h,item,users).squeeze(1)
            loss=F.mse_loss(pred[:,0],w)+.2*F.binary_cross_entropy(pred[:,1],r)
            opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach().item()))
        q=feedback_quality(rm,reward_val,seqs,args,device,wm,rmn)
        logs.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'val':q})
        score=q['watch_rmse']+.25*q['return_brier']
        if score<best_score:
            best_score=score;best_state=copy.deepcopy(rm.state_dict())
        print('feedback',logs[-1],flush=True)
    rm.load_state_dict(best_state)
    # Fit a conservative shrinkage to the train mean using validation only.
    rm.eval();raw=[];actual=[]
    with torch.no_grad():
        for start in range(0,len(reward_val),args.batch):
            h,item,w,r,users=batch(reward_val[start:start+args.batch],seqs,args.length,device)
            raw.extend(rm(h,item,users).squeeze(1).cpu().numpy())
            actual.extend(np.stack([w.cpu().numpy(),r.cpu().numpy()],axis=1))
    raw=np.asarray(raw);actual=np.asarray(actual)
    means=np.array([wm,rmn]);alphas=[]
    for j in range(2):
        scores=[np.mean((a*raw[:,j]+(1-a)*means[j]-actual[:,j])**2)
                for a in np.linspace(0,1,101)]
        alphas.append(float(np.argmin(scores)/100))
    rm.calibration_alpha.copy_(torch.tensor(alphas,device=device))
    rm.calibration_mean.copy_(torch.tensor(means,device=device,dtype=torch.float32))
    calibration={'alpha_watch':alphas[0],'alpha_return':alphas[1],
                 'validated':feedback_quality(rm,reward_val,seqs,args,device,wm,rmn)}
    print('calibration',calibration,flush=True)
    return logs,wm,rmn,calibration


def fit_policy(base,rm,train,seqs,first_pos,pop,args,device,
               *,horizon,beta,mixture):
    policy=copy.deepcopy(base).to(device)
    opt=torch.optim.AdamW(policy.parameters(),lr=args.rl_lr,weight_decay=.001)
    rng=np.random.default_rng(args.seed+3)
    selected=rng.choice(len(train),min(args.rl_examples,len(train)),replace=False)
    subset=[train[i] for i in selected]
    pop_tensor=torch.tensor(pop,device=device,dtype=torch.float32)
    pop_penalty=torch.log1p(pop_tensor)/torch.log1p(pop_tensor.max())
    logs=[];base.eval();rm.eval();policy.train()
    for xs in iter_shuffled(subset,args.rl_batch,rng):
        h,_,_,_,users=batch(xs,seqs,args.length,device)
        generated=[set() for _ in xs]
        logps=[];rewards=[];kls=[];repeat_track=[]
        for step in range(horizon):
            with torch.no_grad():
                ref=base(h)
                candidates=ref.topk(args.candidates,dim=1).indices
                ref_probs=F.softmax(ref,dim=1)
            logits=policy(h)
            cp=F.softmax(logits.gather(1,candidates),dim=1)
            choice=torch.multinomial(cp,1)
            action=candidates.gather(1,choice).squeeze(1)
            logp=F.log_softmax(logits.gather(1,candidates),dim=1).gather(1,choice).squeeze(1)
            kl=F.kl_div(F.log_softmax(logits,dim=1),ref_probs,reduction='none').sum(1)
            with torch.no_grad():
                feedback=rm(h,action,users).squeeze(1)
                repeated=[]
                for j,item in enumerate(action.cpu().tolist()):
                    x=xs[j]
                    prior=first_pos[x[0]].get(item,10**12)<x[1]
                    repeated.append(int(prior or item in generated[j]))
                    generated[j].add(item)
                repeat=torch.tensor(repeated,device=device,dtype=torch.float32)
                reward=feedback[:,0]
                if mixture:
                    reward=.7*feedback[:,0]+.3*feedback[:,1] \
                           -.1*repeat-.05*pop_penalty[action]
            logps.append(logp);rewards.append(reward);kls.append(kl)
            repeat_track.append(float(repeat.mean()))
            h=torch.cat([h[:,1:],action[:,None]],dim=1)
        # REINFORCE with discounted return-to-go and a per-step batch baseline.
        returns=[None]*horizon;future=torch.zeros_like(rewards[-1])
        for step in range(horizon-1,-1,-1):
            future=rewards[step]+args.gamma*future
            returns[step]=future
        terms=[]
        for logp,ret in zip(logps,returns):
            advantage=(ret-ret.mean())/(ret.std(unbiased=False)+1e-6)
            terms.append(logp*advantage.detach())
        pg=-torch.stack(terms).mean()
        kl=torch.stack(kls).mean()
        loss=pg+beta*kl
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(policy.parameters(),1);opt.step()
        logs.append((float(torch.stack(rewards).mean().detach().item()),
                     float(kl.detach().item()),float(pg.detach().item()),
                     float(np.mean(repeat_track))))
    return policy,{'horizon':horizon,'beta':beta,'mixture':mixture,
                   'mean_sampled_reward':float(np.mean([x[0] for x in logs])),
                   'mean_kl':float(np.mean([x[1] for x in logs])),
                   'mean_pg':float(np.mean([x[2] for x in logs])),
                   'mean_sampled_repeat':float(np.mean([x[3] for x in logs]))}


@torch.no_grad()
def evaluate(policy,rm,test,seqs,first_pos,pop,lengths,args,device):
    policy.eval();rm.eval();hits=[];ndcgs=[];positive=[];recommended=[]
    pred_watch=[];pred_return=[];repeat=[];item_lengths=[];popularity=[];user_ids=[];rec_hits=[]
    for start in range(0,len(test),args.batch):
        xs=test[start:start+args.batch]
        h,_,_,_,uid_tensor=batch(xs,seqs,args.length,device)
        top=policy(h).topk(10,dim=1).indices
        pred=rm(h,top,uid_tensor).cpu().numpy()
        top=top.cpu().numpy()
        for j,x in enumerate(xs):
            rec=list(map(int,top[j]))
            rank=rec.index(x[2])+1 if x[2] in rec else None
            hit=int(rank is not None)
            hits.append(hit);ndcgs.append(1/math.log2(rank+1) if rank else 0)
            positive.append(x[3]>=.8);user_ids.append(x[0])
            if x[5]==0:rec_hits.append(hit)
            recommended.extend(rec)
            pred_watch.extend(pred[j,:,0].tolist());pred_return.extend(pred[j,:,1].tolist())
            repeat.extend([int(first_pos[x[0]].get(i,10**12)<x[1]) for i in rec])
            item_lengths.extend([float(lengths[i]) for i in rec])
            popularity.extend([int(pop[i]) for i in rec])
    counts=Counter(recommended);hits=np.array(hits);ndcgs=np.array(ndcgs)
    positive=np.array(positive);users=np.array(user_ids)
    per_user_h=[];per_user_n=[]
    for uid in np.unique(users):
        ix=users==uid
        per_user_h.append(float(hits[ix].mean()))
        per_user_n.append(float(ndcgs[ix].mean()))
    metrics={'n':len(test),'users':len(set(users)),'n_positive':int(positive.sum()),
             'recall@10':float(hits.mean()),'ndcg@10':float(ndcgs.mean()),
             'positive_recall@10':float(hits[positive].mean()),
             'recommended_event_recall@10':float(np.mean(rec_hits)),
             'user_balanced_recall@10':float(np.mean(per_user_h)),
             'user_balanced_ndcg@10':float(np.mean(per_user_n)),
             'predicted_watch_fraction':float(np.mean(pred_watch)),
             'predicted_return_7d':float(np.mean(pred_return)),
             'repeat@10':float(np.mean(repeat)),
             'catalog_coverage@10':len(counts)/len(pop[1:]),
             'top10_items_share':sum(v for _,v in counts.most_common(10))/len(recommended),
             'mean_train_item_popularity':float(np.mean(popularity)),
             'mean_track_length_seconds':float(np.mean(item_lengths))}
    arrays={'users':users,'hits':hits,'ndcgs':ndcgs}
    return metrics,arrays


def paired_bootstrap(a,b,users,draws=3000,seed=2026):
    deltas=[]
    for uid in np.unique(users):
        ix=users==uid
        deltas.append(float((a[ix]-b[ix]).mean()))
    deltas=np.array(deltas)
    rng=np.random.default_rng(seed)
    sampled=deltas[rng.integers(0,len(deltas),size=(draws,len(deltas)))].mean(1)
    return {'user_balanced_difference':float(deltas.mean()),
            'ci95':list(map(float,np.quantile(sampled,[.025,.975])))}


def make_rollout_starts(test,seqs,max_users=200):
    grouped=defaultdict(list)
    for x in test:grouped[x[0]].append(x)
    starts=[]
    for uid,events in sorted(grouped.items()):
        future=[];seen_times=set()
        for x in events:
            if x[6] not in seen_times:
                future.append(x[2]);seen_times.add(x[6])
            if len(future)==5:break
        if len(future)==5:
            starts.append((uid,list(seqs[uid][:events[0][1]]),future))
        if len(starts)>=max_users:break
    return starts


def histories_tensor(histories,length,device):
    h=torch.zeros((len(histories),length),dtype=torch.long,device=device)
    for row,hist in enumerate(histories):
        recent=hist[-length:]
        h[row,-len(recent):]=torch.tensor(recent,device=device)
    return h


@torch.no_grad()
def rollout(policy,rm,starts,args,device):
    policy.eval();rm.eval()
    synth=[list(x[1]) for x in starts];real=[list(x[1]) for x in starts]
    users=torch.tensor([x[0]//1000 for x in starts],device=device,dtype=torch.long)
    watch_curve=[];return_curve=[];repeat_curve=[];teacher_overlap=[];open_loop_hit=[]
    for step in range(5):
        hs=histories_tensor(synth,args.length,device)
        hr=histories_tensor(real,args.length,device)
        top_s=policy(hs).topk(10,dim=1).indices
        top_r=policy(hr).topk(10,dim=1).indices
        action=top_s[:,0]
        feedback=rm(hs,action,users).squeeze(1).cpu().numpy()
        rows_s=top_s.cpu().tolist();rows_r=top_r.cpu().tolist()
        watch_curve.append(float(feedback[:,0].mean()))
        return_curve.append(float(feedback[:,1].mean()))
        repeat_curve.append(float(np.mean([int(int(a) in s) for a,s in zip(action.cpu(),synth)])))
        teacher_overlap.append(float(np.mean([len(set(s)&set(r))/10 for s,r in zip(rows_s,rows_r)])))
        open_loop_hit.append(float(np.mean([int(starts[i][2][step] in rec)
                                            for i,rec in enumerate(rows_s)])))
        for i,a in enumerate(action.cpu().tolist()):
            synth[i].append(a);real[i].append(starts[i][2][step])
    return {'n_users':len(starts),'predicted_watch_by_step':watch_curve,
            'predicted_return_by_step':return_curve,
            'repeat_by_step':repeat_curve,'top10_overlap_with_real_context':teacher_overlap,
            'open_loop_logged_hit@10_by_step':open_loop_hit}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,default=Path('data'))
    p.add_argument('--out-dir',type=Path,default=Path('yambda_results'))
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--catalog',type=int,default=3000)
    p.add_argument('--length',type=int,default=50)
    p.add_argument('--train-cap',type=int,default=150000)
    p.add_argument('--eval-cap',type=int,default=15000)
    p.add_argument('--test-per-user',type=int,default=20)
    p.add_argument('--reward-cap',type=int,default=300000)
    p.add_argument('--batch',type=int,default=256)
    p.add_argument('--rl-batch',type=int,default=48)
    p.add_argument('--base-epochs',type=int,default=2)
    p.add_argument('--reward-epochs',type=int,default=4)
    p.add_argument('--rl-examples',type=int,default=3000)
    p.add_argument('--candidates',type=int,default=64)
    p.add_argument('--gamma',type=float,default=.9)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--rl-lr',type=float,default=.0001)
    p.add_argument('--device',default='auto')
    args=p.parse_args();seed_all(args.seed)
    device=('mps' if torch.backends.mps.is_available() else 'cpu') if args.device=='auto' else args.device
    args.out_dir.mkdir(parents=True,exist_ok=True)
    path=get_parquet(args.data_dir/'yambda_listens.parquet')
    df=load_cohort(path)
    mapping,seqs,first_pos,train,all_train,val,test,all_test,pop,lengths,stats=prepare(df,args)
    del df
    print('data',stats,flush=True)
    base=GPTRec(len(mapping),args.length).to(device)
    rm=FeedbackModel(len(mapping),max(seqs)//1000,lengths).to(device)
    base_log=fit_base(base,train,val,seqs,args,device)
    rm_log,wm,return_mean,calibration=fit_feedback(rm,all_train,val,seqs,args,device)
    reward_test=[x for x in test if x[5]==0 and x[7]]
    result={'config':vars(args)|{'data_dir':str(args.data_dir),
                                 'out_dir':str(args.out_dir),'device':str(device)},
            'data':stats,'base_training':base_log,'feedback_training':rm_log,
            'feedback_calibration':calibration,
            'feedback_test':feedback_quality(rm,reward_test,seqs,args,device,wm,return_mean),
            'policies':{}}
    starts=make_rollout_starts(all_test,seqs)
    base_metrics,base_arrays=evaluate(base,rm,test,seqs,first_pos,pop,lengths,args,device)
    result['policies']['base']={'metrics':base_metrics,'rollout':rollout(base,rm,starts,args,device)}
    torch.save({'model':base.state_dict(),'items':mapping,'length':args.length},args.out_dir/'base.pt')
    torch.save(rm.state_dict(),args.out_dir/'feedback.pt')
    experiments=[('one_step_watch',1,.1,False),('five_watch_weak',5,.03,False),
                 ('five_watch_strong',5,.3,False),('five_mixed',5,.1,True)]
    for name,horizon,beta,mixture in experiments:
        policy,rl_log=fit_policy(base,rm,train,seqs,first_pos,pop,args,device,
                                horizon=horizon,beta=beta,mixture=mixture)
        metrics,arrays=evaluate(policy,rm,test,seqs,first_pos,pop,lengths,args,device)
        comparison={'recall@10':paired_bootstrap(arrays['hits'],base_arrays['hits'],arrays['users']),
                    'ndcg@10':paired_bootstrap(arrays['ndcgs'],base_arrays['ndcgs'],arrays['users'])}
        result['policies'][name]={'rl_training':rl_log,'metrics':metrics,
                                  'paired_bootstrap_vs_base':comparison,
                                  'rollout':rollout(policy,rm,starts,args,device)}
        torch.save(policy.state_dict(),args.out_dir/f'{name}.pt')
        print(name,metrics,flush=True)
    (args.out_dir/'metrics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print('saved',args.out_dir/'metrics.json',flush=True)


if __name__=='__main__':main()
