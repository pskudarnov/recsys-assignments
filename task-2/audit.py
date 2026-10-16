#!/usr/bin/env python3
"""Independently reload checkpoints, verify metrics, and bootstrap users."""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from experiment import GPTRec, RewardModel, batch, examples, get_data


def audit_policy(model, reward, test, length, batch_size, device):
    model.eval(); reward.eval()
    hits=[]; ndcgs=[]; predicted=[]; users=[]; positive=[]; recommendations=[]
    with torch.no_grad():
        for start in range(0,len(test),batch_size):
            xs=test[start:start+batch_size]
            h,y,_=batch(xs,length,device)
            scores=model(h)
            scores[:,0]=-1e9
            for j,x in enumerate(xs):
                scores[j,list(set(x[1]))]=-1e9
            top=scores.topk(10,dim=1).indices
            reward_scores=reward(h,top).mean(1)
            top_cpu=top.cpu().tolist()
            for j,x in enumerate(xs):
                rank=top_cpu[j].index(x[2])+1 if x[2] in top_cpu[j] else None
                hits.append(int(rank is not None))
                ndcgs.append(1/math.log2(rank+1) if rank else 0)
                predicted.append(1+4*float(reward_scores[j]))
                users.append(x[0]); positive.append(x[3]>=.75)
                recommendations.extend(top_cpu[j])
    counts=Counter(recommendations)
    return {'hits':np.array(hits),'ndcgs':np.array(ndcgs),
            'predicted':np.array(predicted),'users':np.array(users),
            'positive':np.array(positive),'top10_counts':counts}


def user_bootstrap(a,b,users,seed=2026,draws=5000):
    per_user=[]
    for u in sorted(set(users)):
        ix=users==u
        per_user.append(float(np.mean(a[ix]-b[ix])))
    diffs=np.array(per_user)
    rng=np.random.default_rng(seed)
    boot=diffs[rng.integers(0,len(diffs),size=(draws,len(diffs)))].mean(1)
    return {'user_balanced_difference':float(diffs.mean()),
            'ci95':list(map(float,np.quantile(boot,[.025,.975]))),
            'users':len(diffs)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,default=Path('data'))
    p.add_argument('--results-dir',type=Path,default=Path('results'))
    p.add_argument('--device',default='auto')
    args=p.parse_args()
    device=('mps' if torch.backends.mps.is_available() else 'cpu') if args.device=='auto' else args.device
    saved=json.loads((args.results_dir/'metrics.json').read_text())
    length=saved['config']['length']; batch_size=saved['config']['batch']
    rows,a,b,mapping=get_data(args.data_dir)
    train,val,test=examples(rows,a,b,mapping,length)
    assert rows[a-1][0]<rows[a][0] and rows[b-1][0]<rows[b][0]
    synthetic=[(1,1,1,5),(1,1,2,4),(2,1,3,3),(2,1,4,5),(3,1,5,4)]
    tie_examples=examples(synthetic,2,4,{i:i for i in range(1,6)},length)
    assert [x[1] for x in tie_examples[1]] == [(1,2),(1,2)]
    assert tie_examples[2][0][1] == (1,2,3,4)
    assert (len(train),len(val),len(test)) == (saved['data']['train_examples'],
                                              saved['data']['val_examples'],
                                              saved['data']['test_examples'])
    # A user/timestamp group can contain several unordered ratings. The
    # experiment must never put one of them in another's context.
    repeated_groups=sum(n>1 for n in Counter((t,u) for t,u,_,_ in rows).values())
    base_checkpoint=torch.load(args.results_dir/'base.pt',map_location='cpu',weights_only=True)
    assert base_checkpoint['items']==mapping
    rm=RewardModel(len(mapping)).to(device)
    rm.load_state_dict(torch.load(args.results_dir/'reward.pt',map_location='cpu',weights_only=True))
    audit={}
    for name in saved['policies']:
        model=GPTRec(len(mapping),length).to(device)
        state=(base_checkpoint['model'] if name=='base' else
               torch.load(args.results_dir/f'{name}.pt',map_location='cpu',weights_only=True))
        model.load_state_dict(state)
        audit[name]=audit_policy(model,rm,test,length,batch_size,device)
        x=audit[name]; m=saved['policies'][name]['metrics']
        checks={'recall@10':float(x['hits'].mean()),
                'ndcg@10':float(x['ndcgs'].mean()),
                'positive_recall@10':float(x['hits'][x['positive']].mean()),
                'predicted_rating':float(x['predicted'].mean()),
                'catalog_coverage@10':len(x['top10_counts'])/len(mapping),
                'top10_items_share':sum(v for _,v in x['top10_counts'].most_common(10)) /
                                     (len(test)*10)}
        for metric,value in checks.items():
            if abs(value-m[metric])>1e-5:
                raise AssertionError(f'{name} {metric}: reload={value}, saved={m[metric]}')
    base=audit['base']; comparisons={}
    for name,x in audit.items():
        if name=='base':continue
        assert np.array_equal(x['users'],base['users'])
        comparisons[name]={
            'recall@10':user_bootstrap(x['hits'],base['hits'],x['users']),
            'ndcg@10':user_bootstrap(x['ndcgs'],base['ndcgs'],x['users']),
            'predicted_rating':user_bootstrap(x['predicted'],base['predicted'],x['users'])}
    report={'verified':True,'strict_split':True,'same_timestamp_context_test':True,
            'same_user_same_timestamp_groups':repeated_groups,
            'test_events':len(test),'test_users':len(set(base['users'])),
            'user_cluster_bootstrap_vs_base':comparisons}
    out=args.results_dir/'audit.json';out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
