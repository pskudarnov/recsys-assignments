#!/usr/bin/env python3
"""Independent reload and protocol checks for the Yambda experiment."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiment import GPTRec
from yambda_experiment import (DAY, FeedbackModel, evaluate, feedback_quality,
                               get_parquet, load_cohort, prepare, seed_all)


def popularity_baseline(test, pop, first_pos):
    top=np.argsort(-pop[1:], kind='stable')[:10]+1
    hits=[];ndcg=[];repeats=[]
    for x in test:
        ix=np.flatnonzero(top==x[2])
        hits.append(bool(len(ix)))
        ndcg.append(1/math.log2(int(ix[0])+2) if len(ix) else 0)
        repeats.extend(first_pos[x[0]].get(int(item),10**12)<x[1] for item in top)
    return {'recall@10':float(np.mean(hits)),
            'ndcg@10':float(np.mean(ndcg)),
            'repeat@10':float(np.mean(repeats)),
            'catalog_coverage@10':10/len(pop[1:])}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',type=Path,default=Path('data'))
    ap.add_argument('--results-dir',type=Path,default=Path('yambda_results'))
    ap.add_argument('--device',default='cpu')
    args=ap.parse_args()
    saved=json.loads((args.results_dir/'metrics.json').read_text())
    cfg=argparse.Namespace(**saved['config'])
    seed_all(cfg.seed)
    df=load_cohort(get_parquet(args.data_dir/'yambda_listens.parquet'))
    max_time=int(df.timestamp.max())
    mapping,seqs,first_pos,train,all_train,val,test,all_test,pop,lengths,stats=prepare(df,cfg)
    mapped=df[df.item_id.isin(mapping)]
    position_times={int(uid):g.timestamp.to_numpy()
                    for uid,g in mapped.groupby('uid',sort=False)}
    strict_histories=all(position_times[x[0]][x[1]-1]<x[6]
                         for examples in (train,val,test) for x in examples)
    assert strict_histories, 'A history contains an event at or after its target timestamp'
    del df,all_train,all_test
    assert stats==saved['data'], 'Data preparation differs from saved result'
    assert all(x[6]<stats['cutoff_train'] for x in train)
    assert all(stats['cutoff_train']<=x[6]<stats['cutoff_test'] for x in val)
    # The target must be later than every input token. prepare() groups tied
    # timestamps before assigning a common history position.
    assert all(0<x[1]<=len(seqs[x[0]]) and x[2]>0 for x in test)
    assert all(x[6]>=stats['cutoff_test'] for x in test)
    valid_horizon=all((not x[7]) or x[6]+7*DAY<end
                      for examples,end in ((train,stats['cutoff_train']),
                                           (val,stats['cutoff_test']),
                                           (test,max_time+1)) for x in examples)
    assert valid_horizon
    base_blob=torch.load(args.results_dir/'base.pt',map_location=args.device,
                         weights_only=False)
    assert mapping==base_blob['items']
    base=GPTRec(len(mapping),cfg.length).to(args.device)
    base.load_state_dict(base_blob['model'])
    rm=FeedbackModel(len(mapping),max(seqs)//1000,lengths).to(args.device)
    rm.load_state_dict(torch.load(args.results_dir/'feedback.pt',
                                  map_location=args.device,weights_only=True))
    reward_test=[x for x in test if x[5]==0 and x[7]]
    wm=float(rm.calibration_mean[0]);return_mean=float(rm.calibration_mean[1])
    quality=feedback_quality(rm,reward_test,seqs,cfg,args.device,wm,return_mean)
    differences={k:quality[k]-saved['feedback_test'][k]
                 for k in ('watch_rmse','return_auc','return_brier')}
    checks={}
    for name,record in saved['policies'].items():
        if name=='base':policy=base
        else:
            policy=GPTRec(len(mapping),cfg.length).to(args.device)
            policy.load_state_dict(torch.load(args.results_dir/f'{name}.pt',
                                               map_location=args.device,weights_only=True))
        fresh,_=evaluate(policy,rm,test,seqs,first_pos,pop,lengths,cfg,args.device)
        for key in ('n','users','recall@10','ndcg@10','repeat@10',
                    'catalog_coverage@10','predicted_watch_fraction'):
            assert abs(fresh[key]-record['metrics'][key])<1e-5, (name,key,fresh[key])
        checks[name]={k:fresh[k] for k in ('recall@10','ndcg@10',
                                           'predicted_watch_fraction','repeat@10')}
    assert max(map(abs,differences.values()))<1e-5,differences
    result={'checkpoint_metrics_match':True,'data_stats_match':True,
            'strict_earlier_histories':bool(strict_histories),
            'test_after_cutoff':True,'valid_horizon_sanity':bool(valid_horizon),
            'feedback_test':quality,'feedback_numeric_differences':differences,
            'popularity_baseline':popularity_baseline(test,pop,first_pos),
            'policies':checks}
    (args.results_dir/'audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
