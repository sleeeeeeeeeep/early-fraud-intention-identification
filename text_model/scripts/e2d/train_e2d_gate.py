"""Train an independent CPU gate. Only fixed M1 Train predictions are inputs."""
import argparse
import hashlib
from collections import defaultdict, Counter
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from e2d_gate_common import (FEATURE_NAMES, M1_SHA256, GateMLP, guarded, read_jsonl,
                             pair_features, sha256_file, write_json)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--prediction-dir',type=Path,required=True)
    ap.add_argument('--preregistration',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args()
    root=guarded(args.data_dir)
    if root.name != 'v365_e2d_gate_train_s202':
        raise ValueError('Non-allowlisted Train directory')
    out=guarded(args.output_dir)
    if out.exists():
        raise ValueError('Refusing to overwrite gate training')
    prereg=json.loads(guarded(args.preregistration).read_text(encoding='utf-8'))
    cfg=prereg['gate_training']
    assert cfg == {'seed':202,'epochs':20,'batch_size':64,'learning_rate':0.001,
                   'weight_decay':0.0001,'hidden_size':16,'initial_pass_probability':0.9,
                   'gradient_clip_norm':1.0,'device':'cpu'}
    pairs=read_jsonl(root/'train_pair_labels.jsonl')
    inputs=read_jsonl(root/'train_prefix_inputs_label_blind.jsonl')
    data_manifest=json.loads(guarded(root/'input_manifest.json').read_text(encoding='utf-8'))
    assert sha256_file(root/'train_pair_labels.jsonl') == data_manifest['outputs']['pairs']['sha256']
    assert sha256_file(root/'train_prefix_inputs_label_blind.jsonl') == data_manifest['outputs']['prefixes']['sha256']
    rows=read_jsonl(args.prediction_dir/'predictions.jsonl')
    met=json.loads(guarded(args.prediction_dir/'metrics.json').read_text(encoding='utf-8'))
    assert not met['checkpoint']['missing'] and not met['checkpoint']['unexpected']
    assert Path(met['checkpoint']['checkpoint']).resolve() == Path(prereg['m1_checkpoint']).resolve()
    assert Path(met['config']['input_jsonl']).resolve() == (root/'train_prefix_inputs_label_blind.jsonl').resolve()
    for k,v in prereg['inference'].items():
        if k != 'max_chunks':
            assert met['config'][k] == v, (k,met['config'].get(k),v)
    preds={r['sample_id']:r for r in rows}
    assert len(preds)==len(rows)==len(inputs) and set(preds)=={r['sample_id'] for r in inputs}
    assert len(pairs)==4633 and len(inputs)==8258
    assert len({p['pair_id'] for p in pairs})==len(pairs)
    for row in rows:
        assert float(row['full_text_coverage'])==1.0
    features=[]
    groups=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for i,p in enumerate(pairs):
        assert p['split']=='train' and p['decision']!='ambiguous'
        assert p['call_id'].startswith('V365_TRAIN_')
        assert p['current_turn']==p['previous_turn']+1
        assert p['target']==int(p['group']=='fraud_post_positive')
        if p['decision']=='normal':
            assert p['group']=='normal_negative'
        elif p['target']==0:
            assert p['current_turn']<p['tE_turn']
        else:
            assert p['current_turn']>=p['tE_turn']
        features.append(pair_features(preds[p['previous_sample_id']],preds[p['current_sample_id']]))
        groups[p['group']][p['scenario_tag']][p['call_id']].append(i)
    masses={'normal_negative':0.25,'fraud_pre_negative':0.25,'fraud_post_positive':0.5}
    assert set(groups)==set(masses)
    w=np.zeros(len(pairs),dtype=np.float64)
    for name,scenarios in groups.items():
        for calls in scenarios.values():
            for indices in calls.values():
                mass=masses[name]/len(scenarios)/len(calls)/len(indices)
                w[indices]=mass
    assert abs(w.sum()-1)<1e-9
    seed=cfg['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    x=np.asarray(features,dtype=np.float64)
    mean=np.average(x,axis=0,weights=w)
    scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=w))
    scale[scale<1e-6]=1.0
    z=torch.tensor(np.clip((x-mean)/scale,-8,8),dtype=torch.float32)
    y=torch.tensor([p['target'] for p in pairs],dtype=torch.float32)
    # Scale weights to mean 1; minibatch estimates preserve the preregistered masses.
    weights=torch.tensor(w*len(pairs),dtype=torch.float32)
    model=GateMLP()
    initial={k:v.clone() for k,v in model.state_dict().items()}
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    lossfn=torch.nn.BCEWithLogitsLoss(reduction='none')
    started=time.perf_counter(); history=[]
    for epoch in range(cfg['epochs']):
        model.train(); order=torch.randperm(len(pairs)); total=0.0
        for idx in order.split(cfg['batch_size']):
            opt.zero_grad(set_to_none=True)
            loss=(lossfn(model(z[idx]),y[idx])*weights[idx]).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm']); opt.step()
            total+=loss.item()*len(idx)
        history.append({'epoch':epoch+1,'weighted_bce':total/len(pairs)})
        print(json.dumps(history[-1]),flush=True)
    model.eval()
    with torch.no_grad():
        final_probs=model(z).sigmoid().tolist()
    group_diagnostics={}
    for group in masses:
        ix=[i for i,p in enumerate(pairs) if p['group']==group]
        candidates=[i for i in ix if min(float(preds[pairs[i]['previous_sample_id']]['p_binary'][1]),
                                          float(preds[pairs[i]['current_sample_id']]['p_binary'][1]))>=0.5]
        group_diagnostics[group]={'pairs':len(ix),'m1_raw_k2_candidates':len(candidates),
                                  'gate_pass_pairs':sum(final_probs[i]>=0.5 for i in ix),
                                  'candidate_vetoes':sum(final_probs[i]<0.5 for i in candidates),
                                  'candidate_passes':sum(final_probs[i]>=0.5 for i in candidates)}
    changed=sum(not torch.equal(initial[k],v) for k,v in model.state_dict().items())
    assert changed>0
    out.mkdir(parents=True)
    ckpt=out/'temporal_gate.pt'
    payload={'schema_version':'v365.e2d.independent_gate.v1','state_dict':model.state_dict(),
             'feature_names':FEATURE_NAMES,'mean':mean.tolist(),'scale':scale.tolist(),
             'm1_sha256':M1_SHA256,'training_used_dev':False,'training_used_test':False,
             'config':cfg,'preregistration_sha256':sha256_file(args.preregistration),
             'train_pair_sha256':sha256_file(root/'train_pair_labels.jsonl'),
             'train_prediction_sha256':sha256_file(args.prediction_dir/'predictions.jsonl'),
             'training_data_scope':'v365_train_only','scaler_fit_scope':'train_only',
             'gate_common_sha256':sha256_file(Path(__file__).with_name('e2d_gate_common.py')),
             'trainer_sha256':sha256_file(Path(__file__)),
             'dev_rows_used_for_training':0,'test_or_restricted_files_opened':0,
             'train_pair_rows':len(pairs),'train_call_ids_sha256':hashlib.sha256(
                 ('\n'.join(sorted({p['call_id'] for p in pairs}))+'\n').encode()).hexdigest()}
    tmp=ckpt.with_suffix('.pt.tmp'); torch.save(payload,tmp); tmp.replace(ckpt)
    manifest={k:v for k,v in payload.items() if k not in ('state_dict','mean','scale')}
    manifest.update({'status':'complete','selection_policy':'fixed final epoch; no Dev selection',
                     'parameters':sum(p.numel() for p in model.parameters()),'changed_tensors':changed,
                     'pairs':len(pairs),'unique_calls':len({p['call_id'] for p in pairs}),
                     'group_counts':dict(Counter(p['group'] for p in pairs)),
                     'group_weight_masses':masses,'history':history,
                     'group_training_diagnostics':group_diagnostics,
                     'train_seconds':time.perf_counter()-started,'checkpoint_sha256':sha256_file(ckpt),
                     'm1_parameters_loaded_in_trainer':0,'fulltext_model_updated':False,
                     'test_or_restricted_files_opened':0})
    write_json(out/'training_manifest.json',manifest)
    print(json.dumps({'status':'complete','pairs':len(pairs),'parameters':manifest['parameters']}),flush=True)

if __name__=='__main__':
    main()
