"""Run the existing inference unchanged, with causal input guard and progress."""
import json
import sys
import time
from pathlib import Path
from e2d_gate_common import guarded, read_jsonl

def main():
    inp=Path(sys.argv[sys.argv.index('--input-jsonl')+1])
    if guarded(inp).name!='train_prefix_inputs_label_blind.jsonl':
        raise ValueError('Only E2-D label-blind Train prefix inference is permitted')
    allowed={'sample_id','call_id','prefix_end_turn','n_prefix_turns','model_input_text',
             'binary_target','binary_loss_mask','sample_weight','input_view','input_text_sha256',
             'is_future_free'}
    rows=read_jsonl(inp)
    for r in rows:
        assert set(r)<=allowed and r['is_future_free']
        assert r['binary_loss_mask']==0 and r['sample_weight']==0
    import text_model.inference.run_longdoc_inference as inference
    Original=inference.DataLoader
    class ProgressLoader(Original):
        def __iter__(self):
            start=time.perf_counter(); done=0
            for batch in super().__iter__():
                yield batch
                done+=len(batch['sample_id'])
                if done%100==0 or done==len(rows):
                    seconds=time.perf_counter()-start
                    print(json.dumps({'phase':'frozen_M1_train_features','done':done,'total':len(rows),
                                      'seconds':round(seconds,1),
                                      'estimated_remaining_seconds':round(seconds/max(done,1)*(len(rows)-done),1)}),flush=True)
    inference.DataLoader=ProgressLoader
    return inference.main()

if __name__=='__main__':
    raise SystemExit(main())
