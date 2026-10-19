"""Offline loader and binary-only public API for the frozen M1 delivery."""
from pathlib import Path
import json
import hashlib
import os
import sys
import time

# The delivery never downloads models or uses optional TensorFlow/Flax backends.
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['USE_TF']='0'
os.environ['USE_FLAX']='0'

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from text_model.data.longdoc_chunker import ChunkConfig
from text_model.data.longdoc_collator import LongDocCollator
from text_model.models.longdoc_multitask_classifier import LongDocMultiTaskClassifier

PACKAGE_ROOT=Path(__file__).resolve().parents[1]

def normalize_record(record, fallback_id=None):
    """Labels and other metadata are never forwarded to the model."""
    if not isinstance(record,dict):
        raise ValueError('Each input line must be a JSON object')
    sid=record.get('sample_id',fallback_id)
    if not isinstance(sid,str) or not sid.strip():
        raise ValueError('sample_id must be a non-empty string')
    texts=[record[k] for k in ('text','model_input_text','transcript') if k in record]
    if texts and 'turns' in record:
        raise ValueError('Use one text field or turns, not both')
    if texts:
        if any(not isinstance(t,str) for t in texts) or len(set(texts))!=1:
            raise ValueError('Text fields must be strings; multiple text fields must agree')
        text=texts[0]
    elif 'turns' in record:
        turns=record['turns']
        if not isinstance(turns,list) or not turns:
            raise ValueError('turns must be a non-empty list')
        pieces=[]
        for i,t in enumerate(turns,1):
            if not isinstance(t,dict) or not isinstance(t.get('text'),str) or not t['text'].strip():
                raise ValueError('Each turn requires non-empty text')
            if 'turn_id' in t and t['turn_id']!=i:
                raise ValueError('turn_id, when present, must be sequential starting at 1')
            if not isinstance(t.get('speaker'),str) or not t['speaker'].strip():
                raise ValueError('Each turn requires a speaker string')
            pieces.append('{:03d} {}：{}'.format(i,t['speaker'].strip(),t['text'].strip()))
        text='\n'.join(pieces)
    else:
        raise ValueError('Input requires text, model_input_text, transcript, or turns')
    if not text.strip():
        raise ValueError('Empty text is not supported')
    return {'sample_id':sid,'model_input_text':text,'label_13':0,'sample_weight':0.0,
            'binary_loss_mask':0.0,'disable_window_evidence':True}

class FraudDetector:
    def __init__(self, device='cuda:0', encoder_micro_batch_size=12, package_root=None):
        self.root=Path(package_root or PACKAGE_ROOT).resolve()
        self.config=json.loads((self.root/'config/inference.json').read_text(encoding='utf-8'))
        c=self.config
        fixed={'threshold':0.5,'route':'B','chunk_length':320,'overlap':64,
               'head_context_length':96,'input_mode':'head_context_enhanced_chunks',
               'diagnostic_max_chunks':12,'hard_chunk_cap':None,'hard_last':True,
               'precision':'float32','projection_dim':512,'fusion_hidden':512}
        if any(c.get(k)!=v for k,v in fixed.items()):
            raise ValueError('Release inference contract differs from validated M1 configuration')
        if c['evidence_components']!=['attention','top1','top2','last','mean']:
            raise ValueError('Release evidence configuration mismatch')
        self.device=torch.device(device)
        if self.device.type=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; install a CUDA PyTorch build or choose --device cpu')
        if encoder_micro_batch_size<1:
            raise ValueError('encoder_micro_batch_size must be positive')
        manifest_path=self.root/'manifest.json'
        if not manifest_path.exists():
            raise ValueError('Unsealed delivery: manifest.json is required')
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        entries=[entry for entry in manifest['files'] if entry['path']==c['weights']]
        if len(entries)!=1:
            raise ValueError('Weight identity missing from package manifest')
        weight_path=self.root/c['weights']
        entry=entries[0]
        if weight_path.stat().st_size!=entry['bytes']:
            raise ValueError('Weight file size mismatch')
        digest=hashlib.sha256()
        with weight_path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):
                digest.update(block)
        if digest.hexdigest()!=entry['sha256']:
            raise ValueError('Weight SHA256 mismatch')
        self.tokenizer=AutoTokenizer.from_pretrained(str(self.root/c['tokenizer']),local_files_only=True)
        cfg=AutoConfig.from_pretrained(str(self.root/c['encoder_config']),local_files_only=True)
        # All random initial values are overwritten by the complete final state.
        backbone=AutoModel.from_config(cfg)
        self.model=LongDocMultiTaskClassifier(backbone,route=c['route'],
            projection_dim=c['projection_dim'],fusion_hidden=c['fusion_hidden'],
            dropout=c['dropout'],encoder_micro_batch_size=encoder_micro_batch_size,
            evidence_components=c['evidence_components'])
        state=torch.load(weight_path,map_location='cpu',weights_only=True)
        if not isinstance(state,dict) or not state or not all(torch.is_tensor(v) for v in state.values()):
            raise ValueError('Delivery weight file must contain only a tensor state dictionary')
        report=self.model.load_state_dict(state,strict=True)
        self.load_report={'strict':True,'missing_keys':list(report.missing_keys),
                          'unexpected_keys':list(report.unexpected_keys),
                          'parameter_tensors':len(state)}
        del state
        self.model.requires_grad_(False).to(self.device).eval()
        self.collator=LongDocCollator(self.tokenizer,ChunkConfig(
            chunk_length=c['chunk_length'],overlap=c['overlap'],input_mode=c['input_mode'],
            head_context_length=c['head_context_length'],max_chunks_per_call=c['diagnostic_max_chunks']))

    @torch.inference_mode()
    def predict_records(self, records):
        rows=[normalize_record(r) for r in records]
        if not rows:
            return []
        if len({r['sample_id'] for r in rows})!=len(rows):
            raise ValueError('Duplicate sample_id within a batch')
        batch=self.collator(rows)
        tensors={k:v.to(self.device) if torch.is_tensor(v) else v for k,v in batch.items()}
        if self.device.type=='cuda':
            torch.cuda.synchronize(self.device)
        started=time.perf_counter()
        result=self.model(input_ids=tensors['input_ids'],attention_mask=tensors['attention_mask'],
            token_type_ids=tensors['token_type_ids'],chunk_mask=tensors['chunk_mask'],
            chunk_positions=tensors['chunk_positions'],global_length_features=tensors['global_length_features'],
            hard_last=True)
        probs=result.binary_logits.float().softmax(dim=-1).cpu().tolist()
        if self.device.type=='cuda':
            torch.cuda.synchronize(self.device)
        elapsed=(time.perf_counter()-started)*1000/len(rows)
        predictions=[]
        for row,meta,p in zip(rows,batch['metadata'],probs):
            if abs(float(meta['coverage_ratio'])-1.0)>1e-9:
                raise RuntimeError('Incomplete input coverage')
            pred=int(p[1]>=self.config['threshold'])
            predictions.append({'sample_id':row['sample_id'],'prediction_binary':pred,
                'label':self.config['labels'][str(pred)],'p_normal':p[0],'p_fraud':p[1],
                'threshold':self.config['threshold'],'raw_wordpiece_length':meta['raw_wordpiece_length'],
                'num_chunks':meta['num_chunks'],'coverage':meta['coverage_ratio'],
                'diagnostic_max_chunks_exceeded':meta['max_chunks_exceeded'],
                'model_version':self.config['model_version'],'elapsed_ms':elapsed})
        return predictions

    def predict_text(self, text, sample_id='request'):
        return self.predict_records([{'sample_id':sample_id,'text':text}])[0]
