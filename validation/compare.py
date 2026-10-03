import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('old');p.add_argument('new');a=p.parse_args();x=np.load(a.old,allow_pickle=False);y=np.load(a.new,allow_pickle=False);criteria=json.loads(Path(__file__).with_name('criteria.json').read_text());maxima={};failed=[]
for k in x.files:
 if k.endswith('__indices'):continue
 actual=y[k]
 if k+'__indices' in x.files:actual=actual.reshape(-1)[x[k+'__indices']]
 if actual.shape!=x[k].shape:raise ValueError(f'Shape mismatch {k}')
 category='gradient_max_abs' if '_grad_' in k else 'adam_weights_max_abs' if '_weights_' in k else 'loss_max_abs' if k.endswith('_loss') else 'logits_max_abs' if k.endswith('_logits') else 'latent_max_abs'
 diff=float(np.max(abs(x[k]-actual)));maxima[category]=max(diff,maxima.get(category,0))
 if diff>criteria[category]:failed.append({'key':k,'difference':diff,'limit':criteria[category]})
print(json.dumps({'verdict':'FAIL' if failed else 'PASS','scope':'samples only for indexed large parameters, otherwise full arrays','maxima':maxima,'failed':failed},indent=2));raise SystemExit(bool(failed))
