"""One-time conversion of the fixed author-owned dataset in the legacy env."""
from pathlib import Path
import argparse,hashlib,sys,json
import torch,numpy as np
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',required=True)
parser.add_argument('--reference-model-dir',required=True)
parser.add_argument('--output',required=True)
args=parser.parse_args()
source=Path(args.source)
output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
expected='699eb9d47882917043a27e62b9269ee527ed08b5ca1f8d7c214d2e1bb29c5a00'
if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:raise ValueError('Source hash mismatch')
sys.path.insert(0,str(Path(args.reference_model_dir).resolve()))
raw=torch.load(source,map_location='cpu',weights_only=False)
rows=[raw[i] for i in range(len(raw))]
if not all(isinstance(s,str) and isinstance(v,torch.Tensor) for s,v in rows):raise ValueError('Unexpected dataset schema')
np.savez_compressed(output/'drd2-data.npz',vectors=np.stack([v.cpu().numpy() for s,v in rows]))
(output/'drd2-smiles.json').write_text(json.dumps([s for s,v in rows])+'\n')
print('CONVERTED',len(rows),rows[0][1].shape)
