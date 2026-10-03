import argparse,sys,types,json,zipfile
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('repo');p.add_argument('out');p.add_argument('--stochastic',action='store_true');p.add_argument('--all-examples',action='store_true');p.add_argument('--probe',action='store_true');p.add_argument('--data',default=str(Path(__file__).resolve().parents[1]/'MAIN/data'));p.add_argument('--fixtures',default=str(Path(__file__).resolve().parent/'fixtures'));p.add_argument('--rebuild-vocab',action='store_true');a=p.parse_args()
torch.set_num_threads(1)
sys.path.insert(0,str(Path(a.repo,'MAIN/model').resolve()))
sys.modules['wandb']=types.SimpleNamespace(init=lambda **k:None,log=lambda *x,**k:None,finish=lambda:None)
from smiles_vae_20231004 import SmilesVAE,make_vocab
from inter_vae_0110 import InteractionVAE
from iev2mol import CVAE
fixtures=Path(a.fixtures)
vocab=json.loads((fixtures/'vocab.json').read_text())
if a.rebuild_vocab:
 with zipfile.ZipFile(Path(a.data,'Druglike_million_canonical_no_dot_dup.smi.zip')) as z:
  name=next(n for n in z.namelist() if n.endswith('.smi'));lines=z.read(name).decode().splitlines()
  assert make_vocab([line.strip().split(' ')[0] for line in lines])==vocab
raw=list(zip(json.loads((fixtures/'drd2-smiles.json').read_text()),torch.from_numpy(np.load(fixtures/'drd2-data.npz',allow_pickle=False)['vectors'])))
print('RAW',type(raw),len(raw),type(raw[0]),len(vocab),flush=True)
config=dict(encoder_hidden_size=256,encoder_num_layers=1,bidirectional=True,encoder_dropout=.5,latent_size=128,decoder_hidden_size=512,decoder_num_layers=3,decoder_dropout=0)
state=torch.load(Path(a.repo,'MAIN/model/iev2mol_DRD2.pt'),map_location='cpu',weights_only=True)
vec_length=state['pretrained_inter_vae.dec_linear_5.weight'].shape[0]
examples=sorted(list(raw) if a.all_examples else list(raw)[:8],key=lambda pair:len(pair[0]),reverse=True)
print('DIM',vec_length,[(x[0],len(x[1])) for x in examples[:2]],flush=True)
model=CVAE('cpu',SmilesVAE(vocab,config,'cpu'),InteractionVAE('cpu',vec_length));model.load_state_dict(state)
probe={}
if a.probe:
 def capture(name):
  def hook(module,inputs,output):
   if name+'.x' in probe:return
   value=inputs[0]
   if isinstance(value,torch.nn.utils.rnn.PackedSequence):
    probe[name+'.batch_sizes']=value.batch_sizes.numpy();value=value.data
   probe[name+'.x']=value.detach().numpy().copy()
   if len(inputs)>1:probe[name+'.hx']=inputs[1].detach().numpy().copy()
   if isinstance(module,torch.nn.GRU):
    value,hidden=output;probe[name+'.hidden']=hidden.detach().numpy().copy()
    if isinstance(value,torch.nn.utils.rnn.PackedSequence):value=value.data
   else:value=output
   probe[name+'.out']=value.detach().numpy().copy()
  return hook
 for name,module in model.named_modules():
  if isinstance(module,(torch.nn.Linear,torch.nn.GRU,torch.nn.BatchNorm1d,torch.nn.Conv1d,torch.nn.ConvTranspose1d)):module.register_forward_hook(capture(name))
out={};generated={};logits=[]
hook=model.pretrained_smiles_vae.decoder_fc.register_forward_hook(lambda module,inputs,output:logits.append(output.detach().numpy().copy()))
for seed in (0,19,73):
 model.load_state_dict(state);model.zero_grad();torch.manual_seed(seed);logits.clear()
 x=[model.pretrained_smiles_vae.string2tensor(s) for s,_ in examples]
 iev=torch.stack([v for _,v in examples]).float()
 # Disable only Dropout for deterministic arithmetic isolation; training BN retained.
 model.train()
 for layer in model.modules():
  if isinstance(layer,torch.nn.Dropout) and not a.stochastic:layer.p=0
 latent,kl=model.pretrained_smiles_vae.forward_encoder(x)
 recon,condition,interkl=model.pretrained_inter_vae(iev)
 z=model.new_smiles_decoder_head(torch.cat([latent,condition],dim=1))
 losses=model.pretrained_smiles_vae.forward_decoder_eachloss(x,z);loss=sum(losses)/len(losses)
 out[f'{seed}_latent']=latent.detach().numpy();out[f'{seed}_condition']=condition.detach().numpy();out[f'{seed}_recon']=recon.detach().numpy();out[f'{seed}_z']=z.detach().numpy();out[f'{seed}_loss']=loss.detach().numpy();out[f'{seed}_logits']=np.concatenate([v.reshape(-1) for v in logits])
 loss.backward()
 for name,param in model.named_parameters():
  if param.grad is not None:out[f'{seed}_grad_{name}']=param.grad.numpy().copy()
 torch.optim.Adam(model.parameters(),lr=1e-4).step()
 for name,param in model.named_parameters():
  if param.requires_grad:out[f'{seed}_weights_{name}']=param.detach().numpy().copy()
 model.load_state_dict(state);model.eval();torch.manual_seed(seed)
 generated[str(seed)]=[]
 for example in range(3):
  _,real_condition,_=model.pretrained_inter_vae(iev[example].unsqueeze(0))
  generated[str(seed)].append(model.sample_smiles_with_IEV(real_condition[0],2,max_length=201))
 print('SEED',seed,flush=True)
np.savez_compressed(a.out,**out)
if a.probe:np.savez_compressed(Path(a.out).with_name('probe-old.npz'),**probe)
Path(a.out).with_suffix('.json').write_text(json.dumps({'vocab':vocab,'generated':generated},indent=2)+'\n');print('BENCHMARK_PASS',len(out))
