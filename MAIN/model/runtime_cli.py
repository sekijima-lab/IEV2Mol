"""CPU generation and small-scale fine-tuning using numeric inputs and safe state."""
import argparse,json,os
from pathlib import Path
import torch
from iev2mol import CVAE
from inter_vae_0110 import InteractionVAE
from smiles_vae_20231004 import SmilesVAE
from iev_runtime import load_pairs,load_tensors,metadata

CONFIG=dict(encoder_hidden_size=256,encoder_num_layers=1,bidirectional=True,
            encoder_dropout=.5,latent_size=128,decoder_hidden_size=512,
            decoder_num_layers=3,decoder_dropout=0)

def build_model(state, vocab):
    length=state['pretrained_inter_vae.dec_linear_5.weight'].shape[0]
    model=CVAE('cpu',SmilesVAE(vocab,dict(CONFIG),'cpu'),InteractionVAE('cpu',length))
    model.load_state_dict(state)
    return model

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--no-old-compatible',action='store_true')
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--vocab',required=True)
    parser.add_argument('--smiles',required=True)
    parser.add_argument('--vectors',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--epochs',type=int,default=0)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--max-length',type=int,default=201)
    parser.add_argument('--samples',type=int,default=2)
    parser.add_argument('--lr',type=float,default=1e-4)
    args=parser.parse_args()
    if args.epochs<0 or args.batch_size<2 or args.samples<1 or args.max_length<1:
        parser.error('epochs>=0, batch-size>=2, samples>=1, max-length>=1 required')
    os.environ['IEV2MOL_OLD_COMPATIBLE']='0' if args.no_old_compatible else '1'
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    vocab=json.loads(Path(args.vocab).read_text())
    loaded=load_tensors(args.checkpoint)
    checkpoint=loaded if 'model_state_dict' in loaded else {'model_state_dict':loaded,'epoch':0,'steps':0}
    if 'runtime' in checkpoint and checkpoint['runtime'].get('old_compatible',not args.no_old_compatible)!=(not args.no_old_compatible):
        raise ValueError('Resume with the same old-compatible mode as the saved checkpoint')
    model=build_model(checkpoint['model_state_dict'],vocab)
    torch.manual_seed(args.seed)
    pairs=load_pairs(args.smiles,args.vectors)
    if any(vector.numel()!=model.pretrained_inter_vae.vec_length for _,vector in pairs):
        raise ValueError('IEV dimensions do not match checkpoint')
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'rng_state' in checkpoint:
        torch.set_rng_state(checkpoint['rng_state'])
    steps=checkpoint.get('steps',0)
    losses=[]
    # Keep one final batch with >=2 examples for training BatchNorm.
    batches=[pairs[i:i+args.batch_size] for i in range(0,len(pairs),args.batch_size)]
    if len(batches)>1 and len(batches[-1])==1:
        batches[-2].extend(batches.pop())
    if args.epochs and len(pairs)<2:
        raise ValueError('Training needs at least two examples')
    for _ in range(args.epochs):
        for batch in batches:
            batch=sorted(batch,key=lambda pair:len(pair[0]),reverse=True)
            tokens=[model.pretrained_smiles_vae.string2tensor(s) for s,_ in batch]
            vectors=torch.stack([v for _,v in batch])
            optimizer.zero_grad()
            loss=model((tokens,vectors))
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();optimizer.step();steps+=1;losses.append(float(loss.detach()))
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    saved={'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
           'rng_state':torch.get_rng_state(),'epoch':checkpoint.get('epoch',0)+args.epochs,
           'steps':steps,'seed':checkpoint.get('seed',args.seed),'runtime':metadata()}
    torch.save(saved,output/'checkpoint.pt')
    # Safe reload is required before announcing success.
    verified=load_tensors(output/'checkpoint.pt')
    if verified['steps']!=steps:raise ValueError('Checkpoint reload mismatch')
    model.eval()
    with torch.no_grad():
        _,condition,_=model.pretrained_inter_vae(pairs[0][1].unsqueeze(0))
        generated=model.sample_smiles_with_IEV(condition[0],args.samples,args.max_length)
    result={'seed':saved['seed'],'runtime':metadata(),'epoch':saved['epoch'],'steps':steps,'losses':losses,'generated':generated}
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
