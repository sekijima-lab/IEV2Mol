"""GRU recurrence following PyTorch 1.13.1 CPU arithmetic with scalar libm."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence
from .activations import _Activation

class CompatibleGRU(nn.GRU):
    def forward(self, input, hx=None):
        packed=isinstance(input,PackedSequence)
        x=input.data if packed else (input.transpose(0,1) if self.batch_first else input)
        if x.device.type!='cpu' or x.dtype!=torch.float32 or x.ndim!=(2 if packed else 3):
            raise ValueError('Historical GRU requires batched CPU float32 input')
        sizes=input.batch_sizes.tolist() if packed else [x.shape[1]]*x.shape[0]
        if not sizes:raise ValueError('Empty sequence')
        count=2 if self.bidirectional else 1
        hidden=torch.zeros(self.num_layers*count,sizes[0],self.hidden_size,dtype=x.dtype) if hx is None else hx
        if packed and input.sorted_indices is not None:
            hidden=hidden.index_select(1,input.sorted_indices)
        finals=[]
        for layer in range(self.num_layers):
            directions=[]
            for direction in range(count):
                suffix=f'_l{layer}'+('_reverse' if direction else '')
                wi=getattr(self,'weight_ih'+suffix);wh=getattr(self,'weight_hh'+suffix)
                bi=getattr(self,'bias_ih'+suffix) if self.bias else None
                bh=getattr(self,'bias_hh'+suffix) if self.bias else None
                gates=F.linear(x,wi,bi)
                steps=list(gates.split(sizes,dim=0)) if packed else list(gates.unbind(0))
                state=hidden[layer*count+direction]
                outputs=[None]*len(sizes)
                order=range(len(sizes)-1,-1,-1) if direction else range(len(sizes))
                for time in order:
                    n=sizes[time];h=state[:n]
                    ir,iz,inew=steps[time].chunk(3,dim=1)
                    hr,hz,hnew=F.linear(h,wh,bh).chunk(3,dim=1)
                    reset=_Activation.apply(hr+ir,'sigmoid')
                    update=_Activation.apply(hz+iz,'sigmoid')
                    candidate=_Activation.apply(inew+hnew*reset,'tanh')
                    new=(h-candidate)*update+candidate
                    outputs[time]=new
                    state=torch.cat((new,state[n:]),dim=0)
                directions.append(torch.cat(outputs,dim=0) if packed else torch.stack(outputs,dim=0))
                finals.append(state)
            x=torch.cat(directions,dim=-1)
            if self.dropout and self.training and layer<self.num_layers-1:
                x=F.dropout(x,p=self.dropout,training=True)
        h=torch.stack(finals)
        if packed:
            if input.unsorted_indices is not None:h=h.index_select(1,input.unsorted_indices)
            return PackedSequence(x,input.batch_sizes,input.sorted_indices,input.unsorted_indices),h
        return (x.transpose(0,1) if self.batch_first else x),h
