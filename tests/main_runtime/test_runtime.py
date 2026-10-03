import sys,json,os,unittest,tempfile,pickle
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'MAIN/model'))
from runtime_cli import build_model
from iev_runtime import load_tensors,load_pairs,metadata
from iev_math.activations import _Activation,exp
from iev_math import _legacy_math

class RuntimeTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(1);os.environ['IEV2MOL_OLD_COMPATIBLE']='1'
  cls.state=load_tensors(ROOT/'MAIN/model/iev2mol_DRD2.pt');cls.vocab=json.loads((ROOT/'validation/fixtures/vocab.json').read_text())
 def test_scalar_activations_against_original(self):
  with np.load(ROOT/'validation/activation-reference.npz') as old:
   x=torch.from_numpy(np.linspace(-12,12,4096,dtype='f'))
   for name in old.files:np.testing.assert_array_equal(_Activation.apply(x,name).numpy(),old[name])
 def test_dense_bidirectional_and_packed_gru_original(self):
  model=build_model(self.state,self.vocab)
  with np.load(ROOT/'validation/operator-reference.npz') as old:
   for name,module in [('pretrained_smiles_vae.encoder_gru',model.pretrained_smiles_vae.encoder_gru),('pretrained_smiles_vae.decoder_gru',model.pretrained_smiles_vae.decoder_gru)]:
    x=torch.from_numpy(old[name+'.x']);args=[]
    if name+'.batch_sizes' in old:x=torch.nn.utils.rnn.PackedSequence(x,torch.from_numpy(old[name+'.batch_sizes']))
    if name+'.hx' in old:args.append(torch.from_numpy(old[name+'.hx']))
    y,h=module(x,*args)
    if isinstance(y,torch.nn.utils.rnn.PackedSequence):y=y.data
    np.testing.assert_array_equal(y.detach().numpy(),old[name+'.out']);np.testing.assert_array_equal(h.detach().numpy(),old[name+'.hidden'])
 def test_batchnorm_2d_and_3d_original(self):
  model=build_model(self.state,self.vocab);model.train()
  with np.load(ROOT/'validation/operator-reference.npz') as old:
   for name,module in [('pretrained_inter_vae.enc_batchnorm_1',model.pretrained_inter_vae.enc_batchnorm_1),('pretrained_inter_vae.enc_batchnorm_2',model.pretrained_inter_vae.enc_batchnorm_2)]:
    x=torch.from_numpy(old[name+'.x']).requires_grad_();y=module(x);np.testing.assert_array_equal(y.detach().numpy(),old[name+'.out']);y.square().sum().backward();self.assertTrue(torch.isfinite(x.grad).all())
 def test_buffer_and_cpu_dtype_rejections(self):
  with self.assertRaises(ValueError):_legacy_math.exp(np.zeros(4,dtype='d'),np.zeros(4,dtype='f'))
  with self.assertRaises(ValueError):_legacy_math.exp(np.zeros(4,dtype='f'),np.zeros(3,dtype='f'))
  with self.assertRaises(ValueError):exp(torch.zeros(4,dtype=torch.float64))
 def test_safe_metadata_checkpoint_and_object_npz_rejection(self):
  with tempfile.TemporaryDirectory() as folder:
   path=Path(folder)/'state.pt';torch.save({'runtime':metadata(),'x':torch.ones(2)},path);self.assertEqual(load_tensors(path)['runtime']['old_compatible'],True)
   smiles=Path(folder)/'smiles.json';smiles.write_text('["CCO"]');vectors=Path(folder)/'vectors.npz';np.savez(vectors,vectors=np.array([[object()]],dtype=object))
   with self.assertRaises(ValueError):load_pairs(smiles,vectors)
 def test_untrusted_checkpoint_code_is_not_executed(self):
  with tempfile.TemporaryDirectory() as folder:
   marker=Path(folder)/'executed';expression=f"open({str(marker)!r},'w').write('unsafe')"
   class Payload:
    def __reduce__(self):return eval,(expression,)
   path=Path(folder)/'bad.pt';torch.save(Payload(),path)
   with self.assertRaises(pickle.UnpicklingError):load_tensors(path)
   self.assertFalse(marker.exists())
 def test_standard_mode_keeps_state_keys(self):
  model=build_model(self.state,self.vocab)
  os.environ['IEV2MOL_OLD_COMPATIBLE']='0'
  try:
   standard=build_model(self.state,self.vocab);self.assertEqual(set(model.state_dict()),set(standard.state_dict()));self.assertEqual(type(standard.pretrained_smiles_vae.decoder_gru),torch.nn.GRU)
  finally:os.environ['IEV2MOL_OLD_COMPATIBLE']='1'
if __name__=='__main__':unittest.main()
