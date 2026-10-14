"""Exercise candidate class storage guards and the installed CUDA kernels."""
import ast
import json
import os
from pathlib import Path
import torch
from torch import nn
import comfy_kitchen as ck

torch.manual_seed(17)
torch.set_grad_enabled(False)
Method=type('ConvRotInt8ComfyKitchenLinearMethod',(),{})
class ColumnParallelLinear(nn.Module):
    def __init__(self,ind,outd,**kwargs):
        super().__init__()
        self.tp_size=1
        self.bias=None
        self.weight=nn.Parameter(torch.randint(-100,100,(outd,ind),dtype=torch.int8,device='cuda'),requires_grad=False)
        self.weight_scale=nn.Parameter(torch.full((outd,1),0.0001,device='cuda'),requires_grad=False)
        self.quant_method=Method()
        self.quant_method.is_checkpoint_serialized=True
        self.quant_method.group_size=256
    def forward(self,x):
        return ck.int8_linear(x,self.weight,self.weight_scale,convrot=True,convrot_groupsize=256,out_dtype=x.dtype),None
class RowParallelLinear(ColumnParallelLinear):pass
class Gate:
    def can_attempt_once(self):return False
_SILU_MUL_FUSION=Gate()
_is_cuda=True
model_file = Path(__file__).resolve().parents[2] / 'python/sglang/multimodal_gen/runtime/models/dits/qwen_image21.py'
model_tree = ast.parse(model_file.read_text())
model_class = next(n for n in model_tree.body if isinstance(n, ast.ClassDef) and n.name == 'QwenImage21FeedForward')
exec(compile(ast.Module(body=[model_class], type_ignores=[]), str(model_file), 'exec'))
os.environ['SGLANG_QWEN21_FUSED_INT8_MLP']='1'
model=QwenImage21FeedForward(4096,3,None,'micro')
model.pack_int8_gate_up()
assert model._int8_gate_up_packed
rows=[]
for shape in [(1,17,4096),(1,4096,4096)]:
    x=torch.randn(shape,dtype=torch.bfloat16,device='cuda')
    fused=model(x)
    w=torch.cat((model.gate_layer.weight,model.proj.weight),0)
    scales=torch.cat((model.gate_layer.weight_scale,model.proj.weight_scale),0)
    gate_up=ck.int8_linear(x,w,scales,convrot=True,convrot_groupsize=256,out_dtype=x.dtype)
    official=ck.int8_linear(gate_up,model.out.weight,model.out.weight_scale,convrot=True,convrot_groupsize=256,out_dtype=x.dtype,input_act='swiglu')
    assert torch.equal(fused,official), 'packed implementation differs from official kernel sequence'
    model._int8_gate_up_packed=False
    old=model(x)
    model._int8_gate_up_packed=True
    rows.append({'shape':shape,'finite':bool(torch.isfinite(fused).all()),'official_bit_equal':True,'old_cos':float(torch.nn.functional.cosine_similarity(fused.float().flatten(),old.float().flatten(),dim=0)),'old_max_abs':float((fused.float()-old.float()).abs().max())})
# A parameter replacement must invalidate the packed view and use ordinary forwards.
model.proj.weight=nn.Parameter(model.proj.weight.clone(),requires_grad=False)
assert model._fused_int8_forward(x) is None
model.gate_layer.tp_size=2
model._int8_gate_up_packed=False
model.pack_int8_gate_up()
assert not model._int8_gate_up_packed
print(json.dumps({'checks':rows,'storage_replacement_fallback':True,'tp2_not_enabled':True}))
os.environ.pop('SGLANG_QWEN21_FUSED_INT8_MLP',None)
os.environ['SGLANG_QWEN21_PACKED_BF16_MLP']='1'
UnquantizedLinearMethod=type('UnquantizedLinearMethod',(),{})
model=QwenImage21FeedForward(4096,3,None,'micro-bf16')
for layer in (model.gate_layer,model.proj,model.out):
    layer.weight=nn.Parameter(torch.randn(layer.weight.shape,device='cuda',dtype=torch.bfloat16)*0.01,requires_grad=False)
    layer.quant_method=UnquantizedLinearMethod()
    layer.forward=lambda x,layer=layer:(nn.functional.linear(x,layer.weight),None)
model.pack_int8_gate_up()
assert model._bf16_gate_up_packed
x=torch.randn((1,1024,4096),device='cuda',dtype=torch.bfloat16)
y=model(x)
p=nn.functional.linear(x,torch.cat((model.gate_layer.weight,model.proj.weight),0))
g,v=p.chunk(2,dim=-1)
reference=model.out(nn.functional.silu(g)*v)[0]
assert torch.equal(y,reference),'BF16 packed activation is not bit exact'
print(json.dumps({'bf16_official_sequence_bit_equal':True,'finite':bool(torch.isfinite(y).all())}))

# Run the committed numerical regression assertions without requiring pytest.
test_file = Path(__file__).resolve().parents[2] / 'python/sglang/multimodal_gen/test/unit/test_h3_fp32_gate.py'
tree = ast.parse(test_file.read_text())
body = []
for node in tree.body:
    if isinstance(node, ast.Import) and any(a.name == 'pytest' for a in node.names):
        continue
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'pytestmark' for t in node.targets):
        continue
    if isinstance(node, ast.FunctionDef):
        node.decorator_list = []
    body.append(node)
tree.body = body
scope = {}
exec(compile(tree, str(test_file), 'exec'), scope)
for rows, hidden in [(0,5376),(17,5376),(15488,5376)]:
    for inplace in (False, True):
        scope['test_indexed_fp32_gate_matches_eager'](rows, hidden, inplace)
    scope['test_indexed_fp32_scale_shift_matches_eager'](rows, hidden)
print(json.dumps({'h3_numerical_cases_passed':9, 'torch':torch.__version__, 'gpu':torch.cuda.get_device_name()}), flush=True)
