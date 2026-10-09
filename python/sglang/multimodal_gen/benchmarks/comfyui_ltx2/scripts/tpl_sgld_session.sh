#!/bin/bash
# tpl_sgld_session.sh NAME REPEAT [extra json merged into sgld options] : fresh ComfyUI (GPU1), template SGLD runs
NAME=$1; REP=$2; EXTRA=${3:-'{}'}
D=/scratch/data/sgld_comfy/results/ltx25
LW=$(python3 -c "
import json,sys
extra=json.loads(sys.argv[1])
srv={'master_port':30105,'scheduler_port':5655,'dit_layerwise_resident_layers':0.75,'dit_layerwise_residency_lifetime':'permanent','dit_offload_prefetch_size':2}
srv.update(extra.pop('server',{}))
opts={'dit_layerwise_offload':True,'extra_server_args':json.dumps(srv)}
kw={'sgld_options':opts}; kw.update(extra)
print(json.dumps(kw))" "$EXTRA")
$D/scripts/session.sh $NAME "" "python scripts/run_ltx23_tpl.py $NAME sgld $REP 8189 '$LW' | cut -c1-300"
grep -E "OutOfMemory|out of memory|Error" $D/logs/comfy_$NAME.log | grep -v "CUDACachingAllocator" | head -5
grep -c "CUDACachingAllocator.*OOM" $D/logs/comfy_$NAME.log
