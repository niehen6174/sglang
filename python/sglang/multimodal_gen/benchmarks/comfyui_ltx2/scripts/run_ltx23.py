"""Queue the flattened LTX-2.3 T2V template (official or SGLD) and time it."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import run_prompt, workflows
name, variant = sys.argv[1], sys.argv[2]
repeat = int(sys.argv[3]) if len(sys.argv) > 3 else 1
kw = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
opts = {"extra_server_args": json.dumps({"master_port": 30105, "scheduler_port": 5655})}
opts.update(kw.pop("sgld_options", {}))
for i in range(repeat):
    p = workflows.ltx23_t2v(sgld=variant == "sgld", prefix=f"ltx23/{name}_run{i}",
                            seed=810138461690240 + i, sgld_options=opts, **kw)
    pid, wall, exec_s, outputs = run_prompt.run(p)
    rec = {"name": name, "run": i, "prompt_id": pid, "wall_s": round(wall, 2), "exec_s": exec_s and round(exec_s, 2), "sgld": variant == "sgld", "model": "ltx23", **kw}
    print(json.dumps(rec), flush=True)
    with open(os.path.join(run_prompt.RES, "logs", "runs.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
