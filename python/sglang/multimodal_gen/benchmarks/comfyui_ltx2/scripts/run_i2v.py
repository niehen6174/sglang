"""Queue the flattened LTX-2.5 I2V template (official or SGLD) and time it."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import run_prompt, workflows
name, variant = sys.argv[1], sys.argv[2]
repeat = int(sys.argv[3]) if len(sys.argv) > 3 else 1
opts = {"extra_server_args": json.dumps({"master_port": 30105, "scheduler_port": 5655})}
for i in range(repeat):
    p = workflows.official_i2v(sgld=variant == "sgld", prefix=f"ltx25/{name}_run{i}",
                               image="ltx_i2v_start.png", seed=875362541677469 + i, sgld_options=opts)
    pid, wall, exec_s, outputs = run_prompt.run(p)
    rec = {"name": name, "run": i, "prompt_id": pid, "wall_s": round(wall, 2), "exec_s": exec_s and round(exec_s, 2), "sgld": variant == "sgld", "i2v": True}
    print(json.dumps(rec), flush=True)
    with open(os.path.join(run_prompt.RES, "logs", "runs.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
