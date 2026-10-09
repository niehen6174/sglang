"""Queue the LTX-2.3 template as shipped (dev-FP8 + distilled LoRA) and time it.

usage: run_ltx23_tpl.py NAME official|sgld REPEAT PORT [json kwargs for ltx23_template]
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import run_prompt, workflows
name, variant, repeat, port = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
kw = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
run_prompt.URL = f"http://127.0.0.1:{port}"
opts = {"extra_server_args": json.dumps({"master_port": 30105, "scheduler_port": 5655})}
opts.update(kw.pop("sgld_options", {}))
seed0 = kw.pop("seed", 810138461690240)
if "prompt_file" in kw:
    kw["prompt"] = open(kw.pop("prompt_file")).read()
expect_error = kw.pop("expect_error", False)
run_kw = kw.pop("run_kw", [])  # per-run overrides, e.g. a different prompt on a warm run
base_kw = kw
for i in range(repeat):
    kw = dict(base_kw)
    kw.update(run_kw[i] if i < len(run_kw) else {})
    if "prompt_file" in kw:
        kw["prompt"] = open(kw.pop("prompt_file")).read()
    p = workflows.ltx23_template(sgld=variant == "sgld", prefix=f"ltx23tpl/{name}_run{i}",
                                 seed=seed0 + i, sgld_options=opts, **kw)
    if expect_error:
        try:
            run_prompt.run(p)
            print(json.dumps({"name": name, "unexpected": "prompt succeeded"}))
        except RuntimeError as err:
            msg = str(err)
            print(json.dumps({"name": name, "expected_error": msg[msg.find("exception_message"):][:600]}))
        continue
    pid, wall, exec_s, outputs = run_prompt.run(p)
    enhanced = outputs.get("328", {}).get("text")
    rec = {"name": name, "run": i, "prompt_id": pid, "wall_s": round(wall, 2), "exec_s": exec_s and round(exec_s, 2),
           "sgld": variant == "sgld", "model": "ltx23_dev_lora", "port": port, **kw,
           "enhanced_prompt": enhanced}
    print(json.dumps({k: (v[:80] + "..." if isinstance(v, list) and v and isinstance(v[0], str) else v) for k, v in rec.items() if k != "enhanced_prompt"}), flush=True)
    with open(os.path.join(run_prompt.RES, "logs", "runs.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
