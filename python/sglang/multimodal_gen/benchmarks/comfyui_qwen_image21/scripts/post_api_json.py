#!/usr/bin/env python3
"""POST an API-format workflow JSON file to ComfyUI and wait for it."""
import json
import sys
import time
import urllib.request

path, port = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8188
graph = json.load(open(path))
base = f"http://127.0.0.1:{port}"
req = urllib.request.Request(base + "/prompt", data=json.dumps({"prompt": graph}).encode(),
                             headers={"Content-Type": "application/json"})
pid = json.loads(urllib.request.urlopen(req).read())["prompt_id"]
t0 = time.time()
while True:
    h = json.loads(urllib.request.urlopen(f"{base}/history/{pid}").read())
    if pid in h and (h[pid]["status"].get("completed") or h[pid]["status"].get("status_str") == "error"):
        break
    time.sleep(0.5)
st = h[pid]["status"]
print(json.dumps({"workflow": path.rsplit("/", 1)[-1], "status": st.get("status_str"), "seconds": round(time.time() - t0, 2),
                  "outputs": [i["filename"] for o in h[pid]["outputs"].values() for i in o.get("images", [])]}))
