# SPDX-License-Identifier: Apache-2.0
"""Convert a ComfyUI workflow (UI or API JSON) to run its DiT on SGLD.

Every change is visible in the result and listed in the report:
- the H3 / Qwen / Z-Image / FLUX.1 ``UNETLoader`` becomes ``SGLDUNETLoader``;
- speed-only patch nodes on that model are bypassed (mode 4, still in the graph),
  and the speed-up they asked for goes to an ``SGLDOptions`` node when the model
  supports it (Sage / sparse attention, step caches);
- LoRA nodes on that model become ``SGLDLoraLoader`` (``SGLDLoraLoaderWithCLIP``
  when the node also patches CLIP); LoRA stacks keep their CLIP / trigger outputs
  and hand their model LoRAs to a chain of ``SGLDLoraLoader`` nodes.
Nothing else is touched. Nodes the converter does not know stay as they are.
"""

import copy
import importlib.util
import json

from . import rules

BYPASS = 4
# Diffusion-model loaders: if any stays on ComfyUI (LTX, Wan ...), only nodes that
# can be traced to an SGLD loader are converted.
DIFFUSION_LOADERS = {
    "UNETLoader",
    "UnetLoaderGGUF",
    "CheckpointLoaderSimple",
    "CheckpointLoader",
}


class Report:
    def __init__(self):
        self.changes, self.notes = [], []

    def change(self, where, node, what):
        title = (
            node.get("title")
            or node.get("_meta", {}).get("title")
            or node.get("type")
            or node.get("class_type")
        )
        self.changes.append({"node": where, "title": title, "change": what})

    def note(self, text):
        if text not in self.notes:
            self.notes.append(text)

    def markdown(self) -> str:
        lines = ["# SGLD conversion report", ""]
        if not self.changes:
            lines.append("No SGLD-servable diffusion model found; nothing changed.")
        else:
            lines += ["| node | title | change |", "|---|---|---|"]
            lines += [
                f"| {c['node']} | {c['title']} | {c['change']} |" for c in self.changes
            ]
        if self.notes:
            lines += ["", "## Notes", ""] + [f"- {n}" for n in self.notes]
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- model rules


def _supported(model_type, settings, report):
    """Drop settings the model cannot run in integrated mode (preflight's rules)."""
    try:
        from ..core.preflight import _ATTENTION_BY_MODEL, _CACHE_MODELS
    except Exception:  # converter used without the runtime: same rules, inline
        h3 = ("minimax_h3", "fast_h3", "vdn_h3")
        _CACHE_MODELS = {"cache_dit": h3, "spectrum": h3, "teacache": ()}
        _ATTENTION_BY_MODEL = {
            "generic": ("fa", "torch_sdpa", "sage_attn", "sage_attn_3"),
            "minimax_h3": ("fa", "torch_sdpa", "sage_attn", "sage_attn_3", "sol_attn",
                           "subblock_sparse_attn", "video_sparse_attn_h3"),
            "fast_h3": ("fa", "torch_sdpa", "sage_attn", "video_sparse_attn_h3"),
        }  # fmt: skip
    ids = {"sage_attention2": "sage_attn", "sol_attention": "sol_attn"}
    kept = dict(settings)
    backend = kept.get("attention_backend")
    if model_type == "fast_h3" and backend:
        report.note(
            "FastH3 keeps the VSA-H3 sparse attention it was distilled with; attention patches dropped."
        )
        backend = None
    if backend == "sol_attention" and importlib.util.find_spec("sol_attn") is None:
        report.note(
            "sol-attn is not installed here, so Sol nodes map to SageAttention 2 "
            "(pip install the NVlabs Sana sol-engine sparse_backends for Sol)."
        )
        backend = "sage_attention2"
    allowed = _ATTENTION_BY_MODEL.get(model_type, _ATTENTION_BY_MODEL["generic"])
    if backend and ids.get(backend, backend) not in allowed:
        report.note(
            f"{backend} is not available for {model_type}; attention left at auto."
        )
        backend = None
    for key in ("attention_backend", "attention_backend_config"):
        kept.pop(key, None)
    if backend:
        kept["attention_backend"] = backend
        if backend == "sol_attention" and "attention_backend_config" in settings:
            kept["attention_backend_config"] = settings["attention_backend_config"]
    mode = kept.get("cache_mode")
    if mode and model_type not in _CACHE_MODELS.get(mode, ()):
        report.note(
            f"{mode} step caching is not supported for {model_type} yet; the cache node is bypassed without a replacement."
        )
        kept.pop("cache_mode")
    return kept


def _carry(settings, setting, node):
    """Record a speed node's setting; the first one per kind wins, as on ComfyUI's chain."""
    key, value = setting
    if key in settings:
        return
    settings[key] = value
    if setting == rules.SOL:
        settings["attention_backend_config"] = rules.sol_config(node)


# ---------------------------------------------------------------- UI workflow


class _Graph:
    """Nodes and links of the top-level workflow or of one subgraph definition."""

    def __init__(self, data, where, ids):
        self.data, self.where, self.ids = data, where, ids
        self.nodes = {n["id"]: n for n in data.get("nodes", [])}

    def links(self):
        for link in self.data.get("links", []) or []:
            if isinstance(link, list):
                yield {"id": link[0], "origin_id": link[1], "origin_slot": link[2],
                       "target_id": link[3], "target_slot": link[4], "type": link[5], "_raw": link}  # fmt: skip
            else:
                yield {**link, "_raw": link}

    def link(self, link_id):
        return next((l for l in self.links() if l["id"] == link_id), None)

    def set_origin(self, link, node_id, slot):
        raw = link["_raw"]
        if isinstance(raw, list):
            raw[1], raw[2] = node_id, slot
        else:
            raw["origin_id"], raw["origin_slot"] = node_id, slot

    def add_link(self, origin, origin_slot, target, target_slot, kind):
        link_id = self.ids.link()
        raw = ([link_id, origin["id"], origin_slot, target["id"], target_slot, kind]
               if self.ids.list_links and self.where == "" else
               {"id": link_id, "origin_id": origin["id"], "origin_slot": origin_slot,
                "target_id": target["id"], "target_slot": target_slot, "type": kind})  # fmt: skip
        self.data.setdefault("links", []).append(raw)
        origin["outputs"][origin_slot].setdefault("links", [])
        origin["outputs"][origin_slot]["links"] = (
            origin["outputs"][origin_slot]["links"] or []
        ) + [link_id]
        target["inputs"][target_slot]["link"] = link_id
        return link_id

    def add_node(self, kind, pos, inputs, outputs, widgets, size=(320, 110)):
        node_id = self.ids.node(self)
        node = {"id": node_id, "type": kind, "pos": list(pos), "size": list(size), "flags": {},
                "order": 0, "mode": 0, "inputs": inputs, "outputs": outputs,
                "properties": {"Node name for S&R": kind}, "widgets_values": widgets}  # fmt: skip
        self.data["nodes"].append(node)
        self.nodes[node_id] = node
        return node

    def model_source(self, node, seen=None):
        """The SGLD loader feeding ``node``'s MODEL input in this graph, or None."""
        seen = seen if seen is not None else set()
        if node["id"] in seen:
            return None
        seen.add(node["id"])
        if node.get("type") in rules.SGLD_LOADERS:
            return node
        inputs = sorted(
            node.get("inputs") or [], key=lambda i: i.get("name") != "model"
        )
        for inp in inputs:
            if inp.get("link") is None or inp.get("type") not in ("MODEL", "*"):
                continue
            link = self.link(inp["link"])
            origin = self.nodes.get(link["origin_id"]) if link else None
            if origin is not None:
                found = self.model_source(origin, seen)
                if found is not None:
                    return found
        return None


class _Ids:
    def __init__(self, wf, graphs):
        self.wf = wf
        self.list_links = bool(wf.get("links")) and isinstance(wf["links"][0], list)
        all_links = [l["id"] for g in graphs for l in g.links()]
        self.next_link = max(all_links + [wf.get("last_link_id") or 0]) + 1

    def link(self):
        link_id = self.next_link
        self.next_link += 1
        self.wf["last_link_id"] = link_id
        return link_id

    def node(self, graph):
        node_id = max(list(graph.nodes) + [self.wf.get("last_node_id") or 0, 0]) + 1
        if graph.where == "":
            self.wf["last_node_id"] = node_id
        else:
            graph.data.setdefault("state", {})["lastNodeId"] = node_id
        return node_id


def _options_node(graph, loader, report):
    """The SGLDOptions node wired into ``loader``; created if there is none."""
    slot = next(
        (i for i, inp in enumerate(loader["inputs"]) if inp["name"] == "sgld_options"),
        None,
    )
    if slot is not None and loader["inputs"][slot].get("link") is not None:
        link = graph.link(loader["inputs"][slot]["link"])
        node = graph.nodes.get(link["origin_id"]) if link else None
        return node if node and node.get("type") == "SGLDOptions" else None
    if slot is None:
        loader["inputs"].append(
            {"name": "sgld_options", "type": "SGLD_OPTIONS", "link": None}
        )
        slot = len(loader["inputs"]) - 1
    x, y = loader.get("pos", [0, 0])[:2]
    node = graph.add_node(
        "SGLDOptions", (x, y - 330), [],
        [{"name": "sgld_options", "type": "SGLD_OPTIONS", "links": []}],
        rules.options_values(), size=(420, 300),
    )  # fmt: skip
    graph.add_link(node, 0, loader, slot, "SGLD_OPTIONS")
    report.change(
        f"{graph.where}{node['id']}",
        node,
        f"added, wired into SGLD loader {loader['id']}",
    )
    return node


def _apply_options(graph, loader, settings, report):
    if not settings:
        return
    node = _options_node(graph, loader, report)
    if node is None:
        report.note(
            f"Loader {loader['id']} takes options from a non-SGLDOptions node; speed settings not applied."
        )
        return
    values = node.setdefault("widgets_values", rules.options_values())
    applied = []
    for key, value in settings.items():
        i = rules.options_index(key, values)
        if (
            isinstance(values, list)
            and i < len(values)
            and values[i] == rules.OPTIONS_DEFAULTS[key]
        ):
            values[i] = value
            applied.append(f"{key}={value}")
    if applied:
        report.change(f"{graph.where}{node['id']}", node, "set " + ", ".join(applied))


def _insert_lora_chain(graph, after, loras, report):
    """SGLDLoraLoader nodes for ``loras`` between ``after``'s MODEL output and its consumers."""
    slot = next(i for i, o in enumerate(after["outputs"]) if o.get("type") == "MODEL")
    consumers = list(after["outputs"][slot].get("links") or [])
    x, y = after.get("pos", [0, 0])[:2]
    prev, prev_slot = after, slot
    after["outputs"][slot]["links"] = []
    for k, (name, strength) in enumerate(loras):
        node = graph.add_node(
            "SGLDLoraLoader", (x + 40 * k, y - 140 * (k + 1)),
            [{"name": "model", "type": "MODEL", "link": None}],
            [{"name": "MODEL", "type": "MODEL", "links": []}],
            [name, strength, "", "all"],
        )  # fmt: skip
        graph.add_link(prev, prev_slot, node, 0, "MODEL")
        report.change(
            f"{graph.where}{node['id']}",
            node,
            f"added: {name} @ {strength} (from stack {after['id']})",
        )
        prev, prev_slot = node, 0
    for link_id in consumers:
        link = graph.link(link_id)
        graph.set_origin(link, prev["id"], prev_slot)
        prev["outputs"][prev_slot]["links"].append(link_id)


def _model_name(graph, node, graphs):
    """The file a loader loads. A widget promoted to a subgraph input takes its value
    from the subgraph instance node, not from the (often stale) inner widget."""
    name = (node.get("widgets_values") or [None])[0]
    inp = next(
        (
            i
            for i in node.get("inputs") or []
            if i.get("name") in ("unet_name", "ckpt_name")
        ),
        None,
    )
    if graph.where == "" or not inp or inp.get("link") is None:
        return name
    link = graph.link(inp["link"])
    if not link or link["origin_id"] != -10:
        return name
    sg_inputs = graph.data.get("inputs") or []
    if link["origin_slot"] >= len(sg_inputs):
        return name
    promoted = sg_inputs[link["origin_slot"]]["name"]
    for parent in graphs:
        for inst in parent.nodes.values():
            if inst.get("type") != graph.data.get("id"):
                continue
            widget_inputs = [
                i["name"] for i in inst.get("inputs") or [] if i.get("widget")
            ]
            values = inst.get("widgets_values") or []
            if promoted in widget_inputs and widget_inputs.index(promoted) < len(
                values
            ):
                return values[widget_inputs.index(promoted)]
    return name


def convert_workflow(wf, model_dir_lookup=None):
    """Return (converted copy of a UI workflow, Report)."""
    wf = copy.deepcopy(wf)
    report = Report()
    graphs = [_Graph(wf, "", None)]
    for sg in (wf.get("definitions") or {}).get("subgraphs", []):
        graphs.append(_Graph(sg, f"{sg.get('name') or sg.get('id')}/", None))
    ids = _Ids(wf, graphs)
    for g in graphs:
        g.ids = ids
    lookup = model_dir_lookup or (lambda name: None)

    # 1) loaders
    loaders, others = [], []
    for g in graphs:
        for node in list(g.nodes.values()):
            kind = node.get("type")
            if kind in rules.SGLD_LOADERS:
                name = _model_name(g, node, graphs)
                loaders.append(
                    (g, node, rules.model_type(name, lookup(name)) or "minimax_h3")
                )
                continue
            if kind not in DIFFUSION_LOADERS:
                continue
            name = _model_name(g, node, graphs)
            mtype = rules.model_type(name, lookup(name))
            if (
                mtype is None
                or kind.startswith("Checkpoint")
                or node.get("mode") in (2, BYPASS)
            ):
                if node.get("mode") not in (2, BYPASS):
                    others.append(node)
                continue
            node["type"] = rules.SGLD_LOADER
            node.setdefault("properties", {})["Node name for S&R"] = rules.SGLD_LOADER
            # SGLD ignores weight_dtype (quantization is an SGLDOptions setting).
            node["widgets_values"] = [name, "default"]
            node.setdefault("inputs", [])
            if not any(i.get("name") == "sgld_options" for i in node["inputs"]):
                node["inputs"].append(
                    {"name": "sgld_options", "type": "SGLD_OPTIONS", "link": None}
                )
            loaders.append((g, node, mtype))
            report.change(
                f"{g.where}{node['id']}", node, f"{kind} -> SGLDUNETLoader ({mtype})"
            )
    if not loaders:
        return wf, report
    mixed = bool(others)
    if mixed:
        report.note("Some diffusion models stay on ComfyUI (" + ", ".join(sorted({str((n.get('widgets_values') or ['?'])[0]) for n in others}))
                    + "); only nodes traced to an SGLD loader were converted.")  # fmt: skip

    # 2) speed / LoRA nodes on the SGLD model
    settings = {}  # id(loader node) -> settings
    for g in graphs:
        for node in list(g.nodes.values()):
            kind = node.get("type")
            if node.get("mode") in (2, BYPASS):
                continue
            if (
                kind not in rules.SPEED_NODES
                and kind not in rules.LORA_NODES
                and kind not in rules.STACK_NODES
            ):
                continue
            loader = g.model_source(node)
            if loader is None and mixed:
                report.note(
                    f"{g.where}{node['id']} {kind}: could not trace its model to an SGLD loader; left unchanged."
                )
                continue
            targets = [loader] if loader is not None else [l for gg, l, _ in loaders]
            where = f"{g.where}{node['id']}"
            if kind in rules.SPEED_NODES:
                node["mode"] = BYPASS
                setting = rules.speed_setting(node)
                for t in targets:
                    if setting:
                        _carry(settings.setdefault(id(t), {}), setting, node)
                report.change(
                    where,
                    node,
                    f"bypassed ({kind})"
                    + (f"; -> {setting[0]}={setting[1]}" if setting else ""),
                )
            elif kind in rules.LORA_NODES:
                new_kind, remap = rules.LORA_NODES[kind]
                node["widgets_values"] = remap(node.get("widgets_values") or [])
                node["type"] = new_kind
                node.setdefault("properties", {})["Node name for S&R"] = new_kind
                report.change(where, node, f"{kind} -> {new_kind}")
            else:
                loras = rules.stack_loras(node)
                if loras:
                    _insert_lora_chain(g, node, loras, report)
                    report.change(
                        where,
                        node,
                        f"model LoRAs moved to {len(loras)} SGLDLoraLoader node(s); stack keeps CLIP / triggers",
                    )
                    report.note("LoRA stacks lose the text-encoder half of their LoRAs on SGLD models "
                                "(H3 / Qwen LoRAs normally train the DiT only).")  # fmt: skip

    # 3) speed-ups the bypassed nodes asked for, per loader, if its model runs them
    for g, loader, mtype in loaders:
        wanted = settings.get(id(loader))
        if wanted:
            _apply_options(g, loader, _supported(mtype, wanted, report), report)
    return wf, report


# ---------------------------------------------------------------- API prompt


def convert_api(prompt, model_dir_lookup=None):
    """Same conversion for an API-format prompt (``{id: {class_type, inputs}}``)."""
    prompt = copy.deepcopy(prompt)
    report = Report()
    lookup = model_dir_lookup or (lambda name: None)
    loaders, others = {}, []
    for nid, node in list(prompt.items()):
        kind = node.get("class_type")
        if kind in rules.SGLD_LOADERS:
            name = node["inputs"].get("unet_name")
            loaders[nid] = rules.model_type(name, lookup(name)) or "minimax_h3"
        elif kind in DIFFUSION_LOADERS:
            name = node["inputs"].get("unet_name") or node["inputs"].get("ckpt_name")
            mtype = rules.model_type(name, lookup(name))
            if mtype is None or kind.startswith("Checkpoint"):
                others.append(nid)
                continue
            node["class_type"] = rules.SGLD_LOADER
            node["inputs"] = {"unet_name": name, "weight_dtype": "default"}
            loaders[nid] = mtype
            report.change(nid, node, f"{kind} -> SGLDUNETLoader ({mtype})")
    if not loaders:
        return prompt, report

    def source(nid, seen):
        while nid in prompt and nid not in seen:
            seen.add(nid)
            if prompt[nid]["class_type"] in rules.SGLD_LOADERS:
                return nid
            links = sorted((k != "model", v[0]) for k, v in prompt[nid]["inputs"].items()
                           if isinstance(v, list) and len(v) == 2)  # fmt: skip
            for _, up in links:
                found = source(str(up), seen)
                if found:
                    return found
            return None
        return None

    settings = {}
    for nid in list(prompt):
        node = prompt.get(nid)
        kind = node["class_type"] if node else None
        if (
            kind not in rules.SPEED_NODES
            and kind not in rules.LORA_NODES
            and kind not in rules.STACK_NODES
        ):
            continue
        model = node["inputs"].get("model")
        if not (isinstance(model, list) and len(model) == 2):
            continue
        loader = source(str(model[0]), set())
        if loader is None and others:
            report.note(
                f"{nid} {kind}: could not trace its model to an SGLD loader; left unchanged."
            )
            continue
        targets = [loader] if loader else list(loaders)
        if kind in rules.SPEED_NODES:
            api_node = {"type": kind, "widgets_values": node["inputs"]}
            setting = rules.speed_setting(api_node)
            for t in targets:
                if setting:
                    _carry(settings.setdefault(t, {}), setting, api_node)
            for other in prompt.values():
                for k, v in other["inputs"].items():
                    if (
                        isinstance(v, list)
                        and len(v) == 2
                        and str(v[0]) == nid
                        and v[1] == 0
                    ):
                        other["inputs"][k] = list(model)
            del prompt[nid]
            report.change(
                nid,
                node,
                f"removed ({kind})"
                + (f"; -> {setting[0]}={setting[1]}" if setting else ""),
            )
        elif kind in rules.LORA_NODES:
            new_kind = rules.LORA_NODES[kind][0]
            inputs = node["inputs"]
            strength = inputs.get("strength_model", inputs.get("strength", 1.0))
            node["class_type"] = new_kind
            node["inputs"] = {
                "model": model,
                "lora_name": inputs["lora_name"],
                "strength_model": strength,
            }
            if new_kind == "SGLDLoraLoader":
                node["inputs"].update(nickname="", target="all")
            else:
                node["inputs"].update(
                    clip=inputs.get("clip"),
                    strength_clip=inputs.get("strength_clip", 1.0),
                )
            report.change(nid, node, f"{kind} -> {new_kind}")
        else:
            report.note(
                f"{nid} {kind}: API conversion does not rewrite LoRA stacks; convert the UI workflow."
            )

    for loader, mtype in loaders.items():
        wanted = settings.get(loader)
        if not wanted:
            continue
        kept = _supported(mtype, wanted, report)
        if not kept:
            continue
        link = prompt[loader]["inputs"].get("sgld_options")
        opts_id = str(link[0]) if isinstance(link, list) else f"{loader}.sgld_options"
        opts = prompt.setdefault(opts_id, {"class_type": "SGLDOptions", "inputs": {}})
        prompt[loader]["inputs"]["sgld_options"] = [opts_id, 0]
        applied = []
        for key, value in kept.items():
            if (
                opts["inputs"].get(key, rules.OPTIONS_DEFAULTS[key])
                == rules.OPTIONS_DEFAULTS[key]
            ):
                opts["inputs"][key] = value
                applied.append(f"{key}={value}")
        if applied:
            report.change(opts_id, opts, "set " + ", ".join(applied))
    return prompt, report


def is_api_prompt(data) -> bool:
    return (
        isinstance(data, dict)
        and "nodes" not in data
        and all(isinstance(v, dict) and "class_type" in v for v in data.values())
    )


def dumps(data) -> str:
    return json.dumps(data, indent=1, ensure_ascii=False)
