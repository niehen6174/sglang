# SPDX-License-Identifier: Apache-2.0
"""Workflow converter: stock ComfyUI workflows -> SGLD integrated mode, as visible edits."""

import importlib.util
import json

import pytest

from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.convert import (
    convert_api,
    convert_workflow,
    rules,
)

H3 = "minimax_h3_t2v_int8.safetensors"


def _node(nid, kind, widgets=(), inputs=(), outputs=(("MODEL", "MODEL"),), mode=0):
    return {
        "id": nid, "type": kind, "mode": mode, "pos": [nid * 100, 0], "size": [200, 80],
        "inputs": [{"name": n, "type": t, "link": None} for n, t in inputs],
        "outputs": [{"name": n, "type": t, "links": []} for n, t in outputs],
        "widgets_values": list(widgets),
    }  # fmt: skip


def _wire(wf, link_id, a, a_slot, b, b_slot, kind="MODEL"):
    nodes = {n["id"]: n for n in wf["nodes"]}
    wf["links"].append([link_id, a, a_slot, b, b_slot, kind])
    nodes[a]["outputs"][a_slot]["links"].append(link_id)
    nodes[b]["inputs"][b_slot]["link"] = link_id


def _chain(model_name, *patches):
    """UNETLoader -> patches... -> KSampler, as a UI workflow."""
    wf = {"nodes": [_node(1, "UNETLoader", [model_name, "default"])], "links": [],
          "last_node_id": 1, "last_link_id": 0}  # fmt: skip
    prev = 1
    for i, (kind, widgets) in enumerate(patches, start=2):
        wf["nodes"].append(_node(i, kind, widgets, inputs=[("model", "MODEL")]))
        _wire(wf, i, prev, 0, i, 0)
        prev = i
    wf["nodes"].append(_node(99, "KSampler", inputs=[("model", "MODEL")], outputs=[]))
    _wire(wf, 99, prev, 0, 99, 0)
    return wf


def _by_type(wf, kind):
    return [n for n in wf["nodes"] if n["type"] == kind]


def _options(wf):
    (opts,) = _by_type(wf, "SGLDOptions")
    return dict(zip(rules.OPTIONS_WIDGETS, opts["widgets_values"]))


def test_loader_swap_and_speed_nodes_become_visible_options() -> None:
    wf = _chain(H3, ("PathchSageAttentionKJ", ["auto", False]),
                ("H3MiniMaxCache", [0.05, 0.3, 0.8, 1, "cpu", False]),
                ("TorchCompileModel", ["inductor"]))  # fmt: skip
    out, report = convert_workflow(wf)
    nodes = {n["id"]: n for n in out["nodes"]}
    assert nodes[1]["type"] == "SGLDUNETLoader" and nodes[1]["widgets_values"] == [
        H3,
        "default",
    ]
    # Speed nodes stay in the graph, bypassed; the sampler wiring is untouched.
    assert [nodes[i]["mode"] for i in (2, 3, 4)] == [4, 4, 4]
    assert nodes[99]["inputs"][0]["link"] == 99
    opts = _options(out)
    assert (
        opts["attention_backend"] == "sage_attention2"
        and opts["cache_mode"] == "cache_dit"
    )
    # The options node is wired into the loader's new sgld_options input.
    slot = next(i for i in nodes[1]["inputs"] if i["name"] == "sgld_options")
    link = next(l for l in out["links"] if l[0] == slot["link"])
    assert link[1] == _by_type(out, "SGLDOptions")[0]["id"]
    assert wf["nodes"][0]["type"] == "UNETLoader"  # input not modified
    assert any("bypassed" in c["change"] for c in report.changes)


def test_sol_keeps_its_sparsity_and_user_choices_win(monkeypatch) -> None:
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: object()
    )  # sol-attn present
    sol = ("MiniMaxH3MemoryEfficientSolAttentionPatch",
           [True, 1.3, 4096, False, "diag", False, False, "exact_kv", ""])  # fmt: skip
    out, _ = convert_workflow(_chain(H3, sol))
    opts = _options(out)
    assert opts["attention_backend"] == "sol_attention"
    assert json.loads(opts["attention_backend_config"]) == {
        "dense_layers": "", "dense_steps": 0, "tau": 1.3, "thresh_type": "diag"}  # fmt: skip

    # An SGLDOptions node the user already set keeps its attention choice.
    wf = _chain(H3, sol)
    user = _node(50, "SGLDOptions", [rules.OPTIONS_DEFAULTS[k] for k in rules.OPTIONS_WIDGETS],
                 outputs=[("sgld_options", "SGLD_OPTIONS")])  # fmt: skip
    user["widgets_values"][rules.OPTIONS_WIDGETS.index("attention_backend")] = (
        "flash_attention"
    )
    wf["nodes"].append(user)
    wf["nodes"][0]["type"] = "SGLDUNETLoader"
    wf["nodes"][0]["inputs"].append(
        {"name": "sgld_options", "type": "SGLD_OPTIONS", "link": None}
    )
    _wire(wf, 50, 50, 0, 1, 0, "SGLD_OPTIONS")
    out, _ = convert_workflow(wf)
    assert _options(out)["attention_backend"] == "flash_attention"


def test_settings_the_model_cannot_run_are_not_carried() -> None:
    out, report = convert_workflow(
        _chain(
            "qwen_image_2.1_int8.safetensors",
            ("EasyCache", [0.2, 0.2, 0.7, False]),
            ("ModelAttentionBackend", ["sage attention"]),
        )  # fmt: skip
    )
    assert _options(out)["cache_mode"] == "off"  # no step cache for Qwen-Image 2.1 yet
    assert _options(out)["attention_backend"] == "sage_attention2"
    assert any("not supported for qwen_image21" in n for n in report.notes)

    out, _ = convert_workflow(
        _chain("fasth3_8step_int8.safetensors", ("PathchSageAttentionKJ", []))
    )
    assert not _by_type(out, "SGLDOptions")  # FastH3 keeps its VSA default


def test_other_models_and_their_patches_stay_on_comfyui() -> None:
    """An LTX refine stage next to H3: only what feeds the H3 loader changes."""
    wf = _chain(H3, ("PathchSageAttentionKJ", []))
    wf["nodes"].append(
        _node(10, "UNETLoader", ["ltx-2.5-22b-distilled.safetensors", "default"])
    )
    wf["nodes"].append(
        _node(11, "PathchSageAttentionKJ", [], inputs=[("model", "MODEL")])
    )
    _wire(wf, 11, 10, 0, 11, 0)
    out, report = convert_workflow(wf)
    nodes = {n["id"]: n for n in out["nodes"]}
    assert nodes[10]["type"] == "UNETLoader" and nodes[11]["mode"] == 0
    assert nodes[1]["type"] == "SGLDUNETLoader" and nodes[2]["mode"] == 4
    assert any("stay on ComfyUI" in n for n in report.notes)


def test_lora_nodes_are_rewritten_and_stacks_hand_over_their_loras() -> None:
    wf = _chain(H3, ("LoraLoaderModelOnly", ["turbo.safetensors", 0.8]),
                ("MiniMaxH3TurboLoRA", ["fast.safetensors", 1.0, False]))  # fmt: skip
    stack = _node(5, "Power Lora Loader (rgthree)",
                  [{}, {"type": "PowerLoraLoaderHeaderWidget"},
                   {"on": True, "lora": "style.safetensors", "strength": 0.6},
                   {"on": False, "lora": "off.safetensors", "strength": 1.0}, {}, ""],
                  inputs=[("model", "MODEL"), ("clip", "CLIP")],
                  outputs=[("MODEL", "MODEL"), ("CLIP", "CLIP")])  # fmt: skip
    wf["nodes"].insert(-1, stack)
    sampler = wf["nodes"][-1]
    wf["links"] = [l for l in wf["links"] if l[0] != 99]
    sampler["inputs"][0]["link"] = None
    wf["nodes"][2]["outputs"][0]["links"] = []
    _wire(wf, 98, 3, 0, 5, 0)
    _wire(wf, 99, 5, 0, 99, 0)
    out, _ = convert_workflow(wf)
    nodes = {n["id"]: n for n in out["nodes"]}
    assert nodes[2]["type"] == "SGLDLoraLoader"
    assert nodes[2]["widgets_values"] == ["turbo.safetensors", 0.8, "", "all"]
    assert nodes[3]["type"] == "SGLDLoraLoader" and nodes[3]["widgets_values"][:2] == [
        "fast.safetensors",
        1.0,
    ]
    # The stack keeps running for CLIP with its LoRA switched off ...
    assert nodes[5]["widgets_values"][2]["on"] is False
    # ... and an SGLDLoraLoader carries that LoRA between the stack and the sampler.
    (added,) = [n for n in _by_type(out, "SGLDLoraLoader") if n["id"] not in (2, 3)]
    assert added["widgets_values"][:2] == ["style.safetensors", 0.6]
    feed = next(l for l in out["links"] if l[0] == nodes[99]["inputs"][0]["link"])
    assert feed[1] == added["id"]

    out, _ = convert_workflow(_chain(H3))
    wf = _chain(H3)
    wf["nodes"].insert(1, _node(7, "LoraLoader", ["x.safetensors", 1.0, 0.5],
                                inputs=[("model", "MODEL"), ("clip", "CLIP")],
                                outputs=[("MODEL", "MODEL"), ("CLIP", "CLIP")]))  # fmt: skip
    wf["links"], wf["nodes"][0]["outputs"][0]["links"] = [], []
    wf["nodes"][-1]["inputs"][0]["link"] = None
    _wire(wf, 1, 1, 0, 7, 0)
    _wire(wf, 2, 7, 0, 99, 0)
    out, _ = convert_workflow(wf)
    assert {n["id"]: n for n in out["nodes"]}[7]["type"] == "SGLDLoraLoaderWithCLIP"


def test_loader_inside_a_subgraph_reads_the_promoted_model_name() -> None:
    """PlagueKind: the inner widget holds a stale default; the instance holds the file."""
    inner = _node(1, "UNETLoader", ["ltx-2.3-22b-dev.safetensors", "default"],
                  inputs=[("unet_name", "COMBO")])  # fmt: skip
    inner["inputs"][0]["widget"] = {"name": "unet_name"}
    inner["inputs"][0]["link"] = 7
    cache = _node(
        2,
        "H3MiniMaxCache",
        [0.05, 0.3, 0.8, 1, "cpu", False],
        inputs=[("model", "MODEL")],
    )
    sub = {"id": "sg-1", "name": "Loader", "inputs": [{"name": "unet_name", "linkIds": [7]}],
           "nodes": [inner, cache],
           "links": [{"id": 7, "origin_id": -10, "origin_slot": 0, "target_id": 1, "target_slot": 0, "type": "COMBO"},
                     {"id": 8, "origin_id": 1, "origin_slot": 0, "target_id": 2, "target_slot": 0, "type": "MODEL"}]}  # fmt: skip
    inner["outputs"][0]["links"] = [8]
    cache["inputs"][0]["link"] = 8
    instance = _node(5, "sg-1", [H3], inputs=[("unet_name", "COMBO")])
    instance["inputs"][0]["widget"] = {"name": "unet_name"}
    wf = {
        "nodes": [instance],
        "links": [],
        "definitions": {"subgraphs": [sub]},
        "last_link_id": 8,
    }
    out, report = convert_workflow(wf)
    sub_out = out["definitions"]["subgraphs"][0]
    nodes = {n["id"]: n for n in sub_out["nodes"]}
    assert nodes[1]["type"] == "SGLDUNETLoader" and nodes[1]["widgets_values"][0] == H3
    assert nodes[2]["mode"] == 4
    assert any(n["type"] == "SGLDOptions" for n in sub_out["nodes"])
    assert any(l.get("type") == "SGLD_OPTIONS" for l in sub_out["links"])


def test_api_prompts_convert_the_same_way() -> None:
    prompt = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": H3, "weight_dtype": "default"}},
        "2": {"class_type": "PathchSageAttentionKJ", "inputs": {"model": ["1", 0], "sage_attention": "auto"}},
        "3": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["2", 0], "lora_name": "t.safetensors", "strength_model": 0.5}},
        "4": {"class_type": "KSampler", "inputs": {"model": ["3", 0]}},
    }  # fmt: skip
    out, _ = convert_api(prompt)
    assert out["1"]["class_type"] == "SGLDUNETLoader" and "2" not in out
    assert out["3"]["class_type"] == "SGLDLoraLoader" and out["3"]["inputs"][
        "model"
    ] == ["1", 0]
    opts_id = out["1"]["inputs"]["sgld_options"][0]
    assert out[opts_id]["inputs"] == {"attention_backend": "sage_attention2"}


def test_workflows_without_an_sgld_model_are_left_alone() -> None:
    wf = _chain("ltx-2.5-22b-distilled.safetensors", ("PathchSageAttentionKJ", []))
    out, report = convert_workflow(wf)
    assert out == wf and not report.changes


@pytest.mark.parametrize(
    "name,kind",
    [("minimax_h3_fl2va_pruned_fp8_scaled.safetensors", "minimax_h3"),
     ("fastvideo_fasth3_8step_v2_int8.safetensors", "fast_h3"),
     ("qwen_image_2.1_int8_convrot.safetensors", "qwen_image21"),
     ("H3\\Cobijada_H3_Hybrid_int8.safetensors", "minimax_h3"),
     ("ltx-2.3-22b-dev_int8.safetensors", None),
     ("flux2-klein.safetensors", None)],
)  # fmt: skip
def test_model_type_from_file_name(name, kind) -> None:
    assert rules.model_type(name) == kind
