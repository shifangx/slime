"""Apply the compatibility patch to an explicitly supplied, pinned SGLang source.

Set SUPER_VL_SGLANG_PROCESSOR to nano_nemotron_vl.py from SGLang commit
0b3bb0cbe31873994c9f989fddfe2f87ca839fdd. No network access or installed-source
mutation is performed by these tests. Only the pure rendering method is executed;
processor/tokenizer integration still requires the pinned serving runtime.
"""

import ast
import hashlib
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

NUM_GPUS = 0
pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def expand(tmp_path_factory):
    source = os.environ.get("SUPER_VL_SGLANG_PROCESSOR")
    if not source:
        pytest.skip("Set SUPER_VL_SGLANG_PROCESSOR to the pinned processor source")
    original = Path(source).read_bytes()
    assert hashlib.sha256(original).hexdigest() == "8d45178821fde4e89a3c806d7a51eb0d892f030266b76b9fb88f66d5c44faad3"
    root = tmp_path_factory.mktemp("sglang-patch")
    target = root / "python/sglang/srt/multimodal/processors/nano_nemotron_vl.py"
    target.parent.mkdir(parents=True)
    target.write_bytes(original)
    patch = ROOT / "docker/patch/latest/sglang-nemotron-multi-image.patch"
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "apply", str(patch)], cwd=root, check=True, capture_output=True)
    tree = ast.parse(target.read_text())
    methods = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "expand_image_placeholders"
    ]
    assert len(methods) == 1
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "expand_image_placeholders"
    ]
    assert len(calls) == 2, "Both static and dynamic image paths must use the helper"
    namespace = {}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(target), "exec"), namespace)
    processor = SimpleNamespace(IMG_CONTEXT_TOKEN="<image>")
    return lambda prompt, images: namespace["expand_image_placeholders"](processor, prompt, images)


def test_multiple_images_expand_at_corresponding_placeholders(expand):
    assert expand("Compare <image> with <image>.", ["[A]", "[B]"]) == "Compare [A] with [B]."


def test_expanded_image_markers_are_not_replaced_again(expand):
    assert expand("<image> then <image>", ["[<image><image>]", "[<image>]"]) == ("[<image><image>] then [<image>]")


def test_single_placeholder_preserves_grouped_image_contract(expand):
    assert expand("See <image>.", ["[A]", "[B]"]) == "See [A][B]."
    assert expand("See <image>.", ["[A]"]) == "See [A]."


@pytest.mark.parametrize("prompt,images", [("<image><image>", ["[A]"]), ("text", ["[A]"])])
def test_mismatched_image_count_is_rejected(expand, prompt, images):
    with pytest.raises(ValueError, match="placeholder count"):
        expand(prompt, images)


def test_text_without_images_is_unchanged(expand):
    assert expand("text", []) == "text"
