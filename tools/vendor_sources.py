"""Reproduce inference-only vendoring from the pinned upstream checkouts.

Run from the repository root after cloning the sources under .cache/upstream.
No training code or application UI is imported at runtime.
"""
import ast
from pathlib import Path
import shutil
import subprocess
import json

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "cidenoise" / "vendor"


def main():
    revisions = {}
    for name in ("napari-fluoresfm", "UniFMIR"):
        path = ROOT / ".cache/upstream" / name
        revisions[name] = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    for sub in ("", "fluoresfm", "unifmir"):
        p = DEST / sub
        p.mkdir(parents=True, exist_ok=True)
        (p / "__init__.py").write_text("", encoding="utf-8")
    src = ROOT / ".cache/upstream/napari-fluoresfm"
    for name in ("unet_sd_c.py", "unet_attention.py", "biomedclip_embedder.py"):
        text = (src / "src/napari_fluoresfm/fluoresfm/models" / name).read_text(encoding="utf-8")
        text = text.replace("napari_fluoresfm.fluoresfm.models.unet_attention", "cidenoise.vendor.fluoresfm.unet_attention")
        (DEST / "fluoresfm" / name).write_text(text, encoding="utf-8")
    shutil.copyfile(src / "LICENSE", DEST / "fluoresfm/LICENSE")
    src = ROOT / ".cache/upstream/UniFMIR"
    tree = ast.parse((src / "model/swinir.py").read_text(encoding="utf-8"))
    keep = {"swinir", "Mlp", "window_partition", "window_reverse", "WindowAttention", "SwinTransformerBlock", "PatchMerging", "BasicLayer", "RSTB", "PatchEmbed", "PatchUnEmbed", "Upsample", "Upsample2"}
    class Clean(ast.NodeTransformer):
        def visit_Expr(self, node):
            # Remove diagnostic prints and unconditional CUDA synchronization only.
            if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) in ("print", "torch.cuda.synchronize"):
                return None
            return self.generic_visit(node)
    tree.body = [n for n in tree.body if (isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in keep) or (isinstance(n, (ast.Import, ast.ImportFrom)) and not (isinstance(n, ast.ImportFrom) and n.module == "model.enlcn"))]
    tree = ast.fix_missing_locations(Clean().visit(tree))
    (DEST / "unifmir/swinir.py").write_text("# Inference subset of UniFMIR; see THIRD_PARTY.md.\n" + ast.unparse(tree) + "\n", encoding="utf-8")
    shutil.copyfile(src / "LICENSE", DEST / "unifmir/LICENSE")
    shutil.copyfile(src / "LICENSE", ROOT / "LICENSE")
    (DEST / "revisions.json").write_text(json.dumps(revisions, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
