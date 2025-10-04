import os
from modal import Image, Stub, Mount, Volume, method

image = Image.debian_slim().uv_pip_install("torch","pillow","imageio")
vol_work = Volume.from_name("splats-workspace", create_if_missing=True)
vol_cache = Volume.from_name("splats-cache", create_if_missing=True)


stub = Stub("3dgs-min", image=image)

@stub.function(volumes={"/workspace": vol_work, "/cache": vol_cache)
def sync_workspace(src_scenes: str, src_code: str):
    import shutil, pathlib
    dst_s = pathlib.Path("/workspace/scenes")
    dst_c = pathlib.Path("/workspace/minsplats")
    dst_s.mkdir(parents=True, exist_ok=True)
    dst_c.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_scenes, dst_s, dirs_exist_ok=True)
    shutil.copytree(src_code, dst_c, dirs_exist_ok=True)
    vol_work.commit()
    
    return "ok"

@stub.function(gpu="A10G", volumes={"/workspace": vol_work, "/cache": vol_cache)
def train(scene_rel: str = "garden", steps: int = 100, num: int = 20000, lr: float = 1e-2):
    import sys
    sys.path.insert(0, "/out")
    from minsplats.train import main as train_main
    import argparse
    scene = f"/workspace/scenes/{scene_rel}"
    argv = ["--scene", scene, "--steps", str(steps), "--num", str(num), "--lr", str(lr)]
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene"); parser.add_argument("--steps"); 
    parser.add_argument("--num"); parser.add_argument("--lr")
    train_main.__globals__["__name__"] = "__main__"
    import runpy
    runpy.run_module("minsplats.train", run_name="__main__", alter_sys=True, 
    init_globals={"__name__":"__main__","__package__":"minsplats","__spec__":None})