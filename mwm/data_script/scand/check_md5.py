import os
import time
from pyDataverse.api import NativeApi, DataAccessApi
from pyDataverse.models import Dataverse
import os, requests, sys
import hashlib
from pathlib import Path

def get_md5_map(files_list):
    md5_map = {}
    for file in files_list:
        filename = file["dataFile"]['filename']
        md5_map[filename] = file["dataFile"]['md5']
    return md5_map

def md5_file(path: Path, chunk_size: int = 200 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def iter_local_bags(root: Path):
    # only check .bag / .bag.gz
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if name.endswith(".bag") or name.endswith(".bag.gz"):
            yield p

def check(check_dir, md5_map):
    
    ok = 0
    mismatch = 0
    missing_remote = 0
    local_names = set()
    root = Path(check_dir)
    
    for p in iter_local_bags(root):
        fname = p.name  # match by filename only (Dataverse stores filename, not full path)
        local_md5 = md5_file(p)
        local_names.add(p.name)

        if fname not in md5_map:
            missing_remote += 1
            print(f"[MISS-REMOTE] {p}  md5={local_md5}", flush=True)
            continue

        remote_md5 = md5_map[fname]
        if local_md5 == remote_md5:
            ok += 1
            print(f"[OK] {p}  md5={local_md5}", flush=True)
        else:
            mismatch += 1
            print(f"[MISMATCH] {p}\n  local : {local_md5}\n  remote: {remote_md5}", flush=True)
    
    remote_bags = {name for name in md5_map.keys()
                   if name.lower().endswith(".bag") or name.lower().endswith(".bag.gz")}
    
    missing = sorted(remote_bags - local_names)

    total_checked = ok + mismatch + missing_remote
    print("\n===== SUMMARY =====", flush=True)
    print(f"checked        : {total_checked}", flush=True)
    print(f"ok             : {ok}", flush=True)
    print(f"mismatch       : {mismatch}", flush=True)
    print(f"missing_remote : {missing_remote}", flush=True)
    print(f"missing : {len(missing)}", flush=True)
   

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="which dir to check")
    args = parser.parse_args()

    base_url = 'https://dataverse.tdl.org/'
    api_token = "5ddf7cd3-012d-4892-ad4b-717a2aff92ca"
    api = NativeApi(base_url, api_token)

    DOI = "doi:10.18738/T8/0PRYRH"
    dataset = api.get_dataset(DOI)

    files_list = dataset.json()['data']['latestVersion']['files']

    md5_map = get_md5_map(files_list)
    check(args.dir, md5_map)


