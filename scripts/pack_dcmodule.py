#!/usr/bin/env python3
"""Pack a built libml into dist/ml-<version>-<target>.dcmodule (same layout as `dpm pack`).

Archive layout:
  manifest.json
  lib/<target>/<libml.dylib|libml.so|ml.dll>

Also writes <file>.sha256 and prints a registry `artifacts` entry as JSON.
Used by CI, where the `dpm` binary is not available.
"""
import argparse
import hashlib
import json
import os
import zipfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lib", required=True, help="path to built cdylib")
    p.add_argument("--target", required=True, help="rust target triple, e.g. x86_64-unknown-linux-gnu")
    p.add_argument("--version", required=True)
    p.add_argument("--abi-major", type=int, default=1)
    p.add_argument("--abi-minor", type=int, default=7)
    p.add_argument("--out-dir", default="dist")
    p.add_argument("--url-base", default="", help="release download URL prefix for the printed artifacts entry")
    a = p.parse_args()

    lib_name = os.path.basename(a.lib)
    lib_rel = "lib/{}/{}".format(a.target, lib_name)
    manifest = {
        "schema_version": 1,
        "name": "ml",
        "version": a.version,
        "abi_version": {"major": a.abi_major, "minor": a.abi_minor},
        "targets": {a.target: lib_rel},
    }

    os.makedirs(a.out_dir, exist_ok=True)
    file_name = "ml-{}-{}.dcmodule".format(a.version, a.target)
    out = os.path.join(a.out_dir, file_name)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest) + "\n")
        z.write(a.lib, lib_rel)

    h = hashlib.sha256()
    with open(out, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    with open(out + ".sha256", "w") as f:
        f.write("{}  {}\n".format(digest, file_name))

    entry = {
        "target": a.target,
        "url": (a.url_base.rstrip("/") + "/" + file_name) if a.url_base else file_name,
        "sha256": digest,
        "version": a.version,
    }
    with open(os.path.join(a.out_dir, "artifact-{}.json".format(a.target)), "w") as f:
        json.dump(entry, f)
    print(json.dumps(entry))


if __name__ == "__main__":
    main()
