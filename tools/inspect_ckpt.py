"""Tool: inspect a pyaxon checkpoint (.ckpt).

Reads the binary format (magic AXON + version + n + [ndim, dims..., floats]) and
prints the shape and parameter count of each tensor, without loading a model.

Usage: python tools/inspect_ckpt.py path/model.ckpt
"""

import struct
import sys


def inspect(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"AXON":
            raise SystemExit("file is not a pyaxon checkpoint (invalid magic)")
        (version,) = struct.unpack("<i", f.read(4))
        (count,) = struct.unpack("<q", f.read(8))
        print(f"checkpoint v{version} — {count} tensors")
        total = 0
        for t in range(count):
            (ndim,) = struct.unpack("<q", f.read(8))
            dims = [struct.unpack("<q", f.read(8))[0] for _ in range(ndim)]
            numel = 1
            for d in dims:
                numel *= d
            total += numel
            f.seek(numel * 4, 1)  # skip the floats
            print(f"  [{t:3d}] shape={dims} params={numel}")
        print(f"total parameters: {total:,}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python tools/inspect_ckpt.py <file.ckpt>")
    inspect(sys.argv[1])
