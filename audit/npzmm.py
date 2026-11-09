"""Memory-map one array stored (uncompressed) inside an .npz without loading it."""
import zipfile, numpy as np
from numpy.lib import format as npf


def npz_memmap(path, key):
    zf = zipfile.ZipFile(path)
    info = zf.getinfo(key + ".npy")
    assert info.compress_type == 0, "compressed member"
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        h = f.read(30)
        nlen = int.from_bytes(h[26:28], "little"); elen = int.from_bytes(h[28:30], "little")
        start = info.header_offset + 30 + nlen + elen
        f.seek(start)
        ver = npf.read_magic(f)
        rd = npf.read_array_header_1_0 if ver == (1, 0) else npf.read_array_header_2_0
        shape, fortran, dtype = rd(f)
        data_off = f.tell()
    return np.memmap(path, dtype=dtype, mode="r", offset=data_off, shape=shape, order="F" if fortran else "C")
