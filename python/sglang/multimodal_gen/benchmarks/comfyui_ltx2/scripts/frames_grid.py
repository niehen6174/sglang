"""Side-by-side frame grid (rows = output dirs, cols = frame indices)."""
import sys, av, numpy as np
from PIL import Image
out, idxs, dirs = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], sys.argv[3:]
rows = []
for d in dirs:
    import glob
    c = av.open(glob.glob(d + "/*.mp4")[0])
    fr = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    rows.append(np.concatenate([fr[min(i, len(fr) - 1)] for i in idxs], axis=1))
img = Image.fromarray(np.concatenate(rows, axis=0))
img.thumbnail((1800, 1800))
img.save(out, quality=88)
