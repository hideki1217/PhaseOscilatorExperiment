import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

import opypy


def main():
    file = Path(__file__)
    data = file.parent / "output" / file.stem
    if not data.exists():
        data.mkdir()

    ndim = 3
    w = np.array([-1, 0, 1.])
    model = opypy.OrderEvaluator(
        window=30000, epsilon=1e-4, sampling_dt=0.01, max_iteration=100000, ndim=ndim)

    def f(K1, K2, K3, w):
        K_ = np.array([0, K1, K2,
                       K1, 0, K3,
                       K2, K3, 0])

        status = model.eval(K_, w)

        print(f"{K1}, {K2}, {K3}")
        if (status.value == 0):
            return model.result()
        else:
            return np.nan

    def make_isoenergetic(E):
        E *= ndim / 2
        def iso(p): return f(p[0] * E, p[1] * E, p[2] * E, w=w)
        return iso

    import ternary

    def draw(E):
        figure, tax = ternary.figure(scale=30)
        ternary.heatmapf(make_isoenergetic(E), boundary=True,
                         style="triangular", vmin=0, vmax=1, scale=100)
        tax.boundary(linewidth=2.0)
        plt.box(False)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(data / f"E={E:.2f}.png")
        plt.close()

    for E in map(lambda x: x*0.01, range(301)):
        draw(E)

    ims = [(float(f.stem[len("E="):]), Image.open(f))
           for f in data.glob("E=*.png")]
    ims.sort()
    ims = [x[1] for x in ims]
    ims[0].save(data / "isoenergetic.gif",
                save_all=True, append_images=ims[1:], optimize=False, duration=40, loop=0)


if __name__ == "__main__":
    main()
