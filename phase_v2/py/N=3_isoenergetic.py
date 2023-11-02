# NOTE: https://qiita.com/JuvenileTalk9/items/e857b9a62b447cc725e3　を元にcv2を自前ビルド
import cv2
import opypy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    file = Path(__file__)
    data = file.parent / "output" / file.stem
    if not data.exists():
        data.mkdir()

    ndim = 3
    w = np.array([-1, 0, 1.])
    model = opypy.OrderEvaluator.default(ndim)

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

    imfs = sorted([(float(f.stem[len("E="):]), f)
                  for f in data.glob("E=*.png")])
    imgs = []
    for E, f in imfs:
        img = cv2.imread(str(f))
        height, width, layers = img.shape
        size = (width, height)
        cv2.putText(img, f"E = {E:.2f}", (0, 50), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 0, 0), 2, cv2.LINE_AA)
        imgs.append(img)
    video = cv2.VideoWriter(str(data / "summary.mp4"),
                            cv2.VideoWriter_fourcc(*'avc1'), 30, size, True)
    for img in imgs:
        video.write(img)
    video.release()


if __name__ == "__main__":
    main()
