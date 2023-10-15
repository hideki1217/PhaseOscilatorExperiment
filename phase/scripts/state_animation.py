import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import simulate

from pathlib import Path


def record_video(K, w0, fname, T=100, N=1000):
    sampling_times = np.linspace(0, T, N)

    model = simulate.OscilatorNetwork()
    sampling_states = model.solve_ivp(K, w0, sampling_times)

    create_video(sampling_times, sampling_states, fname=fname)


def create_video(sampling_times, sampling_states, fname=Path(__file__).stem):
    if isinstance(fname, str):
        fname = Path(fname)
    assert isinstance(fname, Path)
    fname = fname.with_suffix(".mp4")

    fig = plt.figure(figsize=(4.8, 4.8))

    def plot(index):
        plt.cla()

        state = sampling_states[index]
        im = plt.scatter(np.cos(state), np.sin(state), c=w0, cmap=plt.cm.cool)
        plt.plot([0, np.mean(np.cos(state))], [0, np.mean(np.sin(state))])
        plt.xlim((-1.5, 1.5))
        plt.ylim((-1.5, 1.5))

    ani = animation.FuncAnimation(
        fig, plot, interval=(sampling_times[1] - sampling_times[0]) * 1000 / 4, save_count=sampling_states.shape[0])
    ani.save(Path(__file__).parent /
             f"{fname}.mp4", writer="ffmpeg")


if __name__ == "__main__":
    k = 1.5
    p = 0.5
    K = np.array([[0, k], [k * p, 0]])
    w0 = np.array([-1.0, 1.0])

    record_video(K, w0, Path(__file__).parent /
                 f"{Path(__file__).stem}_sample.mp4")
