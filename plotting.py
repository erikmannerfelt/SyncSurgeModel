import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def make_surge_cycle(times: np.ndarray, period: float = 100.0, phase: float = 0.0, surge_length: float = 2.):
    phase_corr = times - phase

    distance = (phase_corr % period)

    closer_to_upper = distance > (period / 2)
    distance[closer_to_upper] = period - distance[closer_to_upper]


    cycle = np.clip(surge_length - distance, 0, surge_length * 1.5) ** 2

    cycle /= cycle.max()

    return cycle

    print(cycle)

    return

    return np.clip(np.sin(2 * np.pi * (phase_corr / period)), 0, None) ** 10


def show_multiple_surges(
    n_glaciers: int = 15, min_period: float = 75.0, max_period: float = 150.0, random_state: int = 1
):
    rng: np.random.Generator = np.random.default_rng(random_state)

    periods = rng.normal(np.mean([min_period, max_period]), scale=max_period - min_period, size=n_glaciers)
    random_phases = rng.normal(size=n_glaciers) * periods

    shape = (3, 2)
    times = np.linspace(0, 300, num=1000)

    plt.figure(figsize=(8, 5))

    for i, scenario in enumerate(["Random phases", "Synchronized phases"]):

        if i == 0:
            phases = random_phases
        else:
            phases = np.repeat(random_phases[0], random_phases.size)

        plt.subplot2grid(shape, ((i + 1), 0))
        cycles = []
        for j in range(n_glaciers):
            cycle = make_surge_cycle(times, period=periods[j], phase=phases[j])

            if j == 0 and i == 0:
                plt.subplot2grid(shape, (0, 0), colspan=2)
                plt.plot(times, cycle)
                plt.title("Glacier surge example")
                plt.subplot2grid(shape, (i + 1, 0))

            cycles.append(cycle)

            plt.plot(times, cycle)

        plt.title(f"{n_glaciers} glaciers at the same time\n({scenario})")

        all_cycles = pd.Series(np.ravel(cycles), np.repeat(times, len(cycles)))
        all_cycles = (all_cycles > 0.9).astype(float)

        all_cycles = all_cycles.groupby(all_cycles.index.astype(int)).sum()

        sync_time = 10
        rolling = all_cycles.rolling(sync_time, min_periods=1).sum()

        plt.subplot2grid(shape, ( i + 1, 1))
        plt.title(f"N concurrent surges (<{sync_time} yrs)\n({scenario})")
        plt.bar(rolling.index, rolling, width=1)

    out_path = Path("figures/multi_surge_example.jpg")
    out_path.parent.mkdir(exist_ok=True)
    plt.tight_layout()

    plt.savefig(out_path, dpi=600)
    plt.show()

    
