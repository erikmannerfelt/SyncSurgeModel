import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def generate_test_cases(
    n_glaciers: int,
    n_iters: int,
    min_period: float,
    max_period: float,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate (pseudo-)random test cases of surge periodicities and phases.

    The test cases are not drawn from a normal distribution but are (supposed to be) completely random.

    Parameters
    ----------
    n_glaciers
        The number of glaciers to generate test cases for (for each iteration).
    n_iters
        The number of random cases to generate for each glacier.
    min_period
        The minimum bound of randomly generated surge periodicities.
    max_period
        The maximum bound of randomly generated surge periodicities.
    random_state
        Optional. The random state of the random number generator.

    Returns
    -------
    A tuple of (phases, periods).

    Each array is a two-dimensional array:
        First dimension: Each glacier
        Second dimension: Each iteration
    """
    rng: np.random.Generator = np.random.default_rng(random_state)

    # Initialize empty lists that will be filled up with random values for each glacier
    phases = []
    periods = []

    for _ in range(n_glaciers):

        # Generate phases as a fraction of the cycle (see below to convert to yrs)
        # These will be shuffled and act as different test cases
        phase = np.linspace(0, 1, n_iters)

        # Generate periodicities between the min and max. These will be shuffled
        period = np.linspace(min_period, max_period, n_iters)

        # Shuffle both arrays
        for arr in (phase, period):
            rng.shuffle(arr)

        # Convert the phase from a fraction of a cycle to an actual value in yrs
        phase *= period

        phases.append(phase)
        periods.append(period)

    phases = np.vstack(phases)
    periods = np.vstack(periods)

    return phases, periods


def count_surges(phases: np.ndarray, periods: np.ndarray, test_year: float, sync_threshold: float) -> np.ndarray:
    """
    Count the amount of surges that should occur at a given year within a given acceptable range.

    Parameters
    ----------
    phases
        The surge phases for every glacier. See `generate_test_cases` for the expected shape.
    periods
        The surge periodicities for every glacier. See `generate_test_cases` for the expected shape.
    test_year
        The year to evaluate how many surges occur on.
    sync_threshold
        The amount of years +- the test year to accept as synchronous.
    """
    # Find out where the test time is in the cycle and return the remainder
    time_in_cycle = (test_year - phases) % periods

    # If the time is either near the beginning of a cycle or near the end, it's surging
    is_surging = np.min([time_in_cycle, periods - time_in_cycle], axis=0) < (sync_threshold)

    # Count the occurrences of surges for each iteration
    n_surges = np.count_nonzero(is_surging, axis=0)

    return n_surges
    

def main(
    n_glaciers: int = 50,
    max_surging_n: int = 15,
    n_iters: int = int(1e6),
    min_period: float = 75.0,
    max_period: float = 150.0,
    random_state: int = 1,
    test_year: int = 0,
):
    """
    Run the main simulation.

    Parameters
    ----------
    n_glaciers
        The number of glaciers to generate test cases for (for each iteration).
    max_surging_n
        The maximum amount of syncronized surges to evaluate probabilities for.
    n_iters
        The number of random cases to generate for each glacier.
    min_period
        The minimum bound of randomly generated surge periodicities.
    max_period
        The maximum bound of randomly generated surge periodicities.
    random_state
        The random state of the random number generator.
    test_year
        The year to evaluate how many surges occur on.
 
    """
    phases, periods = generate_test_cases(
        n_glaciers=n_glaciers, n_iters=n_iters, min_period=min_period, max_period=max_period, random_state=random_state
    )

    all_in_phase = phases.copy()
    all_in_phase[:, :] = phases[[0], :]

    plt.figure(figsize=(8, 5))
    for i, (scenario, phase_to_use) in enumerate([("Random phase", phases), ("Synchronized phase", all_in_phase)], start=1):

        plt.subplot(1, 2, i)
        # Test the probabilities of between 2 and N glaciers
        n_glaciers_arr = np.arange(2, max_surging_n + 1)
        # Test lots of different thresholds to accept as synchronous and plot them all
        for sync_threshold in [5., 10., 15., 30., 50., 75., 150.]:
            n_surges = count_surges(phases=phase_to_use, periods=periods, test_year=test_year, sync_threshold=sync_threshold)

            n_surge_likelihood = 100 * np.count_nonzero(n_surges[:, None] >= n_glaciers_arr[None, :], axis=0) / n_iters

            plt.plot(n_glaciers_arr, n_surge_likelihood, label=f"<{sync_threshold:.0f} yrs")

        plt.grid()
        plt.title(scenario)
        plt.legend()
        #plt.yscale("log")
        plt.ylabel("Likelihood percentage (%)")
        plt.xlabel("N synchronized glaciers")

    out_path = Path("figures/surge_sync_likelihood.jpg")
    out_path.parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=600)

    plt.show()


def main2(
    n_glaciers: int = 50,
    max_surging_n: int = 17,
    n_iters: int = int(1e4),
    min_period: float = 75.0,
    max_period: float = 150.0,
    random_state: int = 1,
    sync_threshold: float = 10.,
):
    phases, periods = generate_test_cases(
        n_glaciers=n_glaciers, n_iters=n_iters, min_period=min_period, max_period=max_period, random_state=random_state
    )
    n_glaciers_arr = np.arange(2, n_glaciers + 1)
    n_independent_surges = count_surges(phases=phases, periods=periods, test_year=0, sync_threshold=sync_threshold)

    independent_surge_likelihood = 100 * np.count_nonzero(n_independent_surges[:, None] >= n_glaciers_arr[None, :], axis=0) / n_iters

    independent_surge_spread = []
    rng: np.random.Generator = np.random.default_rng(random_state)
    n_evals = 200
    for i in range(n_evals):
        selection = rng.integers(0, n_iters, size=n_iters // n_evals)

        independent_likelihood_subset = 100 * np.count_nonzero(n_independent_surges[selection, None] >= n_glaciers_arr[None, :], axis=0) / selection.size
        independent_surge_spread.append(independent_likelihood_subset)
        
    independent_surge_spread = np.percentile(independent_surge_spread, [25, 75], axis=0)

    times = np.linspace(0, 600, 50)

    periods_repeated = np.repeat(periods[:, :, None], times.size, 2)
    time_in_cycle = times[None, None, :] % periods_repeated
    n_sync_surges = np.count_nonzero((time_in_cycle < sync_threshold) | ((periods_repeated - time_in_cycle) < sync_threshold), 0)

    n_sync_surges_mean = np.mean(n_sync_surges, axis=0)
    n_sync_surges_spread = np.percentile(n_sync_surges, [25, 75], axis=0)

    def fmt_n_surges(data, position):
        return int(data)

    plt.figure(figsize=(8, 5))

    plt.subplot(1, 2, 1)
    plt.title("Case: Independent surges")
    plt.fill_between(n_glaciers_arr, independent_surge_spread[0, :], independent_surge_spread[1, :], alpha=0.5)
    plt.plot(n_glaciers_arr, independent_surge_likelihood)
    ylim = plt.gca().get_ylim()
    plt.vlines(max_surging_n, *ylim, color="red")
    plt.ylabel(f"Likelihood of N surges within {int(sync_threshold * 2)} years")
    plt.xlabel(f"Number (N) of surges within {int(sync_threshold * 2)} years")
    plt.ylim(ylim)
    plt.grid()
    xlim = plt.gca().get_xlim()
    xticks = plt.gca().get_xticks()
    xticks = xticks[(xticks > xlim[0]) & (xticks < xlim[1])]
    xticks = np.unique(np.r_[xticks, [max_surging_n]])
    plt.xticks(xticks) 
    plt.gca().get_xticklabels()[np.argwhere(xticks == max_surging_n).ravel()[0]].set_color("red")
    #plt.text(max_surging_n + 1, np.mean(ylim), f"{max_surging_n} surge line (like observed)")

    plt.subplot(1, 2, 2)
    plt.title("Case: Phase-synchronised surges")
    plt.fill_between(times, n_sync_surges_spread[0, :], n_sync_surges_spread[1, :], alpha=0.5)
    plt.plot(times, n_sync_surges_mean, color="royalblue")

    xlim = plt.gca().get_xlim()
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position("right")
    plt.hlines(max_surging_n, *xlim, color="red")
    #plt.text(np.mean(xlim), max_surging_n + 1, f"{max_surging_n} surge line (like observed)", ha="center")
    plt.xlim(xlim)

    ylim = plt.gca().get_ylim()
    yticks = plt.gca().get_yticks()
    yticks = yticks[(yticks > ylim[0]) & (yticks < ylim[1])]
    yticks = np.unique(np.r_[yticks, [max_surging_n]])
    plt.yticks(yticks) 
    plt.gca().get_yticklabels()[np.argwhere(yticks == max_surging_n).ravel()[0]].set_color("red")
    plt.ylabel(f"Number of surges within {int(sync_threshold * 2)} years")
    plt.xlabel("Years since synchronisation")
    plt.grid()

    plt.tight_layout()
    plt.savefig("figures/surges_independent_vs_sync.jpg", dpi=600)
    plt.show()

    

if __name__ == "__main__":
    main()
