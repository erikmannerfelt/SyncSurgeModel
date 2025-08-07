import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import itertools

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


def generate_test_cases_std(
    n_glaciers: int,
    n_iters: int,
    mean_period: float,
    std_period: float,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate (pseudo-)random test cases of surge periodicities and phases.

    The test cases are drawn from a normal distribution.

    Parameters
    ----------
    n_glaciers
        The number of glaciers to generate test cases for (for each iteration).
    n_iters
        The number of random cases to generate for each glacier.
    mean_period
        The mean of the randomly generated surge periodicities.
    std_period
        The standard deviation of the randomly generated surge periodicities.
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

    periods = rng.normal(mean_period, std_period, size=(n_glaciers, n_iters))
    phases = rng.uniform(0, 1, size=(n_glaciers, n_iters))

    phases *= periods

    return phases, periods


def count_surges(phases: np.ndarray, periods: np.ndarray, sync_threshold: float, test_year: float = 0.) -> np.ndarray:
    """
    Count the amount of surges that should occur at a given year within a given acceptable range.

    Parameters
    ----------
    phases
        The surge phases for every glacier. See `generate_test_cases` for the expected shape.
    periods
        The surge periodicities for every glacier. See `generate_test_cases` for the expected shape.
    sync_threshold
        The amount of years +- the test year to accept as synchronous.
    test_year
        The year to evaluate how many surges occur on.
    """
    # Find out where the test time is in the cycle and return the remainder
    time_in_cycle = (test_year - phases) % periods

    # If the time is either near the beginning of a cycle or near the end, it's surging
    is_surging = np.min([time_in_cycle, periods - time_in_cycle], axis=0) < (sync_threshold)

    # Count the occurrences of surges for each iteration
    n_surges = np.count_nonzero(is_surging, axis=0)

    return n_surges


def calculate_surge_frequency_spread(baseline_frequency: float = 1.5, year_span: float = 10., n_tests: int = 15, n_simulations: int = 5000):
    import tqdm
    import tqdm.contrib.concurrent

    test_mean_periods = np.linspace(50, 300, n_tests)
    test_std_periods = np.linspace(10, 200, n_tests)
    test_n_glaciers = np.linspace(50, 1000, n_tests, dtype=int)
    combinations = list(itertools.product(test_mean_periods, test_std_periods, test_n_glaciers))

    bins = np.linspace(0, 3, 300)
    histograms = []

    def process(combination):
        mean_period, std_period, n_glaciers = combination        
        phases, periods = generate_test_cases_std(n_glaciers=n_glaciers, n_iters=n_simulations, mean_period=mean_period, std_period=std_period,random_state=int(f"{mean_period:.0f}{std_period:.0f}{n_glaciers}"))
        # Clip the periodicity to min=10 years because we've never seen anything faster than that
        periods = np.clip(periods, a_min=10, a_max=None)
   
        n_surges = count_surges(phases=phases, periods=periods, sync_threshold=year_span / 2) / year_span

        if abs(n_surges.mean() - baseline_frequency) > 0.25:
            return

        hist= np.histogram(n_surges, bins=bins, range=(0, 3))[0] 
        return hist

    result = tqdm.contrib.concurrent.thread_map(process, combinations)
    histograms = [l for l in result if l is not None]
    histogram = np.sum(histograms, axis=0)

    n_over_threshold = np.sum(histogram[bins[1:] >= 3.])
    frac_over_threshold = n_over_threshold / np.sum(histogram)
    print(f"3: {n_over_threshold=}, {frac_over_threshold=}, n_total={np.sum(histogram)}")
    n_over_threshold = np.sum(histogram[bins[1:] >= 2.])
    frac_over_threshold = n_over_threshold / np.sum(histogram)
    print(f"2: {n_over_threshold=}, {frac_over_threshold=}, n_total={np.sum(histogram)}")
    
    plt.bar(bins[:-1] + np.diff(bins) /2, np.sum(histograms, axis=0))
    plt.ylabel("Simulation frequency")
    plt.xlabel("N surges / 10 years")
    plt.show()
    

    
    

def main(
    n_glaciers: int = 15,
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
        n_glaciers_arr = np.arange(2, n_glaciers + 1)
        # Test lots of different thresholds to accept as synchronous and plot them all
        for sync_threshold in [5., 10., 15., 30., 50., 75., 150.]:
            n_surges = count_surges(phases=phase_to_use, periods=periods, test_year=test_year, sync_threshold=sync_threshold)

            n_surge_likelihood = 100 * np.count_nonzero(n_surges[:, None] >= n_glaciers_arr[None, :], axis=0) / n_iters

            plt.plot(n_glaciers_arr, n_surge_likelihood, label=f"<{sync_threshold:.0f} yrs")

        plt.grid()
        plt.title(scenario)
        plt.legend()
        plt.yscale("log")
        plt.ylabel("Likelihood percentage (%)")
        plt.xlabel("N synchronized glaciers")

    out_path = Path("figures/surge_sync_likelihood.jpg")
    out_path.parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=600)

    plt.show()

if __name__ == "__main__":
    main()
