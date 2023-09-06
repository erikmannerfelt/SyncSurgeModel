import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def generate_test_cases(
    n_glaciers: int,
    n_iters: int,
    min_cycle: float,
    max_cycle: float,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate (pseudo-)random test cases of surge cycles and phases.

    The test cases are not drawn from a normal distribution but are (supposed to be) completely random.

    Parameters
    ----------
    n_glaciers
        The number of glaciers to generate test cases for (for each iteration).
    n_iters
        The number of random cases to generate for each glacier.
    min_cycle
        The minimum bound of randomly generated surge cycles.
    max_cycle
        The maximum bound of randomly generated surge cycles.
    random_state
        Optional. The random state of the random number generator.

    Returns
    -------
    A tuple of (phases, cycles).

    Each array is a two-dimensional array:
        First dimension: Each glacier
        Second dimension: Each iteration
    """
    rng: np.random.Generator = np.random.default_rng(random_state)

    # Initialize empty lists that will be filled up with random values for each glacier
    phases = []
    cycles = []

    for _ in range(n_glaciers):

        # Generate phases as a fraction of the cycle (see below to convert to yrs)
        # These will be shuffled and act as different test cases
        phase = np.linspace(0, 1, n_iters)

        # Generate cycles between the min and max. These will be shuffled
        cycle = np.linspace(min_cycle, max_cycle, n_iters)

        # Shuffle both arrays
        for arr in (phase, cycle):
            rng.shuffle(arr)

        # Convert the phase from a fraction of a cycle to an actual value in yrs
        phase *= cycle

        phases.append(phase)
        cycles.append(cycle)

    phases = np.vstack(phases)
    cycles = np.vstack(cycles)

    return phases, cycles


def count_surges(phases: np.ndarray, cycles: np.ndarray, test_year: float, sync_threshold: float) -> np.ndarray:
    """
    Count the amount of surges that should occur at a given year within a given acceptable range.

    Parameters
    ----------
    phases
        The surge phases for every glacier. See `generate_test_cases` for the expected shape.
    cycles
        The surge cycles for every glacier. See `generate_test_cases` for the expected shape.
    test_year
        The year to evaluate how many surges occur on.
    sync_threshold
        The amount of years +- the test year to accept as synchronous.
    """
    # Find out where the test time is in the cycle and return the remainder
    time_in_cycle = (test_year - phases) % cycles

    # If the time is either near the beginning of a cycle or near the end, it's surging
    is_surging = np.min([time_in_cycle, cycles - time_in_cycle], axis=0) < (sync_threshold)

    # Count the occurrences of surges for each iteration
    n_surges = np.count_nonzero(is_surging, axis=0)

    return n_surges
    

def main(
    n_glaciers: int = 15,
    n_iters: int = int(1e6),
    min_cycle: float = 75.0,
    max_cycle: float = 150.0,
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
    min_cycle
        The minimum bound of randomly generated surge cycles.
    max_cycle
        The maximum bound of randomly generated surge cycles.
    random_state
        The random state of the random number generator.
    test_year
        The year to evaluate how many surges occur on.
 
    """
    phases, cycles = generate_test_cases(
        n_glaciers=n_glaciers, n_iters=n_iters, min_cycle=min_cycle, max_cycle=max_cycle, random_state=random_state
    )

    all_in_phase = phases.copy()
    all_in_phase[:, :] = phases[[0], :]

    plt.figure(figsize=(8, 5))
    for i, (scenario, phase_to_use) in enumerate([("Random phase", phases), ("Synchronized phase", all_in_phase)], start=1):

        plt.subplot(1, 2, i)
        # Test lots of different thresholds to accept as synchronous and plot them all
        n_glaciers_arr = np.arange(2, n_glaciers + 1)
        for sync_threshold in [5., 10., 15., 30., 50., 75., 150.]:
            n_surges = count_surges(phases=phase_to_use, cycles=cycles, test_year=test_year, sync_threshold=sync_threshold)

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
