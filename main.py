import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random as rng


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


#
def generate_glaciers(
        phases: np.ndarray,
        periods: np.ndarray,
        variances: np.array,
        rng: np.random.Generator,
        n_surges: int = 200
) -> np.array:
    """
    generates the surge events for every glacier according to a normal distribution with mean period and variance,
    we then offset by the value contained in the vector phases

    Parameters
    ----------
    phases
        An array that contains the expected value of the phase of the glaciers
    periods
        An array that contains the expected value of the periods of the glaciers
    variances
        An array that contains the variance od the periods of the glaciers
    rng
        random number generator
    n_surges
        number of surges simulated for each glacier (note, importanat to make sure that the nuimber is long enough to reach the test year we want to have)

    """
    n_glaciers = len(periods)
    # generate standard normal distributions
    surges = rng.normal(0, 1, size=(n_glaciers, n_surges))
    # trasform standard normal distribution in normal distribution with mean periods and varianve var
    surges = surges * np.reshape(np.sqrt(variances), (-1, 1))
    surges = surges + np.reshape(periods, (-1, 1))


    np.cumsum(surges, axis=1, out=surges)
    surges = surges + np.reshape(phases, (-1, 1))
    return surges


def count_surges(test_data: list, test_year: float, sync_threshold: float) -> np.ndarray:
    """
    Count the amount of surges that should occur at a given year within a given acceptable range.

    Parameters
    ----------
    test_data
        An array where each cell contains a matrix that represent the surge life for every glacier. See `generate_glaciers` for the expected shape.
    test_year
        The year to evaluate how many surges occur on.
    sync_threshold
        The amount of years +- the test year to accept as synchronous.
    """
    # Find out where the test time is in the cycle and return the remainder
    n_surges = []
    for glaciers_life in test_data:
        is_surging = np.min(np.abs(glaciers_life - test_year), axis=0) < (sync_threshold)
        n_surges.append(np.count_nonzero(is_surging))

    n_surges = np.array(n_surges)
    return n_surges


def main(
        n_glaciers: int = 15,
        n_iters: int = int(1e5),
        min_period: float = 75.0,
        max_period: float = 150.0,
        random_state: int = 1,
        test_year: int | None = 2020,
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

    rng: np.random.Generator = np.random.default_rng(random_state)

    # generate phases and periods and variances, to be reviewed
    # ---------------------------------------------------------------------
    phases, periods = generate_test_cases(
        n_glaciers=n_glaciers, n_iters=n_iters, min_period=min_period, max_period=max_period, random_state=random_state
    )

    variances = rng.normal(11, scale=1, size=n_glaciers)
    # ---------------------------------------------------------------------

    test_data = [generate_glaciers(phases[:,i], periods[:,i], variances, rng, n_surges=300) for i in range(n_iters)]

    all_in_phase = phases.copy()
    all_in_phase[:, :] = phases[[0], :]

    plt.figure(figsize=(8, 5))
    for i, (scenario, phase_to_use) in enumerate([("Random phase", phases), ("Synchronized phase", all_in_phase)],
                                                 start=1):

        plt.subplot(1, 2, i)
        # Test the probabilities of between 2 and N glaciers
        n_glaciers_arr = np.arange(2, n_glaciers + 1)
        # Test lots of different thresholds to accept as synchronous and plot them all
        for sync_threshold in [5., 10., 15., 30., 50., 75., 150.]:
            n_surges = count_surges(test_data=test_data, test_year=test_year,
                                    sync_threshold=sync_threshold)

            n_surge_likelihood = 100 * np.count_nonzero(n_surges[:, None] >= n_glaciers_arr[None, :], axis=0) / n_iters

            plt.plot(n_glaciers_arr, n_surge_likelihood, label=f"<{sync_threshold:.0f} yrs")

        plt.grid()
        plt.title(scenario)
        plt.legend()
        # plt.yscale("log")
        plt.ylabel("Likelihood percentage (%)")
        plt.xlabel("N synchronized glaciers")

    out_path = Path("figures/surge_sync_likelihood.jpg")
    out_path.parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=600)

    plt.show()


if __name__ == "__main__":
    main()
