import argparse
import logging
from math import ceil
from pathlib import Path
from pickle import PicklingError

import emcee
import loky
import numpy as np
import pandas as pd
from loky.backend.context import cpu_count
from omegaconf import OmegaConf

import benchflow


def main():
    parser = argparse.ArgumentParser(description="emcee_sampler")
    parser.add_argument(
        "-i",
        "--input",
        dest="task_filename",
        type=str,
        required=True,
        help="Input config file",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_filename",
        type=str,
        default=None,
        help="Output file",
    )
    parser.add_argument(
        "-r", "--resume", dest="resume_filename", help="Resume from this chain"
    )
    parser.add_argument(
        "-n", "--n_samples", dest="n_samples", type=int, default=10000
    )
    parser.add_argument(
        "-c",
        "--n_workers",
        dest="max_workers",
        type=int,
        default=1,
        help="Max number of worker threads. -1 for all available cores. Default is 1.",
    )

    args = parser.parse_args()

    task_filename = args.task_filename
    output_filename = args.output_filename
    resume_filename = args.resume_filename
    max_workers = args.max_workers
    n_samples = args.n_samples
    if max_workers == -1:
        max_workers = cpu_count()
    print("Max number of worker threads:", max_workers)

    base_cfg_path = Path(__file__).parent.parent / "benchflow/config/task"
    base_output_path = (
        Path(__file__).parent.parent / "benchflow/reference_samples"
    )

    full_cfg_path = base_cfg_path / f"{task_filename}.yaml"
    task_cfg = OmegaConf.load(full_cfg_path)

    task = benchflow.utilities.cfg.cfg_to_task(task_cfg)
    if task.mcmc_info.get("bounded"):
        # Transform to unconstrained coordinates for MCMC
        task._transform_task_to_unconstrained()

    if output_filename is None:
        output_filename = task.sample_filename
    full_output_path = base_output_path / Path(output_filename).with_suffix(
        ".csv"
    )

    if resume_filename is None:
        full_resume_path = None
    else:
        full_resume_path = base_output_path / resume_filename
        print(f"Resuming sampler from {full_resume_path}")

    emcee_sample(
        task,
        full_output_path,
        n_samples,
        resume=full_resume_path,
        max_workers=max_workers,
    )


def emcee_sample(
    task,
    output="test.csv",
    n_samples=10000,
    resume=None,
    max_workers=1,
):
    """Generate and save MCMC samples from a task, using Emcee.

    Args:
        task: A task object.
        output: The path to save the samples.
        n_samples: The minimum number of samples to generate, after burn-in.
        resume: The path to a previous chain h5 file to resume.
        max_workers: The maximum number of worker threads to use.
    """
    n_walkers = 16 * task.D

    if resume is not None:
        # Resume from previous chain
        backend = emcee.backends.HDFBackend(resume)
        n_walkers, n_dim = backend.shape
        assert n_dim == task.D
        start = backend.get_last_sample()
        burn_in = 0
    else:
        backend = emcee.backends.HDFBackend(
            str(Path(output).with_suffix("")) + "_emcee_autosave.h5"
        )  # Autosave chains
        backend.reset(n_walkers, task.D)  # Reset file
        # Randomized starting position
        start = task.x0(randomize=True, sz=n_walkers)
        # start = start + np.random.normal(0, 0.01, size=start.shape)
        burn_in = min(100, ceil(n_samples / n_walkers))

    if not task.mcmc_info["multimodal"]:
        moves = emcee.moves.StretchMove()
    else:
        moves = [
            (emcee.moves.WalkMove(), 0.25),
            (emcee.moves.KDEMove(), 0.25),
            (emcee.moves.DESnookerMove(), 0.25),
            (emcee.moves.DEMove(), 0.25),
        ]

    if max_workers is not None and max_workers > 1:
        sampler = try_multiprocessing(
            task,
            moves,
            n_walkers,
            n_samples,
            start,
            burn_in,
            backend=backend,
            max_workers=max_workers,
        )
    else:
        sampler = emcee.EnsembleSampler(
            n_walkers,
            task.D,
            log_joint,
            args=[task],
            moves=moves,
            backend=backend,
        )
        sampler = run_sampler_until_convergence(
            sampler, n_samples, burn_in, start
        )
        # sampler.run_mcmc(start, n_steps, progress=True)

    n_samples_walker = ceil(n_samples / sampler.nwalkers)
    thin = sampler.iteration // n_samples_walker
    samples = sampler.get_chain(flat=True, thin=thin).copy()[-n_samples:]
    # Invert transformation, if necessary
    if task.transform_to_unconstrained_coordinates:
        samples = task.transform.inverse(samples)
    # Save samples
    samples = pd.DataFrame(samples)
    samples.to_csv(output, index=False, header=False)
    print(f"Samples saved to {output}")
    print(
        f"Note: the burn-in periods may not be large enough; therefore, the saved samples may not be good enough. It is necessary to check the MCMC trace plot, etc., to see if discarding more initial iterations and running the chain for longer is needed."
    )


def run_sampler_until_convergence(
    sampler: emcee.EnsembleSampler, n_samples, burn_in, start, max_n_steps=None
):
    """Run the sampler until convergence, or until max_n_steps."""
    if burn_in > 0:
        sampler.run_mcmc(start, burn_in, progress=True)
        sampler.reset()
    if max_n_steps is None:
        max_n_steps = 10 * n_samples

    n_steps_check = 100
    old_tau = np.inf
    # Now we'll sample for up to max_n steps
    for sample in sampler.sample(start, iterations=max_n_steps, progress=True):
        # Only check convergence every a few steps
        if sampler.iteration % n_steps_check:
            continue

        # Compute the autocorrelation time so far
        try:
            tau = sampler.get_autocorr_time()
            tau = np.maximum(tau, 1)
            # Check convergence
            converged = np.all(
                ceil(n_samples / sampler.nwalkers) * tau < sampler.iteration
            )
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            old_tau = tau
            print(
                f"tau: {tau}, effective sample size: {sampler.iteration * sampler.nwalkers // tau.max()}"
            )
        except emcee.autocorr.AutocorrError:
            print("AutocorrError")
            converged = False

        if converged:
            break
    return sampler


def try_multiprocessing(
    task, moves, n_walkers, n_samples, start, burn_in, backend, max_workers
):
    """Attempt to use ``multiprocessing`` library to run Emcee in parallel."""
    try:  # Try to run with multiprocessing
        if task.mcmc_info.get("separate_tasks"):
            # Initialize separate task objects for each worker process,
            # and use more advanced multiprocessing library.
            # (needed for multiple MATLAB engines)
            with loky.get_reusable_executor(
                max_workers=max_workers,
                initializer=separate_tasks,
                initargs=(task.cfg,),
                timeout=100,
                reuse=True,
            ) as pool:
                sampler = emcee.EnsembleSampler(
                    n_walkers,
                    task.D,
                    separate_log_joint,
                    moves=moves,
                    pool=pool,
                    backend=backend,
                )
                sampler = run_sampler_until_convergence(
                    sampler, n_samples, burn_in, start
                )
        else:
            with loky.get_reusable_executor(max_workers) as pool:
                sampler = emcee.EnsembleSampler(
                    n_walkers,
                    task.D,
                    log_joint,
                    args=[task],
                    moves=moves,
                    pool=pool,
                    backend=backend,
                )
                sampler = run_sampler_until_convergence(
                    sampler, n_samples, burn_in, start
                )
    except PicklingError as err:
        logging.warn(f"PicklingError {err}")
        logging.warn(
            "\nAttempting to continue sampling without multiprocessing..."
            + "\n"
        )
        sampler = emcee.EnsembleSampler(
            n_walkers,
            task.D,
            log_joint,
            args=[task],
            moves=moves,
            backend=backend,
        )
        sampler = run_sampler_until_convergence(
            sampler, n_samples, burn_in, start
        )
    return sampler


def separate_tasks(task_cfg):
    """Initialize separate tasks for each multiprocessing instance.

    Otherwise all tasks will try to access the same MATLAB engine, losing most
    of the parallel advantage.
    """
    task = benchflow.utilities.cfg.cfg_to_task(task_cfg)
    if task.mcmc_info.get("bounded"):
        # Transform to unconstrained coordinates for MCMC
        task._transform_task_to_unconstrained()
    loky._TASK = task
    loky._COUNT = 0  # Record the number of function evaluations.


def log_joint(theta, task):
    """Evaluate the log joint on the appropriate MATLAB instance."""
    log_likelihoods = task.log_likelihood(theta)
    log_priors = task.log_prior(theta)
    if not np.isscalar(log_priors) and not np.isscalar(log_likelihoods):
        assert log_likelihoods.shape == log_priors.shape, (
            "likelihood(s) and prior(s) have incompatible shapes: "
            f"{log_likelihoods.shape} and {log_priors.shape}."
        )
    return log_likelihoods + log_priors, log_priors


def separate_log_joint(theta):
    """Evaluate the log joint on the appropriate MATLAB instance."""
    loky._COUNT += 1
    task = loky._TASK
    log_likelihoods = task.log_likelihood(theta)
    log_priors = task.log_prior(theta)
    if not np.isscalar(log_priors) and not np.isscalar(log_likelihoods):
        assert log_likelihoods.shape == log_priors.shape, (
            "likelihood(s) and prior(s) have incompatible shapes: "
            f"{log_likelihoods.shape} and {log_priors.shape}."
        )
    return log_likelihoods + log_priors, log_priors


if __name__ == "__main__":
    main()
