import os
import sys
import traceback

import hydra

import benchflow
from benchflow.algorithms import GenerateInitialSet
from benchflow.utilities.cfg import cfg_to_algorithm

"""Example usage: python benchflow --multirun +seed=range(0,3)"""


def run_config(cfg):
    algo = cfg_to_algorithm(cfg)
    if isinstance(algo, GenerateInitialSet):
        algo.run()
    else:
        checkpoint_path = cfg["algorithm"].get("checkpoint_path", None)
        if checkpoint_path is not None:
            # Some algorithms may support loading from a checkpoint
            posterior = algo.run(checkpoint_path=checkpoint_path)
        else:
            posterior = algo.run()

        if posterior is not None:
            if cfg.get("post_process", True):
                posterior.post_process()
            posterior.compute_metrics(cfg.get("eval_iter", 1))
            save_options = cfg.get("save_options", {})
            posterior.save_results(**save_options)


if __name__ == "__main__":
    config_dir = os.path.join(os.path.dirname(benchflow.__file__), "config")
    try:
        hydra.main(
            version_base=None, config_path=config_dir, config_name="default"
        )(run_config)()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise
