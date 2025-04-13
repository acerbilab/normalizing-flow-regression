from .posterior import Posterior


class FlowRegressionPosterior(Posterior):
    """Class for accessing inferred ``pyvbmc`` posteriors.

    Attributes
    ----------
    nfr_object : nfr.NFR
        The ``NFR`` object used to execute the inference.
    final_flow :
        The final (best) flows computed by NFR.
    final_lml : float
        An estimate of the final LML for the returned ``flow``.
    final_success_flag : bool
        ``final_success_flag`` is ``True`` if the inference reached stability within
        the provided budget of function evaluations, suggesting convergence.
        If ``False``, the returned solution has not stabilized and should
        not be trusted.
    final_result_dict : dict
        A dictionary with additional information about the NFR run.
    history : [Posterior]
        A list of ``FlowPosterior`` objects for each iteration of the
        algorithm.
    metrics : dict
        A dictionary containing the computed metrics. Keys are metrics names
        (e.g.  "c2st") and values are lists of computed metrics (one for each
        ``Posterior`` object in ``self.history``, and one for the final
        ``Posterior``)
    cfg : omegaconf.DictConfig
        The original ``hydra`` config describing the run.
    task : benchflow.task.Task
        The target ``Task`` for inference.
    """

    def __init__(
        self,
        cfg,
        task,
        final_surrogate,
        final_lml,
        final_success_flag=None,
        final_result_dict=None,
        nfr_object=None,
        iteration=None,
    ):
        super().__init__(cfg, task)
        self.nfr_object = nfr_object
        self.iteration = iteration
        self.final_surrogate = final_surrogate
        self.final_lml = final_lml
        self.final_success_flag = final_success_flag
        self.final_result_dict = final_result_dict

    def sample(self, n_samples=10000):
        """Draw samples from the final posterior surrogate.

        Parameters
        ----------
        n_samples : int
            The number of samples to draw.

        Returns
        -------
        samples : np.array
            The posterior samples, shape ``(n_samples, D)`` where ``D`` is the
            task dimension.
        """
        samples = self.final_surrogate.sample(n_samples)
        # To original constrained space
        samples = self.final_surrogate.parameter_transformer.inverse(samples)
        return samples

    def get_lml_estimate(self):
        """Get the estimated log marginal likelihood (LML)."""
        return self.final_lml

    def get_lml_sd(self):
        """Get the standard deviation of the estimated LML."""
        raise NotImplementedError()
