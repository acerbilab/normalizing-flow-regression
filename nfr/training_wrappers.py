import logging
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar
from torch.autograd import Variable

from nflows.flows.base import Flow
from nflows.transforms.base import CompositeTransform
from nfr.nfr_helpers import noise_shaping
from nfr.utils import wandb_log

from .lbfgs import LBFGS

logger = logging.getLogger("NFR_debug")


class FlowEnsembles(nn.Module):
    def __init__(
        self,
        base_dists: list,
        transforms: list,
        options,
        D: int,
        lnZ=None,
    ) -> None:
        super().__init__()
        self.D = D
        self.n_ensembles = len(transforms)
        self.flows = nn.ModuleList(
            [
                Flow(
                    CompositeTransform(transforms[i]),
                    base_dists[i],
                )
                for i in range(self.n_ensembles)
            ]
        )
        self.momentum = options.get("momentum")
        self.weight_decay = options.get("weight_decay")
        if torch.cuda.is_available():
            self.use_cuda = True
            self.device = torch.device("cuda")
        else:
            self.use_cuda = False
            self.device = torch.device("cpu")
        for i in range(len(self.flows)):
            self.flows[i] = self.flows[i].to(self.device)

        self._optimize_lnZ_flag = options.get("optimize_lnZ", False)
        if lnZ is not None:
            # The normalizing constant is provided. For debugging purpose.
            self.raw_lnZ = torch.tensor(
                [lnZ], requires_grad=False, device=self.device
            )
            self.lnZ_constraint = None
            self._optimize_lnZ_flag = False
            assert not options.get("annealed_target")
        else:
            # Add log normalization constant parameters
            self.raw_lnZ = nn.Parameter(
                torch.zeros(1, requires_grad=True, device=self.device)
            )  # unconstrained
            self.lnZ_constraint = None

        self.options = options
        self.batch_size = self.options.get("batch_size", 32)
        self.num_epochs = self.options.get("num_epochs", 500)

        self._total_loss_calls = 0
        self._fit_loss_calls = 0

        self.cache = {}

        self.parameter_transformer = None
        self.bounds_info = {
            "need_rejection_sampling": False,
            "lb": None,
            "ub": None,
        }

        self._info = {}

    def get_info(self):
        info = self._info
        info.update(self.bounds_info)
        return info

    def update_variables(self, X_train, Y_train, S_train, S_train_raw):
        """Some variables need to be updated according to current dataset."""
        # Update number of training points
        self.N_train = X_train.shape[0]
        # Update the maximum of the lower confidence bound for log density values
        self.y_lcb_max = (Y_train - 1.96 * S_train_raw).max()
        self.y_max = Y_train.max()

    @property
    def optimize_lnZ_flag(self):
        return self._optimize_lnZ_flag

    @optimize_lnZ_flag.setter
    def optimize_lnZ_flag(self, value: bool):
        assert isinstance(value, bool)
        self._optimize_lnZ_flag = value
        if value:
            self.raw_lnZ.requires_grad_(True)
        else:
            self.raw_lnZ.requires_grad_(False)

    @property
    def lnZ(self):
        if self.lnZ_constraint is not None:
            return self.lnZ_constraint.transform(self.raw_lnZ)
        else:
            return self.raw_lnZ

    def set_lnZ(self, lnZ):
        with torch.no_grad():
            lnZ = torch.as_tensor(lnZ).to(self.device)
            if self.lnZ_constraint is not None:
                raw_lnZ = self.lnZ_constraint.inverse_transform(lnZ)
            else:
                raw_lnZ = lnZ
            self.raw_lnZ.fill_(raw_lnZ.item())

    def set_lnZ_constraint(self, Y_train, S_train_raw):
        """Deprecated. The lnZ constraint is not used anymore."""
        with torch.no_grad():
            # TODO: the upper bound is not appropriate. 1. It assumes the MAP estimate is in current set. However, if the input space is transformed, the MAP estimate in the new space will not be in the current train set anymore. The log-density for current points can change arbitraly but the true lnZ is the same. 2. If the prior is normalized, we can use the likelihood's maximum as the upper bound for the target posterior. However, the target is often tempered with p0 and \beta, and after tempering the normalizing constant seems not possible to estimate, i.e., there is no relationship between tempered target's lnZ and (target's lnZ + p0's normalizing constant).
            upper_bound = torch.max(Y_train + S_train_raw).item() + 1
            # Get lnZ
            lnZ = self.lnZ
            # Set new constraint
            assert self.options.get("constrain_lnZ") is False, (
                "lnZ constraint is not used."
            )
            self.lnZ_constraint = None
            # if self.options.get("constrain_lnZ"):
            #     self.lnZ_constraint = LessThan(upper_bound)
            #     lnZ = lnZ.clip(max=upper_bound - 1e-3)
            #     raw_lnZ = self.lnZ_constraint.inverse_transform(lnZ)
            #     self.raw_lnZ.fill_(raw_lnZ.item())
            # else:
            #     self.lnZ_constraint = None

    def fit(self, X_train, Y_train, S_train, S_train_raw, opt_method="lbfgs"):
        """
        S_train: possibly added shaping noise
        S_train_raw: original noise
        """
        self._fit_loss_calls = 0  # Reset to count from 0
        self.update_variables(X_train, Y_train, S_train, S_train_raw)

        self.set_lnZ_constraint(Y_train, S_train_raw)

        self.create_opt(opt_method=opt_method)
        self.flows.train()
        logger = logging.getLogger("NFR_debug")

        # Fit lnZ
        if self.optimize_lnZ_flag:
            self.optimize_lnZ(
                X_train, Y_train, S_train, S_train_raw, self.D, logger
            )
        # Fit lnZ + flow parameters
        self._fit_all(X_train, Y_train, S_train, S_train_raw)
        return

    def _fit_all(self, X_train, Y_train, S_train, S_train_raw, n_anneal=0):
        self.flows.train()
        assert self.n_ensembles == 1, "Only one flow is supported."
        i = 0
        if self.opt_method == "lbfgs":

            def closure(i=i):
                self.optimizers[i].zero_grad()
                loss = self.compute_loss(
                    X_train, Y_train, S_train, S_train_raw, i
                )
                loss.backward()
                return loss

            self.optimizers[i].step(closure)
        else:
            raise NotImplementedError
        return

    def compute_per_point_log_likelihoods(self, preds, y, sigma, sigma_raw):
        """
        preds: flow's predicted log pdfs + lnZ
        y: observed unnormalized log pdfs
        sigma: noise std (including added shaping noise)
        sigma_raw: original noise std
        """
        log_likes = _compute_likelihood(
            preds,
            y,
            sigma,
            sigma_raw,
            self.options,
            self.y_lcb_max,
            self.y_max,
        )
        return log_likes

    def compute_loss(
        self,
        x,
        y,
        sigma,
        sigma_raw,
        i,
        annealed_weight=1.0,
        logger=None,
        debug=False,
    ):
        lnZ = self.lnZ
        log_pdf_vals = self.flows[i].log_prob(x)
        preds = log_pdf_vals.unsqueeze(-1) + lnZ

        weights = torch.ones_like(y)
        log_likes = self.compute_per_point_log_likelihoods(
            preds, y, sigma, sigma_raw
        )
        loss_pp = -log_likes  # per point loss
        loss = (weights * loss_pp).sum()

        loss_reg = 0
        for p in self.flows[i].parameters():
            loss_reg = loss_reg + p.square().sum()
        loss_reg = self.options["lambd_weight"] * loss_reg
        loss = loss + loss_reg

        info_to_print = f"total loss: {loss.item():.1f}, loss_reg: {loss_reg.item():.1f}, lnZ: {self.lnZ.item():.2f}"

        if self._fit_loss_calls == 0:
            logger = logging.getLogger("NFR_debug")
        if logger is not None:
            logger.info(info_to_print)
        if self._fit_loss_calls % self.options.get("wandb_log_steps", 1) == 0:
            info_wandb = {
                "fit_loss_num_calls": self._fit_loss_calls,
                "loss": loss.item(),
                "loss_reg": loss_reg.item(),
                "lnZ": self.lnZ.item(),
            }
            wandb_log(info_wandb)

        self._total_loss_calls += 1
        self._fit_loss_calls += 1

        if debug:
            return loss, {
                "loss_pp": loss_pp,
                "weights": weights,
                "loss_reg": loss_reg,
            }
        else:
            return loss

    def create_opt(self, opt_method="lbfgs"):
        self.opt_method = opt_method
        all_opt_params_ensemble = [[] for i in range(self.n_ensembles)]
        for i in range(self.n_ensembles):
            all_opt_params_ensemble[i] = list(self.flows[i].parameters()) + [
                self.raw_lnZ
            ]
        if opt_method == "lbfgs":
            max_iter = self.options.get("lbfgs_max_iter")
            self.optimizers = [
                LBFGS(
                    all_opt_params_ensemble[i],
                    max_iter=max_iter,
                    max_eval=self.options.get("lbfgs_max_eval"),
                    tolerance_grad=1e-3,
                    tolerance_change=self.options.get(
                        "lbfgs_tolerance_change", 1e-9
                    ),
                    line_search_fn="strong_wolfe",
                    tolerance_history_length=self.options.get(
                        "lbfgs_tolerance_history_length", 2
                    ),
                    abs_tol=self.options.get("lbfgs_abs_tol"),
                )
                for i in range(self.n_ensembles)
            ]
        else:
            raise NotImplementedError

    def _sample(
        self,
        num_samples: int,
        to_numpy: bool = True,
        balance_flag: bool = True,
    ):
        with torch.no_grad():
            samples_list = [
                self.flows[i].sample(num_samples)
                for i in range(self.n_ensembles)
            ]
        probs = np.ones(self.n_ensembles) / self.n_ensembles

        if balance_flag:
            nums = (num_samples * probs).astype(int)
            ind = np.argmax(nums)
            nums[ind] = num_samples - np.sum(nums) + nums[ind]
            samples = []
            for j, num in enumerate(nums):
                samples.append(samples_list[j][:num])
            samples = torch.cat(samples)
        else:
            raise NotImplementedError
        if to_numpy:
            samples = samples.cpu().detach().numpy()
        return samples

    def sample(
        self,
        num_samples: int,
        to_numpy: bool = True,
        balance_flag: bool = True,
    ):
        if self.bounds_info["need_rejection_sampling"]:
            assert to_numpy, (
                "only to_numpy=True is supported as we need to reject samples that are outside the bounds."
            )
            # Reject samples that are outside the bounds
            N = 0
            i = 0
            accept_rates = []
            samples_orig = []
            while N < num_samples:
                samples_tmp = self._sample(num_samples, to_numpy, balance_flag)
                if self.parameter_transformer is not None:
                    samples_orig_tmp = self.parameter_transformer.inverse(
                        samples_tmp
                    )
                else:
                    samples_orig_tmp = samples_tmp
                samples_orig_tmp, rate = reject_outside(
                    samples_orig_tmp,
                    self.bounds_info["lb"],
                    self.bounds_info["ub"],
                )
                accept_rates.append(rate)

                N += samples_orig_tmp.shape[0]
                samples_orig.append(samples_orig_tmp)
                i += 1
                if i > 30:
                    logger.warning(
                        f"Rejection sampling is taking too long. Accepted sample size {N} out of {len(accept_rates) * num_samples}."
                    )
                    break
            samples_orig = np.concatenate(samples_orig)
            samples_orig = samples_orig[:num_samples]
            if self.parameter_transformer is not None:
                samples = self.parameter_transformer(samples_orig)
            else:
                samples = samples_orig

            total_num_samples = len(accept_rates) * num_samples
            num_accepted = samples_orig.shape[0]
            mean_accept_rate = np.mean(accept_rates)
            logger.info(f"acceptance: {mean_accept_rate}")
            if self._info.get("acceptance_rate") is None:
                self._info["acceptance_rate"] = (
                    mean_accept_rate,
                    len(accept_rates) * num_samples,
                )
            else:
                # Update acceptance rate
                old_acceptance_rate, old_total_num_samples = self._info[
                    "acceptance_rate"
                ]
                new_acceptance_rate = (
                    old_acceptance_rate * old_total_num_samples + num_accepted
                ) / (old_total_num_samples + total_num_samples)
                self._info["acceptance_rate"] = (
                    new_acceptance_rate,
                    old_total_num_samples + total_num_samples,
                )
        else:
            samples = self._sample(num_samples, to_numpy, balance_flag)
        return samples

    def _log_prob(self, x):
        (x,) = to_variable(var=(x,), cuda=self.use_cuda)
        if x.ndim == 1:
            X = x.unsqueeze(0)
        else:
            X = x
        act_vec = [self.flows[i].log_prob(X) for i in range(self.n_ensembles)]
        act_vec = torch.stack(act_vec, 0)
        q = torch.ones(self.n_ensembles, device=self.device) / self.n_ensembles
        while len(q.shape) < len(act_vec.shape):
            q = q.unsqueeze(1)

        assert act_vec.ndim == 2
        prob = torch.logsumexp(torch.log(q) + act_vec, 0)
        return prob

    def forward(self, x):
        return self._log_prob(x)

    def log_prob(self, x, no_grad=True):
        self.flows.eval()
        with torch.no_grad() if no_grad else nullcontext():
            return self._log_prob(x)

    def pdf(self, x, no_grad=True):
        self.flows.eval()
        with torch.no_grad() if no_grad else nullcontext():
            (x,) = to_variable(var=(x,), cuda=self.use_cuda)
            if x.ndim == 1:
                X = x.unsqueeze(0)
            else:
                X = x
            act_vec = [
                self.flows[i].log_prob(X) for i in range(self.n_ensembles)
            ]
            act_vec = torch.stack(act_vec, 0)
            q = torch.ones(self.n_ensembles) / self.n_ensembles
            while len(q.shape) < len(act_vec.shape):
                q = q.unsqueeze(1)

            assert act_vec.ndim == 2
            prob = torch.sum(q * torch.exp(act_vec), 0)
            return prob

    def maximum_likelihood_density_estimation(self, samples):
        assert self.n_ensembles == 1
        flow = self.flows[0]
        flow.train()
        optimizer = LBFGS(
            flow.parameters(),
            max_iter=self.options.get("mle_max_iter"),
            max_eval=self.options.get("lbfgs_max_eval"),
            tolerance_grad=1e-3,
            tolerance_change=self.options.get("lbfgs_tolerance_change", 1e-9),
            line_search_fn="strong_wolfe",
            tolerance_history_length=self.options.get(
                "lbfgs_tolerance_history_length", 2
            ),
            abs_tol=self.options.get("lbfgs_abs_tol"),
        )

        def closure():
            optimizer.zero_grad()
            loss = -flow.log_prob(samples).sum()
            loss_reg = 0
            for p in flow.parameters():
                loss_reg = loss_reg + p.square().sum()
            loss_reg = loss_reg
            loss = loss + self.options["lambd_weight"] * loss_reg
            loss.backward()
            wandb_log({"mle_density_estimation_loss": loss.item()})
            return loss

        optimizer.step(closure)

    def moments(self, N: int = int(1e5), cov_flag=False):
        """
        Estimate mean and covariance by samples.
        """
        X = self.sample(N)
        mu = np.mean(X, axis=0)
        if cov_flag:
            cov = np.cov(X.T)
            return mu, cov
        return mu

    def optimize_lnZ(self, x, y, sigma, sigma_raw, D, logger):
        if self.n_ensembles != 1:
            raise NotImplementedError(
                "`optimize_lnZ` only implemented for a single-flow."
            )
        with torch.no_grad():
            log_pdf_vals = self.flows[0].log_prob(x).unsqueeze(-1)

            def loss(raw_lnZ):
                raw_lnZ = torch.as_tensor(raw_lnZ).to(self.device)
                if self.lnZ_constraint is not None:
                    lnZ = self.lnZ_constraint.transform(raw_lnZ)
                else:
                    lnZ = raw_lnZ
                log_likes = self.compute_per_point_log_likelihoods(
                    log_pdf_vals + lnZ, y, sigma, sigma_raw
                )
                loss = -log_likes.sum()
                return loss.cpu().numpy().item()

            current_raw_lnZ = self.raw_lnZ.item()
            if self.lnZ_constraint is not None:
                max_less_D = self.lnZ_constraint.inverse_transform(
                    torch.max(y).item() - D
                ).item()
            else:
                max_less_D = torch.max(y).item() - D

            opt = minimize_scalar(
                loss, method="brent", bracket=(current_raw_lnZ, max_less_D)
            )
        if opt.success and opt.fun < loss(self.raw_lnZ.item()):
            with torch.no_grad():
                self.raw_lnZ.fill_(opt.x)
            logger.info(f"Optimized lnZ to {self.lnZ.item()}.")
        else:
            logger.info("Failed to optimize lnZ.")

    def transform(self, X):
        """Transform X from inference space to base space."""
        assert self.n_ensembles == 1
        X = torch.atleast_2d(X)
        flow: Flow = self.flows[0]
        X_base, logabsdet = flow._transform(X)[:2]
        return X_base, logabsdet

    def inverse_transform(self, X_base):
        """Transform X from base space to inference space."""
        assert self.n_ensembles == 1
        X_base = torch.atleast_2d(X_base)
        flow: Flow = self.flows[0]
        X, logabsdet = flow._transform.inverse(X_base)[:2]
        return X, logabsdet

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError as e:
            try:
                return self.__getattribute__(name)
            except AttributeError:
                raise e from None


def _inverse_softplus(x):
    return x + torch.log(-torch.expm1(-x))


def _compute_likelihood(preds, y, sigma, sigma_raw, options, y_lcb_max, y_max):
    # dist = torch.distributions.normal.Normal(preds, sigma)
    if (
        options.get("loss_type") == "MSE"
        or options.get("loss_type") == "NormalLogPDF"
    ):
        log_likes = (
            -0.5 * (preds - y) ** 2 / sigma**2
        )  # per point log likelihoods
        return log_likes

    # Tobit-like likelihoods
    assert y_lcb_max is not None
    delta_thresh = options["low_region_delta_thresh"]
    lb = y_lcb_max - delta_thresh  # lower bound for Tobit
    # Get the added shaping noise at the Tobit threshold
    s2_shaping_max = noise_shaping(
        np.zeros(1), lb.cpu().numpy(), options, y_max.item()
    )
    s2_shaping_max = s2_shaping_max.item()
    tobit_scale = torch.sqrt(s2_shaping_max + (sigma_raw**2))

    pdf_sigma = options.get("pdf_sigma", 1e-2)
    if options.get("loss_type") == "Tobit":
        log_likes = torch.where(
            y >= lb,
            -0.5 * (preds - y) ** 2 / sigma**2,
            torch.special.log_ndtr((lb - preds) / tobit_scale),
        )
    elif options.get("loss_type") == "TobitGaussian":
        log_likes = torch.where(
            y >= lb,
            -0.5 * (preds - y) ** 2 / sigma**2,
            torch.special.log_ndtr((lb - preds) / tobit_scale)
            - 0.5 * (preds - y) ** 2 / sigma**2,
        )

        if np.isfinite(options["super_low_region_delta_thresh"]):
            # For super low regions, use a pure Tobit likelihood with scale=delta_thresh
            lb_super_low = y_lcb_max - options["super_low_region_delta_thresh"]
            log_likes = torch.where(
                y <= lb_super_low,
                torch.special.log_ndtr((lb_super_low - preds) / delta_thresh),
                log_likes,
            )
    elif options.get("loss_type") == "NormalPDF":
        # if torch.any(sigma != 0.0):
        #    raise NotImplementedError("Noise not implemented for NormalPDF likelihood")
        # TODO: this is not supposed to work with minibatches since y.max() would be different for each batch
        log_likes = (
            -0.5
            * ((preds - y.max()).exp() - (y - y.max()).exp()) ** 2
            / pdf_sigma**2
            + y
        )
        if options.get("truncate_normal_pdf"):
            alpha = -preds.exp() / pdf_sigma
            lnZ = torch.special.log_ndtr(-alpha)
            log_likes -= lnZ
    elif options.get("loss_type") == "NormalPDFSqrt":
        log_likes = (
            -0.5
            * (((preds - y.max()) / 2).exp() - ((y - y.max()) / 2).exp()) ** 2
            / pdf_sigma**2
            + y / 2
        )
    elif options.get("loss_type") == "LaplacePDFSqrt":
        b = pdf_sigma / np.sqrt(2)
        log_likes = (
            -(((preds - y.max()) / 2).exp() - ((y - y.max()) / 2).exp()).abs()
            / b
            + y / 2
        )
    else:
        raise NotImplementedError
    return log_likes


def reject_outside(samples_orig, lb, ub):
    inds = (lb <= samples_orig).all(-1) & (ub >= samples_orig).all(-1)
    samples_orig = samples_orig[inds]
    rate = inds.sum() / inds.size
    logger.info(f"acceptance: {inds.sum()} / {inds.size}")
    return samples_orig, rate


def to_variable(var=(), cuda=True, volatile=False):
    out = []
    for v in var:
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)
        if not v.is_cuda and cuda:
            v = v.cuda()
        if not isinstance(v, Variable):
            v = Variable(v, volatile=volatile)
        out.append(v)
    return out
