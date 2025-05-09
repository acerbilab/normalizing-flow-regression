import numpy as np
from scipy.special import erfc, erfcinv

from .utils import handle_0D_1D_input


class ParameterTransformer:
    """
    A class used to enable transforming of variables from unconstrained to
    constrained space and vice versa.

    Parameters
    ----------
    D : int
        The dimension of the space.
    lb_orig : np.ndarray, optional
        The lower bounds of the space. ``lb_orig`` and ``ub_orig`` define a set
        of strict lower and upper bounds for each parameter, given in the
        original space. By default `None`.
    ub_orig : np.ndarray, optional
        The upper bounds of the space. ``lb_orig`` and ``ub_orig`` define a set
        of strict lower and upper bounds for each parameter, given in the
        original space. By default `None`.
    plb_orig : np.ndarray, optional
        The plausible lower bounds such that ``lb_orig < plb_orig < pub_orig <
        ub_orig``. ``plb_orig`` and ``pub_orig`` represent a "plausible" range
        for each parameter, given in the original space. By default `None`.
    pub_orig : np.ndarray, optional
        The plausible upper bounds such that ``lb_orig < plb_orig < pub_orig <
        ub_orig``. ``plb_orig`` and ``pub_orig`` represent a "plausible" range
        for each parameter, given in the original space. By default `None`.
    bounded_transform_type : str, optional
        A string indicating the type of transform for bounded variables: one of
        ["logit", ("norminv" || "probit"), "student4"]. Default "probit".
    """

    def __init__(
        self,
        D: int,
        lb_orig: np.ndarray = None,
        ub_orig: np.ndarray = None,
        plb_orig: np.ndarray = None,
        pub_orig: np.ndarray = None,
        scale: np.ndarray = None,
        rotation_matrix: np.ndarray = None,
        transform_type="probit",
    ):
        self.scale = scale
        self.R_mat = rotation_matrix

        # Empty LB and UB are Infs
        if lb_orig is None:
            lb_orig = np.ones((1, D)) * -np.inf
        if ub_orig is None:
            ub_orig = np.ones((1, D)) * np.inf

        # Empty plausible bounds equal hard bounds
        if plb_orig is None:
            plb_orig = np.copy(lb_orig)
        if pub_orig is None:
            pub_orig = np.copy(ub_orig)

        # Convert scalar inputs to row vectors (I do not think it is necessary)
        if not (
            np.all(lb_orig <= plb_orig)
            and np.all(plb_orig < pub_orig)
            and np.all(pub_orig <= ub_orig)
        ):
            raise ValueError(
                """Variable bounds should be LB <= PLB < PUB <= UB
                for all variables."""
            )

        # Transform to log coordinates
        self.lb_orig = lb_orig
        self.ub_orig = ub_orig

        # Select and validate the type of transform:
        transform_types = {
            "logit": 3,
            "norminv": 12,
            "probit": 12,
            "student4": 13,
        }
        if type(transform_type) is str:
            try:
                bounded_type = transform_types[transform_type]
            except KeyError as exc:
                raise ValueError(
                    f"Unrecognized bounded transform {transform_type}."
                ) from exc
        else:
            if transform_type not in transform_types.values():
                raise ValueError(
                    f"Unrecognized bounded transform {transform_type}."
                )
            bounded_type = transform_type

        # Setup bounded transforms:
        self.bounded_types = [bounded_type]
        self._bounded_transforms = {}
        self._set_bounded_transforms()

        self.type = np.zeros(D)
        for i in range(D):
            if (
                np.isfinite(lb_orig[:, i])
                and np.isfinite(ub_orig[:, i])
                and lb_orig[:, i] < ub_orig[:, i]
            ):
                self.type[i] = bounded_type

        # Centering (at the end of the transform)
        self.mu = np.zeros(D)
        self.delta = np.ones(D)

        # Get transformed PLB and ULB
        if not (
            np.all(plb_orig == self.lb_orig)
            and np.all(pub_orig == self.ub_orig)
        ):
            plb_tran = self.__call__(plb_orig)
            pub_tran = self.__call__(pub_orig)

            # Center in transformed space
            for i in range(D):
                if np.isfinite(plb_tran[0, i]) and np.isfinite(pub_tran[0, i]):
                    self.mu[i] = 0.5 * (plb_tran[0, i] + pub_tran[0, i])
                    self.delta[i] = pub_tran[0, i] - plb_tran[0, i]

    @handle_0D_1D_input(patched_kwargs=["x"], patched_argpos=[0])
    def __call__(self, x: np.ndarray):
        """
        Performs direct transform of original variables ``x`` into
        unconstrained variables ``u``.

        Parameters
        ----------
        x : np.ndarray
            A N x D array, where N is the number of input data
            and D is the number of dimensions

        Returns
        -------
        u : np.ndarray
            The variables transformed to unconstrained variables.
        """
        x = x.astype(float)
        u = np.copy(x)

        # Unbounded scalars (possibly center and rescale)
        mask = self.type == 0
        if np.any(mask):
            u[:, mask] = (x[:, mask] - self.mu[mask]) / self.delta[mask]

        # Lower and upper bounded scalars
        for t in self.bounded_types:
            mask = self.type == t
            if np.any(mask):
                u[:, mask] = self._bounded_transforms[t]["direct"](
                    self, x, mask
                )

        # Rotoscale whitening:
        # Rotate and rescale points in transformed space.
        if self.R_mat is not None:
            u = u @ self.R_mat
        if self.scale is not None:
            u = u / self.scale

        return u

    @handle_0D_1D_input(patched_kwargs=["u"], patched_argpos=[0])
    def inverse(self, u: np.ndarray):
        """
        Performs inverse transform of unconstrained variables ``u`` into
        variables ``x`` in the original space

        Parameters
        ----------
        u : np.ndarray
            The unconstrained variables that will be transformed.

        Returns
        -------
        x : np.ndarray
            The original variables which result of the transformation.
        """
        u = u.astype(float)
        x = np.copy(u)

        # Rotoscale whitening:
        # Undo rescaling and rotation.
        if self.scale is not None:
            x = x * self.scale
        if self.R_mat is not None:
            x = x @ np.transpose(self.R_mat)

        xNew = np.copy(x)

        # Unbounded scalars (possibly unscale and uncenter)
        mask = self.type == 0
        if np.any(mask):
            xNew[:, mask] = x[:, mask] * self.delta[mask] + self.mu[mask]

        # Lower and upper bounded scalars
        for t in self.bounded_types:
            mask = self.type == t
            if np.any(mask):
                xNew[:, mask] = self._bounded_transforms[t]["inverse"](
                    self, x, mask
                )

        return xNew

    @handle_0D_1D_input(
        patched_kwargs=["u"], patched_argpos=[0], return_scalar=True
    )
    def log_abs_det_jacobian(self, u: np.ndarray):
        r"""
        ``log_abs_det_jacobian(u)`` returns the log absolute value of the
        determinant of the Jacobian of the parameter transformation evaluated
        at ``u``, that is :math: `log \|D \du(g^-1(u))\|`.

        Parameters
        ----------
        u : np.ndarray
            The points where the log determinant of the Jacobian should be
            evaluated (in transformed space).

        Returns
        -------
        p : np.ndarray
            The log absolute determinant of the Jacobian.
        """
        u = u.astype(float)
        u_c = np.copy(u)

        # Rotoscale whitening:
        # Undo rescaling and rotation.
        if self.scale is not None:
            u_c = u_c * self.scale
        if self.R_mat is not None:
            u_c = u_c @ np.transpose(self.R_mat)

        p = np.zeros(u_c.shape)

        # Unbounded scalars
        mask = self.type == 0
        if np.any(mask):
            p[:, mask] = np.log(self.delta[mask])[np.newaxis]

        # Lower and upper bounded scalars
        for t in self.bounded_types:
            mask = self.type == t
            if np.any(mask):
                p[:, mask] = self._bounded_transforms[t]["jacobian"](
                    self, u_c, mask
                )

        # Whitening/rotoscaling density correction:
        if self.scale is not None:
            p = p + np.log(self.scale)

        p = np.sum(p, axis=1)
        return p

    def _set_bounded_transforms(self):
        r"""Initialize functions for bounded transform(s) by type.

        Stores the resulting callables in a dictionary
        ``self._bounded_transforms[t][case]``, where ``t`` is the integer name
        of the transform type, and ``case`` is one of ``["direct", "inverse",
        "jacobian"]`` for the corresponding function.
        """
        for t in self.bounded_types:
            self._bounded_transforms[t] = {}
            if t == 3:
                # logit

                def bounded_transform(self, x, mask):
                    return _center(
                        _logit(
                            _to_unit_interval(
                                x[:, mask],
                                self.lb_orig[:, mask],
                                self.ub_orig[:, mask],
                            )
                        ),
                        self.mu[mask],
                        self.delta[mask],
                    )

                self._bounded_transforms[t]["direct"] = bounded_transform

                def bounded_inverse(self, u, mask):
                    return _from_unit_interval(
                        _inverse_logit(
                            _uncenter(
                                u[:, mask], self.mu[mask], self.delta[mask]
                            )
                        ),
                        self.lb_orig[:, mask],
                        self.ub_orig[:, mask],
                    )

                self._bounded_transforms[t]["inverse"] = bounded_inverse

                def bounded_jacobian(self, u, mask):
                    j1 = np.log(self.ub_orig[:, mask] - self.lb_orig[:, mask])
                    y = _uncenter(u[:, mask], self.mu[mask], self.delta[mask])
                    z = -np.log1p(np.exp(-y))
                    j2 = -y + 2 * z
                    j3 = np.log(self.delta[mask])
                    return j1 + j2 + j3

                self._bounded_transforms[t]["jacobian"] = bounded_jacobian

            elif t == 12:
                # probit: inverse normal CDF (probit) transform (default)

                def bounded_transform(self, x, mask):
                    return _center(
                        _probit(
                            _to_unit_interval(
                                x[:, mask],
                                self.lb_orig[:, mask],
                                self.ub_orig[:, mask],
                            )
                        ),
                        self.mu[mask],
                        self.delta[mask],
                    )

                self._bounded_transforms[t]["direct"] = bounded_transform

                def bounded_inverse(self, u, mask):
                    return _from_unit_interval(
                        _inverse_probit(
                            _uncenter(
                                u[:, mask], self.mu[mask], self.delta[mask]
                            )
                        ),
                        self.lb_orig[:, mask],
                        self.ub_orig[:, mask],
                    )

                self._bounded_transforms[t]["inverse"] = bounded_inverse

                def bounded_jacobian(self, u, mask):
                    j1 = np.log(self.ub_orig[:, mask] - self.lb_orig[:, mask])
                    y = _uncenter(u[:, mask], self.mu[mask], self.delta[mask])
                    j2 = -0.5 * np.log(2 * np.pi) - 0.5 * y**2
                    j3 = np.log(self.delta[mask])
                    return j1 + j2 + j3

                self._bounded_transforms[t]["jacobian"] = bounded_jacobian

            elif t == 13:
                # student4: Student's T with nu=4 CDF transform

                def bounded_transform(self, x, mask):
                    return _center(
                        _student4(
                            _to_unit_interval(
                                x[:, mask],
                                self.lb_orig[:, mask],
                                self.ub_orig[:, mask],
                            )
                        ),
                        self.mu[mask],
                        self.delta[mask],
                    )

                self._bounded_transforms[t]["direct"] = bounded_transform

                def bounded_inverse(self, u, mask):
                    return _from_unit_interval(
                        _inverse_student4(
                            _uncenter(
                                u[:, mask], self.mu[mask], self.delta[mask]
                            )
                        ),
                        self.lb_orig[:, mask],
                        self.ub_orig[:, mask],
                    )

                self._bounded_transforms[t]["inverse"] = bounded_inverse

                def bounded_jacobian(self, u, mask):
                    j1 = np.log(self.ub_orig[:, mask] - self.lb_orig[:, mask])
                    y = _uncenter(u[:, mask], self.mu[mask], self.delta[mask])
                    j2 = np.log(3 / 8) - (5 / 2) * np.log1p(y**2 / 4)
                    j3 = np.log(self.delta[mask])
                    return j1 + j2 + j3

                self._bounded_transforms[t]["jacobian"] = bounded_jacobian

            else:
                raise NotImplementedError

    def __eq__(self, other):
        return (
            np.all(self.scale == self.scale)
            and np.all(self.R_mat == other.R_mat)
            and np.all(self.lb_orig == other.lb_orig)
            and np.all(self.ub_orig == other.ub_orig)
            and np.all(self.bounded_types == other.bounded_types)
            and np.all(
                self._bounded_transforms.keys()
                == other._bounded_transforms.keys()
            )
            and np.all(self.type == other.type)
            and np.all(self.mu == other.mu)
            and np.all(self.delta == other.delta)
        )


def _to_unit_interval(x, lb, ub, safe=True):
    z = (x - lb) / (ub - lb)
    if safe:  # Nudge points away from boundary
        mask = (z == 0) & (x != lb)
        if np.any(mask):
            z[mask] = np.nextafter(0, np.inf)
        mask = (z == 1) & (x != ub)
        if np.any(mask):
            z[mask] = np.nextafter(1, -np.inf)
    return z


def _from_unit_interval(z, lb, ub, safe=True):
    y = z * (ub - lb) + lb
    if safe:  # Nudge points away from boundary
        y = np.maximum(y, np.nextafter(lb, np.inf))
        y = np.minimum(y, np.nextafter(ub, -np.inf))
    return y


def _center(u, mu, delta):
    return (u - mu) / delta


def _uncenter(v, mu, delta):
    return v * delta + mu


def _logit(z):
    # prevent divide by zero
    u = np.zeros_like(z)
    u[z == 0] = -np.inf
    u[z == 1] = np.inf

    u[u == 0] = np.log(z[u == 0] / (1 - z[u == 0]))
    return u


def _inverse_logit(u):
    # prevent overflow
    z = np.zeros_like(u)
    mask = -u > np.log(np.finfo(np.float64).max)
    z[mask] = 0.0
    z[~mask] = 1 / (1 + np.exp(-u[~mask]))
    return z


def _probit(z):
    return -np.sqrt(2) * erfcinv(2 * z)


def _inverse_probit(u):
    return 0.5 * erfc(-u / np.sqrt(2))


def _student4(z):
    aa = np.sqrt(4 * z * (1 - z))
    # prevent divide by zero
    mask = aa == 0.0
    q = np.zeros_like(z)
    q[mask] = np.inf
    q[~mask] = np.cos(np.arccos(aa[~mask]) / 3) / aa[~mask]
    return np.sign(z - 0.5) * (2 * np.sqrt(q - 1))


def _inverse_student4(u):
    t2 = u**2
    return 0.5 + (3 / 8) * (u / np.sqrt(1 + t2 / 4)) * (
        1 - t2 / (1 + t2 / 4) / 12
    )
