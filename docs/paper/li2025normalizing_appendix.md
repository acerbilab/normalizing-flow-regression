# Normalizing Flow Regression for Bayesian Inference with Offline Likelihood Evaluations - Appendix

---

#### Page 19

# Appendix A. 

This appendix provides additional details and analyses to complement the main text, included in the following sections:

- Normalizing flow regression algorithm details, A. 1
- Metrics description, A. 2
- Real-world problems description, A. 3
- Additional experimental results, A. 4
- Black-box variational inference implementation, A. 5
- Limitations, A. 6
- Ablation studies, A. 7
- Diagnostics, A. 8
- Visualization of posteriors, A. 9

## A.1. Normalizing flow regression algorithm details

Inference space. NFR, VSBQ, Laplace approximation, and BBVI all operate in an unbounded parameter space, which we call the inference space. Originally bounded parameters are first mapped to the inference space and then rescaled and shifted based on user-specified plausible ranges, such as the $68.2 \%$ percentile interval of the prior. After transformation, the plausible ranges in the inference space are standardized to $[-0.5,0.5]$. An appropriate Jacobian correction is applied to the log-density values in the inference space. Similar transformations are commonly used in probabilistic inference software (Carpenter et al., 2017; Huggins et al., 2023). The approximate posterior samples are transformed back to the original space via the inverse transform for performance evaluation against the ground truth posterior samples.

Noise shaping hyperparameter choice for NFR. The function $s(\cdot)$ in Eq. 8 acts as a noise shaping mechanism that increases observation uncertainty for lower-density regions, further preventing overfitting to low-density observations (Li et al., 2024). It is worth noting that the noise shaping mechanism introduces artificial noise even when the density is measured exactly, and this is a feature of the algorithm designed to reduce the undesired influence of low-density observations. We define $s(\cdot)$ as a piecewise linear function,

$$
s\left(f_{\max }-f_{n}\right)= \begin{cases}0 & \text { if } f_{\max }-f_{n}<\delta_{1} \\ \lambda\left(f_{\max }-f_{n}-\delta_{1}\right) & \text { if } \delta_{1} \leq f_{\max }-f_{n} \leq \delta_{2} \\ \lambda\left(\delta_{2}-\delta_{1}\right) & \text { if } f_{\max }-f_{n}>\delta_{2}\end{cases}
$$

Here, $\delta_{1}$ and $\delta_{2}$ define the thresholds for moderate and extremely low log-density values, respectively. In practice, we approximate the unknown difference $f_{\max }-f_{n}$ with $y_{\max }-y_{n}$, where $y_{\max }=\max _{n} y_{n}$ is the maximum observed log-density value. We set $y_{\text {low }}=\max _{n}\left(y_{n}-\right.$

---

#### Page 20

$1.96 \sigma_{n})-\delta_{2}$ for Eq. 8. For all problems, we set $\lambda=0.05$ following Li et al. (2024). The thresholds for moderate density and extremely low density are defined as $\delta_{1}=10 D$, $\delta_{2}=50 D$, where $D$ is the target posterior dimension. ${ }^{7}$ The extremely low-density value is computed as $y_{\text {low }}=\max _{n}\left(y_{n}-1.96 \sigma_{n}\right)-\delta_{2}$.

Normalizing flow architecture specifications. For all experiments, we use the masked autoregressive flow (MAF; Papamakarios et al., 2017) with the original implementation from Durkan et al. (2020). The flow consists of 11 transformation layers, each comprising an affine autoregressive transform followed by a reverse permutation transform. As described in Section 3.3, the flow's base distribution is a diagonal multivariate Gaussian estimated from observations with sufficiently high log-density values. Specifically, we select observations satisfying $y_{n}-1.96 \sigma_{n} \geq \delta_{1}$ and compute the mean and covariance directly from these selected points $\mathbf{x}_{n}$. The maximum scaling factor $\alpha_{\max }$ and $\mu_{\max }$ are chosen such that the normalizing flow exhibits controlled flexibility from the base distribution, as illustrated in Section 4.1. We set $\alpha_{\max }=1.5$ and $\mu_{\max }=1$ (Eq. 9) across the experiments.

Initialization of regression model parameters. The parameter set for the normalizing flow regression model is $\boldsymbol{\psi}=(\boldsymbol{\phi}, C)$, where $\boldsymbol{\phi}$ represents the flow parameters, i.e., the parameters of the neural networks. We initialize $\boldsymbol{\phi}$ by multiplying the default PyTorch initialization (Paszke et al., 2019) by $10^{-3}$ to ensure the flow starts close to its base distribution. The parameter $C$ is initialized to zero.

Termination criteria for normalizing flow regression. For all problems, we set the number of annealed steps $t_{\text {end }}=20$ and the maximum number of training iterations $T_{\max }=$ 30. At each training iteration, the L-BFGS optimizer is run with a maximum of 500 iterations and up to 2000 function evaluations. The L-BFGS optimization terminates if the directional derivative falls below a threshold of $10^{-5}$ or if the maximum absolute change in the loss function over five consecutive iterations is less than $10^{-5}$.

Training dataset. For each benchmark problem, MAP estimation is performed to find the target posterior mode. We launch MAP optimization runs from random initial points and collect multiple optimization traces as the training dataset for NFR and VSBQ. The total number of target density evaluations is fixed to $3000 D$. It is worth noting that the MAP estimate depends on the choice of parameterization. We align with the practical usage scenario where optimization is performed in the original parameter space and the parameter bounds are dealt with by the optimizers (in our case, CMA-ES and BADS).

# A.2. Metrics description 

Following Acerbi (2020); Li et al. (2024), we use three metrics: the absolute difference $\Delta$ LML between the true and estimated log normalizing constant (log marginal likelihood); the mean marginal total variation distance (MMTV); and the "Gaussianized" symmetrized KL divergence (GsKL) between the approximate and true posterior. For each problem,

[^0]
[^0]:    7. El Gammal et al. (2023); Li et al. (2024) set the low-density thresholds by referring to the log-density range of a standard $D$-dimensional multivariate Gaussian distribution, which requires computing an inverse CDF of a chi-squared distribution. However, this computation for determining the extremely low-density threshold can numerically overflow to $\infty$. Therefore, we use a linear approximation in $D$, similar to Huggins et al. (2023).

---

#### Page 21

ground-truth posterior samples are obtained through rejection sampling, extensive MCMC, or analytical/numerical methods. The ground-truth log normalizing constant is computed analytically, using numerical quadrature methods, or estimated from posterior samples via Geyer's reverse logistic regression (Geyer, 1994). For completeness, we describe below the metrics and desired thresholds in detail, largely following Li et al. (2024):

- $\Delta$ LML measures the absolute difference between true and estimated log marginal likelihood. We aim for an LML loss $<1$, as differences in log model evidence $\ll 1$ are considered negligible for model selection (Burnham and Anderson, 2003).
- The MMTV quantifies the (lack of) overlap between true and approximate posterior marginals, defined as

$$
\operatorname{MMTV}(p, q)=\sum_{d=1}^{D} \int_{-\infty}^{\infty} \frac{\left|p_{d}^{\mathrm{M}}\left(x_{d}\right)-q_{d}^{\mathrm{M}}\left(x_{d}\right)\right|}{2 D} d x_{d}
$$

where $p_{d}^{\mathrm{M}}$ and $q_{d}^{\mathrm{M}}$ denote the marginal densities of $p$ and $q$ along the $d$-th dimension. An MMTV metric of 0.2 indicates that, on average across dimensions, the posterior marginals have an $80 \%$ overlap. As a rule of thumb, we consider this level of overlap (MMTV $<0.2$ ) as the threshold for a reasonable posterior approximation.

- The (averaged) GsKL metric evaluates differences in means and covariances:

$$
\operatorname{GsKL}(p, q)=\frac{1}{2 D}\left[D_{\mathrm{KL}}(\mathcal{N}[p] \|\mathcal{N}[q])+D_{\mathrm{KL}}(\mathcal{N}[q] \|\mathcal{N}[p])\right]
$$

where $D_{\mathrm{KL}}(p \| q)$ is the Kullback-Leibler divergence between distributions $p$ and $q$ and $\mathcal{N}[p]$ denotes a multivariate Gaussian with the same mean and covariance as $p$ (similarly for $q$ ). This metric has a closed-form expression in terms of means and covariance matrices. For reference, two Gaussians with unit variance whose means differ by $\sqrt{2}$ (resp. $\frac{1}{2}$ ) yield GsKL values of 1 (resp. $\frac{1}{8}$ ). As a rule of thumb, we consider $\mathrm{GsKL}<\frac{1}{8}$ to indicate a sufficiently accurate posterior approximation.

# A.3. Real-world problems description 

Bayesian timing model $(D=5)$. We analyze data from a sensorimotor timing experiment in which participants were asked to reproduce time intervals $\tau$ between a mouse click and screen flash, with $\tau \sim$ Uniform[0.6, 0.975] s (Acerbi et al., 2012). The model assumes participants receive noisy sensory measurements $t_{\mathrm{s}} \sim \mathcal{N}\left(\tau, w_{\mathrm{s}}^{2} \tau^{2}\right)$ and they generate an estimate $\tau_{\star}$ by combining this sensory evidence with a Gaussian prior $\mathcal{N}\left(\tau ; \mu_{\mathrm{p}}, \sigma_{\mathrm{p}}^{2}\right)$ and taking the posterior mean. Their reproduced times then include motor noise, $t_{\mathrm{m}} \sim \mathcal{N}\left(\tau_{\star}, w_{\mathrm{m}}^{2} \tau_{\star}^{2}\right)$, and each trial has probability $\lambda$ of a "lapse" (e.g., misclick) yielding instead $t_{\mathrm{m}} \sim$ Uniform[0, 2] s. The model has five parameters $\boldsymbol{\theta}=\left(w_{\mathrm{s}}, w_{\mathrm{m}}, \mu_{\mathrm{p}}, \sigma_{\mathrm{p}}, \lambda\right)$, where $w_{\mathrm{s}}$ and $w_{\mathrm{m}}$ are Weber fractions quantifying perceptual and motor variability. We adopt a spline-trapezoidal prior for all parameters. The spline-trapezoidal prior is uniform between the plausible ranges of the parameter while falling smoothly as a cubic spline to zero toward the parameter bounds. We infer the posterior for a representative participant from Acerbi et al. (2012). As explained in the main text, we make the inference scenario more challenging and realistic by

---

#### Page 22

including log-likelihood estimation noise with $\sigma_{n}=3$. This noise magnitude is analogous to what practitioners would find by estimating the log-likelihood via Monte Carlo instead of using numerical integration methods (van Opheusden et al., 2020).

Lotka-Volterra model $(D=8)$. The model describes population dynamics through coupled differential equations:

$$
\frac{\mathrm{d} u}{\mathrm{~d} t}=\alpha u-\beta u v ; \quad \frac{\mathrm{d} v}{\mathrm{~d} t}=-\gamma v+\delta u v
$$

where $u(t)$ and $v(t)$ represent prey and predator populations at time $t$, respectively. Using data from Howard (2009), we infer eight parameters: four rate constants $(\alpha, \beta, \gamma, \delta)$, initial conditions $(u(0), v(0))$, and observation noise intensities $\left(\sigma_{u}, \sigma_{v}\right)$. The likelihood is computed by solving the equations numerically using the Runge-Kutta method. See Carpenter (2018) for further details of priors and model implementations.

Bayesian causal inference in multisensory perception $(D=12)$. In the experiment, participants seated in a moving chair judged whether the direction of their motion $s_{\text {vest }}$ matched that of a visual stimulus $s_{\text {vis }}$ ('same' or 'different'). The model assumes participants receive noisy measurements $z_{\text {vest }} \sim \mathcal{N}\left(s_{\text {vest }}, \sigma_{\text {vest }}^{2}\right)$ and $z_{\text {vis }} \sim \mathcal{N}\left(s_{\text {vis }}, \sigma_{\text {vis }}^{2}(c)\right)$, where $\sigma_{\text {vest }}$ is vestibular noise and $\sigma_{\text {vis }}(c)$ represents visual noise under three different coherence levels $c$. Each sensory noise parameter includes both a base standard deviation and a Weber fraction scaling factor. The Bayesian causal inference observer model also incorporates a Gaussian spatial prior, probability of common cause, and lapse rate for random responses, totaling 12 parameters. The model's likelihood is mildly expensive ( $\sim 3 \mathrm{~s}$ per evaluation), due to numerical integration used to compute the observer's posterior over causes, which would determine their response in each trial ('same' or 'different'). We adopt a spline-trapezoidal prior for all parameters, which remains uniform within the plausible parameter range and falls smoothly to zero near the bounds using a cubic spline. We fit the data of representative subject S11 from Acerbi et al. (2018).

# A.4. Additional experimental results 

Lumpy distribution $(D=10)$. Table A. 1 presents the results for the ten-dimensional lumpy distribution, omitted from the main text due to space constraints. All methods, except Laplace, achieve metrics below the target thresholds, with NFR performing best. While the Laplace approximation provides reasonable estimates of the normalizing constant and marginal distributions, it struggles with the full joint distribution.

Results from MAP runs with BADS optimizer. We present here the results of applying NFR and VSBQ to the MAP optimization traces from the BADS optimizer (Acerbi and Ma, 2017), instead of CMA-ES used in the main text. BADS is an efficient hybrid Bayesian optimization method that also deals with noisy observations like CMA-ES. The results for the other baselines (BBVI, Laplace) are the same as those reported in the main text, since these methods do not reuse existing (offline) optimization traces, but we repeat them here for ease of comparison.

The full results are shown in Table A.2, A.3, A.4, A.5, and A.6. From the tables, we can see that NFR still achieves the best performance for all problems. For the challenging 12D

---

#### Page 23

Table A.1: Lumpy $(D=10)$.

|  | $\Delta \mathbf{L M L}(\downarrow)$ | MMTV $(\downarrow)$ | GsKL $(\downarrow)$ |
| :-- | :--: | :--: | :--: |
| Laplace | 0.81 | 0.15 | 0.22 |
| BBVI $(1 \times)$ | $0.42[0.40,0.51]$ | $0.065[0.061,0.079]$ | $0.029[0.023,0.035]$ |
| BBVI $(10 \times)$ | $0.32[0.28,0.41]$ | $0.046[0.041,0.051]$ | $0.013[0.0095,0.015]$ |
| VSBQ | $0.11[0.097,0.15]$ | $0.033[0.031,0.038]$ | $0.0070[0.0066,0.0090]$ |
| NFR | $\mathbf{0 . 0 2 6}[0.016,0.040]$ | $\mathbf{0 . 0 2 2}[0.022,0.024]$ | $\mathbf{0 . 0 0 2 0}[0.0018,0.0023]$ |

multisensory problem, the metrics $\Delta$ LML and GsKL slightly exceed the desired thresholds. Additionally, as shown by comparing Table 4 in the main text and Table A.6, NFR performs slightly worse when using evaluations from BADS, compared to CMA-ES. We hypothesize that this is because BADS converges rapidly to the posterior mode, resulting in less evaluation coverage on the posterior log-density function, as also noted by Li et al. (2024). In sum, our results about the accuracy of NFR qualitatively hold regardless of the optimizer.

Table A.2: Multivariate Rosenbrock-Gaussian $(D=6)$. (BADS)

|  | $\Delta \mathbf{L M L}(\downarrow)$ | MMTV $(\downarrow)$ | GsKL $(\downarrow)$ |
| :-- | :--: | :--: | :--: |
| Laplace | 1.3 | 0.24 | 0.91 |
| BBVI $(1 \times)$ | $1.3[1.2,1.4]$ | $0.23[0.22,0.24]$ | $0.54[0.52,0.56]$ |
| BBVI $(10 \times)$ | $1.0[0.72,1.2]$ | $0.24[0.19,0.25]$ | $0.46[0.34,0.59]$ |
| VSBQ | $0.19[0.19,0.20]$ | $0.038[0.037,0.039]$ | $0.018[0.017,0.018]$ |
| NFR | $\mathbf{0 . 0 0 6 7}[0.0031,0.012]$ | $\mathbf{0 . 0 2 8}[0.026,0.031]$ | $\mathbf{0 . 0 0 5 3}[0.0032,0.0060]$ |

Table A.3: Lumpy. (BADS)

|  | $\Delta \mathbf{L M L}(\downarrow)$ | MMTV $(\downarrow)$ | GsKL $(\downarrow)$ |
| :-- | :--: | :--: | :--: |
| Laplace | 0.81 | 0.15 | 0.22 |
| BBVI $(1 \times)$ | $0.42[0.40,0.51]$ | $0.065[0.061,0.079]$ | $0.029[0.023,0.035]$ |
| BBVI $(10 \times)$ | $0.32[0.28,0.41]$ | $0.046[0.041,0.051]$ | $0.013[0.0095,0.015]$ |
| VSBQ | $\mathbf{0 . 0 2 9}[0.0099,0.043]$ | $0.034[0.033,0.037]$ | $0.0065[0.0060,0.0073]$ |
| NFR | $0.072[0.057,0.087]$ | $\mathbf{0 . 0 2 9}[0.028,0.031]$ | $\mathbf{0 . 0 0 2 1}[0.0017,0.0026]$ |

Runtime analysis. To assess computational efficiency, we reran each method-BBVI (with $1 \times$ budget, 10 Monte Carlo samples, learning rate 0.001 ), VSBQ, and NFR-five

---

#### Page 24

Table A.4: Bayesian timing model. (BADS)

|  | $\Delta$ LML $(\downarrow)$ | MMTV $(\downarrow)$ | GsKL $(\downarrow)$ |
| :--: | :--: | :--: | :--: |
| BBVI $(1 \times)$ | $1.6[1.1,2.5]$ | $0.29[0.27,0.34]$ | $0.77[0.67,1.0]$ |
| BBVI $(10 \times)$ | $\mathbf{0 . 3 2}[0.036,0.66]$ | $0.11[0.088,0.15]$ | $0.13[0.052,0.23]$ |
| VSBQ | $\mathbf{0 . 2 2}[0.18,0.42]$ | $\mathbf{0 . 0 5 7}[0.045,0.074]$ | $\mathbf{0 . 0 1 0}[0.0070,0.14]$ |
| NFR | $\mathbf{0 . 2 4}[0.21,0.27]$ | $\mathbf{0 . 0 6 0}[0.052,0.076]$ | $\mathbf{0 . 0 1 4}[0.0088,0.017]$ |

Table A.5: Lotka-volterra model. (BADS)

|  | $\Delta$ LML $(\downarrow)$ | MMTV $(\downarrow)$ | GsKL $(\downarrow)$ |
| :--: | :--: | :--: | :--: |
| Laplace | 0.62 | 0.11 | 0.14 |
| BBVI $(1 \times)$ | $0.47[0.42,0.59]$ | $0.055[0.048,0.063]$ | $0.029[0.025,0.034]$ |
| BBVI $(10 \times)$ | $0.24[0.23,0.36]$ | $0.029[0.025,0.039]$ | $0.0087[0.0052,0.014]$ |
| VSBQ | $1.0[1.0,1.0]$ | $0.084[0.081,0.087]$ | $0.063[0.061,0.064]$ |
| NFR | $\mathbf{0 . 1 8}[0.17,0.18]$ | $\mathbf{0 . 0 1 5}[0.014,0.016]$ | $\mathbf{0 . 0 0 0 7 4}[0.00057,0.00092]$ |

Table A.6: Multisensory. (BADS)

|  | $\Delta$ LML $(\downarrow)$ | MMTV $(\downarrow)$ | GsKL $(\downarrow)$ |
| :--: | :--: | :--: | :--: |
| VSBQ | $1.5 \mathrm{e}+3[6.2 \mathrm{e}+2,2.1 \mathrm{e}+3]$ | $0.87[0.81,0.90]$ | $1.2 \mathrm{e}+4[2.0 \mathrm{e}+2,1.4 \mathrm{e}+8]$ |
| NFR | $\mathbf{1 . 1}[0.95,1.3]$ | $\mathbf{0 . 1 5}[0.13,0.19]$ | $\mathbf{0 . 2 2}[0.15,0.94]$ |

times independently on an NVIDIA V100 GPU for each problem. Table A. 7 reports the average runtimes (in seconds) along with standard deviations.

The Laplace approximation is generally the fastest approach, except in cases involving expensive likelihood evaluations. BBVI $(1 \times)$ is fast for models with cheap likelihood evaluations, but becomes computationally demanding or infeasible for models with expensive likelihoods. The runtime of BBVI $(10 \times)$ is approximately ten times that of BBVI $(1 \times)$. NFR's runtime is significantly influenced by the number of annealing steps; we used 20 steps across all experiments. However, for several problems, NFR performs comparably well with fewer or even no annealing steps (see Appendix A.7), potentially enabling substantial speed-ups. We defer a more aggressive optimization of the NFR pipeline to future work.

# A.5. Black-box variational inference implementation 

Normalizing flow architecture specifications and initialization. For BBVI, we use the same normalizing flow architecture as in NFR. The base distribution of the normalizing

---

#### Page 25

Table A.7: Average runtime (in seconds) across five runs for each method.

| Model | Laplace | BBVI (1x) | VSBQ | NFR |
| :-- | :--: | :--: | :--: | :--: |
| Rosenbrock-Gaussian $(\mathrm{D}=6)$ | $\sim 1$ | $171 \pm 10$ | $391 \pm 33$ | $1374 \pm 27$ |
| Lumpy $(\mathrm{D}=10)$ | $\sim 1$ | $383 \pm 24$ | $949 \pm 49$ | $1499 \pm 70$ |
| Noisy timing $(\mathrm{D}=5)$ | $\mathrm{N} / \mathrm{A}$ | $1167 \pm 85$ | $301 \pm 20$ | $937 \pm 57$ |
| Lotka-Volterra $(\mathrm{D}=8)$ | $\sim 1$ | $295 \pm 6$ | $817 \pm 151$ | $1384 \pm 38$ |
| Multisensory $(\mathrm{D}=12)$ | $\sim 3 \mathrm{hr}$ | $>30 \mathrm{hr}$ | $1345 \pm 185$ | $1742 \pm 11$ |

flow is set to a learnable diagonal multivariate Gaussian, unlike in NFR where the means and variances can be estimated from the MAP optimization runs. The base distribution is initialized as a multivariate Gaussian with mean zero and standard deviations set to onetenth of the plausible ranges. The transformation layers, parameterized by neural networks, are initialized using the same procedure as in NFR (see Appendix A.1).

Stochastic optimization. As described in the main text, BBVI is performed by optimizing the ELBO using the Adam optimizer. To give BBVI the best chance of performing well, for each problem we conducted a grid search over the learning rate $\{0.01,0.001\}$ and the number of Monte Carlo samples for gradient estimation $\{1,10,100\}$, selecting the bestperforming configuration based on the estimated ELBO value and reporting the performance metrics accordingly. Following Li et al. (2024), we further apply a control variate technique to reduce the variance of the ELBO gradient estimator.

# A.6. Limitations 

In this work, we leverage normalizing flows as a regression surrogate to approximate the log-density function of a probability distribution. This methodology inherits the limitations of surrogate modeling approaches. Regardless of the source, the training dataset needs to sufficiently cover regions of non-negligible probability mass. In high-dimensional settings, this implies that the required number of training points grows exponentially, eventually becoming impractical (Li et al., 2024). In practice, similarly to other surrogate-based methods, we expect our method to be applicable to models with up to 10-15 parameters, as demonstrated by the 12-dimensional example in the main text. Scalability beyond $D \approx 20$ remains to be investigated.

In the paper, we focus on obtaining training data from MAP optimization traces. In this case, care must be taken to ensure the MAP estimate does not fall exactly on parameter bounds; otherwise, transformations into inference space (Appendix A.1) could push logdensity observations to infinity, rendering them uninformative for constructing the normalizing flow surrogate. This issue is an old and well-known problem in approximate Bayesian inference (e.g., for the Laplace approximation, MacKay, 1998) and can be mitigated by imposing priors that vanish at the bounds (Gelman et al., 2013, Chapter 13), such as the spline-trapezoidal prior as in Appendix A.3). Additionally, fitting a regression model to pointwise log-density observations may become less meaningful in certain scenarios, e.g., when the likelihood is unbounded or highly non-smooth.

---

#### Page 26

Our proposed technique, normalizing flow regression, jointly estimates both the flow parameters and the normalizing constant. The latter is a notoriously challenging quantity to infer even when target distribution samples are available (Geyer, 1994; Gutmann and Hyvärinen, 2010; Gronau et al., 2017). We impose priors over the flow for mitigating the non-identifiability issue (Section 3.3) and further apply an annealed optimization technique (Section 3.4), which we empirically find improves the posterior approximation and normalizing constant estimation (Section 4, Appendix A.7). Compared to Gaussian process surrogates, where smoothness is explicitly controlled by the covariance kernel, our priors over the flow are more implicit in governing the regularity of the (log-)density function, yet more explicit in shaping the overall distribution. Nevertheless, these strategies are not silver bullets, and we strongly recommend performing diagnostic checks on the flow approximation (Appendix A.8) whenever possible.

Finally, in this paper, our focus is on problems with relatively smooth, unimodal or mildly multimodal posteriors, which are common in real-world statistical modeling. Normalizing flows are known to struggle when the base distribution and target distribution exhibit significant topological differences (Cornish et al., 2020; Stimper et al., 2022). In the density estimation context - where the flow is trained via maximum likelihood on samples from the target distribution, a substantially easier setting than ours-there exist specialized approaches to improve performance for multimodal distributions (Stimper et al., 2022) and distributions with a mix of light and heavy tails (Amiri et al., 2024). A detailed investigation into handling such challenging posterior structures in regression settings and potential extensions is left for future research.

# A.7. Ablation studies 

To validate our key design choices, we conducted ablation studies examining three components of NFR: the likelihood function (Section 3.2), flow priors (Section 3.3), and annealed optimization (Section 3.4). We tested these using two problems from our benchmark: the Bayesian timing model $(D=5)$ and the challenging multisensory perception model ( $D=12$ ). As shown in Table A.8, our proposed combination of Tobit likelihood, flow prior settings, and annealed optimization achieves the best overall performance. The progression of results reveals several insights.

First, noise shaping in the regression likelihood proves crucial. The basic Gaussian observation noise without noise shaping, as defined in Eq. 7, yields poor approximations of the true target posterior. Adding noise shaping to the regression likelihood significantly improves performance. Switching then to our Tobit likelihood (Eq. 8) provides marginally further benefits. Indeed, the Gaussian likelihood with noise shaping is a special case of the Tobit likelihood where the low-density threshold $y_{\text {low }}$ approaches negative infinity.

Second, the importance of annealing depends on problem complexity. While the lowdimensional timing model performs adequately without annealing, the 12-dimensional multisensory model requires it for stable optimization. This suggests annealing becomes crucial as dimensionality increases.

Finally, flow priors prove essential for numerical stability and performance. Without them, many optimization runs fail due to numerical errors (marked with asterisks in Table A.8), and even successful runs show substantially degraded performance.

---

#### Page 27

Table A.8: Ablation experiments. The abbreviation 'ns' refers to noise shaping (Eq. A.1). Results marked with $*$ indicate that multiple runs failed due to numerical errors.

| Ablation settings |  |  | Bayesian timing model $(D=5)$ |  |  | Multisensory $(D=12)$ |  |  |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| likelihood | $\begin{gathered} \text { with } \\ \text { flow priors } \end{gathered}$ | annealing | $\Delta$ LML | MMTV | GsKL | $\Delta$ LML | MMTV | GsKL |
| Gaussian w/o ns | $\checkmark$ | $\checkmark$ | 0.16 | 0.21 | 0.42 | 4.0 | 0.44 | 5.9 |
| Gaussian w/ ns | $\checkmark$ | $\checkmark$ | [0.089,0.29] | [0.18,0.30] | $[0.24,0.83]$ | $[1.9,7.1]$ | $[0.40,0.51]$ | $[3.4,9.3]$ |
| Tobit | $\checkmark$ | $\checkmark$ | 0.20 | 0.055 | 0.0096 | 0.87 | 0.13 | 0.12 |
|  |  |  | $[0.18,0.23]$ | $[0.043,0.059]$ | $[0.0074,0.013]$ | $[0.69,1.0]$ | $[0.11,0.15]$ | $[0.086,0.17]$ |
| Tobit | $\checkmark$ | $\checkmark$ | 0.20 | 0.048 | 0.0098 | 24. | 0.82 | $2.8 \mathrm{e}+2$ |
|  |  |  | $[0.17,0.23]$ | $[0.044,0.052]$ | $[0.0062,0.011]$ | $[18 ., 42.]$ | $[0.76,0.84]$ | $[62 ., 9.0 \mathrm{e}+2]$ |
| Tobit | $\checkmark$ | $\checkmark$ | $6.7^{*}$ | $0.99^{*}$ | $2.6 \mathrm{e}+3^{*}$ | $0.86^{*}$ | $0.14^{*}$ | $0.25^{*}$ |
|  |  |  | $[6.0,7.9]$ | $[0.99,1.0]$ | $[1.6 \mathrm{e}+3,4.6 \mathrm{e}+3]$ | $[0.73,0.96]$ | $[0.13,0.17]$ | $[0.14,3.6]$ |
| Tobit | $\checkmark$ | $\checkmark$ | 0.18 | 0.049 | 0.0086 | 0.82 | 0.13 | 0.11 |
|  |  |  | $[0.17,0.24]$ | $[0.041,0.052]$ | $[0.0053,0.011]$ | $[0.75,0.90]$ | $[0.12,0.14]$ | $[0.091,0.16]$ |

# A.8. Diagnostics 

When approximating a posterior through regression on a set of log-density evaluations, several issues can lead to poor-quality approximations. The training points may inadequately cover the true target posterior, and while the normalizing flow can extrapolate to missing regions, its accuracy in these areas is not guaranteed. Additionally, since we treat the unknown log normalizing constant $C$ as an optimization parameter, biased estimates can cause problems: overestimation leads to a hallucination of probability mass in low-density regions, while underestimation results in overly concentrated, mode-seeking behavior.

Given these potential issues, we recommend two complementary diagnostic approaches to practitioners to assess the quality of the flow approximation in addition to standard posterior predictive checks.

1. When additional noiseless target posterior density evaluations are available, we can use the fitted flow as a proposal distribution for Pareto smoothed importance sampling (PSIS; Vehtari et al., 2024). PSIS computes a Pareto $\hat{k}$ statistic that quantifies how well the proposal (the flow) approximates the target posterior. A value of $\hat{k} \leq 0.7$ indicates a good approximation, while $\hat{k}>0.7$ suggests poor alignment with the posterior (Yao et al., 2018; Dhaka et al., 2021; Vehtari et al., 2024). ${ }^{8}$ The target log density evaluations needed for this diagnostic can be computed in parallel for efficiency.
2. A simple yet effective complementary diagnostic approach uses corner plots (ForemanMackey, 2016) to visualize flow samples with pairwise two-dimensional marginal densities, alongside log-density observation points $\mathbf{X}$. This visualization can reveal a
[^0]
[^0]:    8. Apart from being a diagnostic, importance sampling can help refine the approximate posterior when $\hat{k}<0.7$.

---

#### Page 28

Table A.9: PSIS diagnostics. Both the median and the $95 \%$ confidence interval (CI) of the median are provided. We show the PSIS- $\hat{k}$ statistic computed with 100, 1000, and 2000 proposal samples. $\hat{k}>0.7$ indicates potential issues and is reported in red. (CMA-ES)

| Problem | PSIS- $\hat{k}$ (100) | PSIS- $\hat{k}$ (1000) | PSIS- $\hat{k}$ (2000) |
| :--: | :--: | :--: | :--: |
| Multivariate Rosenbrock-Gaussian | $0.64[0.38,0.87]$ | $0.88[0.63,1.2]$ | $0.91[0.75,1.0]$ |
| Lumpy | $0.36[0.26,0.45]$ | $0.34[0.30,0.42]$ | $0.39[0.26,0.45]$ |
| Lotka-Volterra model | $0.50[0.24,0.57]$ | $0.41[0.27,0.52]$ | $0.39[0.28,0.56]$ |
| Multisensory | $0.23[0.15,0.50]$ | $0.37[0.31,0.50]$ | $0.53[0.43,0.56]$ |

common failure mode known as hallucination (De Souza et al., 2022; Li et al., 2024), where the surrogate model, the flow in our case, erroneously places significant probability mass in regions far from the training points.

We illustrate these two diagnostics in detail with examples below. For PSIS, we use the normalizing flow $q_{\boldsymbol{\phi}}$ as the proposal distribution for importance sampling and compute the importance weights,

$$
r_{s}=\frac{p_{\text {target }}\left(\mathbf{x}_{s}\right)}{q_{\boldsymbol{\phi}}\left(\mathbf{x}_{s}\right)}, \quad \mathbf{x}_{s} \sim q_{\boldsymbol{\phi}}(\mathbf{x})
$$

PSIS fits a generalized Pareto distribution using the importance ratios $r_{s}$ and returns the estimated shape parameter $\hat{k}$ which serves as a diagnostic for indicating the discrepancy between the proposal distribution and the target distribution. $\hat{k}<0.7$ indicates that the normalizing flow approximation is close to the target distribution. Values of $\hat{k}$ above the 0.7 threshold are indicative of potential issues and reported in red. As shown in Table A. 9 and A.10, PSIS- $\hat{k}$ diagnostics is below the threshold 0.7 for all problems except the multivariate Rosenbrock-Gaussian posterior. However, as we see from the metrics $\Delta$ LML, MMTV, and GsKL in Table 1 and Figure A.2(d), the normalizing flow approximation matches the ground truth target posterior well. We hypothesize that the alarm raised by PSIS- $\hat{k}$ is due to the long tail in multivariate Rosenbrock-Gaussian distribution and PSIS- $\hat{k}$ is sensitive to tail underestimation in the normalizing flow approximation.

In the case of noisy likelihood evaluations or additional likelihood evaluations not available, PSIS cannot be applied. Instead, we can use corner plots (Foreman-Mackey, 2016) to detect algorithm failures. Corner plots visualize posterior samples using pairwise twodimensional marginal density contours, along with one-dimensional histograms for marginal distributions. For diagnostics purposes, we could overlay training points $\mathbf{X}$ for NFR onto the corner plots to check whether high-probability regions are adequately supported by training data (Li et al., 2024). A common failure mode, known as hallucination (De Souza et al., 2022; Li et al., 2024), occurs when flow-generated samples lie far from the training points, indicating that the flow predictions cannot be trusted. Figure A. 1 provides an example of such a diagnostic plot. The failure case shown in Figure A.1(a) was obtained by omitting the flow priors as done in the ablation study (Appendix A.7).

---

#### Page 29

Table A.10: PSIS diagnostics. Both the median and the $95 \%$ confidence interval (CI) of the median are provided. We show the PSIS- $\hat{k}$ statistic computed with 100, 1000, and 2000 proposal samples. $\hat{k}>0.7$ indicates potential issues and is reported in red. (BADS)

| Problem | PSIS- $\hat{k}$ (100) | PSIS- $\hat{k}$ (1000) | PSIS- $\hat{k}$ (2000) |
| :--: | :--: | :--: | :--: |
| Multivariate Rosenbrock-Gaussian | $0.54[0.33,0.79]$ | $0.61[0.46,0.97]$ | $0.77[0.44,1.1]$ |
| Lumpy | $0.41[0.23,0.46]$ | $0.37[0.32,0.45]$ | $0.32[0.25,0.45]$ |
| Lotka-Volterra model | $0.41[0.22,0.58]$ | $0.35[0.27,0.38]$ | $0.35[0.26,0.45]$ |
| Multisensory | $0.52[0.20,0.58]$ | $0.34[0.27,0.41]$ | $0.36[0.26,0.47]$ |

> **Image description.** This image shows two corner plots, labeled (a) and (b), used for diagnostics. Each plot displays a matrix of scatter plots and histograms, representing the relationships between different parameters.
> 
> In both plots:
> 
> *   The diagonal elements of the matrix are histograms, displayed in orange, showing the marginal distribution of each parameter.
> *   The off-diagonal elements are scatter plots, showing the joint distribution of pairs of parameters. These plots are filled with many small blue points.
> 
> Differences between plot (a) and (b):
> 
> *   In plot (a), the blue points in the scatter plots are sparse and concentrated in specific regions, suggesting potential issues with the model.
> *   In plot (b), the blue points in the scatter plots are more densely clustered and form elliptical shapes, with orange density contours overlaid, indicating a better-behaved model. The orange contours highlight regions of high probability density.
> 
> The parameters represented in the plots are:
> 
> *   ws
> *   wm
> *   μp
> *   σp
> *   λ
> 
> The axes are labeled with these parameter names and corresponding numerical values.

Figure A.1: Diagnostics using corner plots. The orange density contours represent the flow posterior samples, while the blue points indicate training data for flow regression. (a) The flow's probability mass escapes into regions with few or no training points, highlighting an unreliable flow approximation. (b) The highprobability region of the flow is well supported by training points, indicating that the qualitative diagnostic check is passed.

---

#### Page 30

# A.9. Visualization of posteriors 

As an illustration of our results, we use corner plots (Foreman-Mackey, 2016) to visualize posterior samples with pairwise two-dimensional marginal density contours, as well as the 1D marginals histograms. In the following pages, we report example solutions obtained from a run for each problem and algorithm (Laplace, BBVI, VSBQ, NFR). ${ }^{9}$ The ground-truth posterior samples are in black and the approximate posterior samples from the algorithm are in orange (see Figure A.2, A.3, A.4, A.5, and A.6).
9. Both VSBQ and NFR use the log-density evaluations from CMA-ES, as described in the main text.

---

#### Page 31

> **Image description.** The image contains two triangular grids of plots, each representing a comparison of probability distributions for six variables (X1 through X6). The top grid is labeled "(a) Laplace" and the bottom grid is labeled "(b) BBVI (10x)". Each grid consists of a 6x6 matrix of plots, but only the lower triangle is shown.
> 
> *   **Diagonal Plots:** The plots along the diagonal of each grid are histograms, showing the marginal distribution of each variable (X1 to X6). Each histogram shows two distributions overlaid: one in black and one in orange.
> 
> *   **Off-Diagonal Plots:** The plots off the diagonal are scatter plots combined with contour plots. These show the joint distribution of pairs of variables. The scatter plots are represented by a cloud of grey points, and the contour plots are represented by black and orange lines.
> 
> *   **Axes Labels:** The x-axis of each plot is labeled with the variable name (X1, X2, X3, X4, X5, X6). The y-axis of the plots in the first column are labeled with the variable names (X2, X3, X4, X5, X6). The scales of the axes vary between plots.

---

#### Page 32

> **Image description.** This image contains two triangular grid plots, one above the other. Each grid plot visualizes the relationships between six variables, labeled X1 through X6. The top plot is labeled "(c) VSBQ" and the bottom plot is labeled "(d) NFR".
> 
> Each plot consists of a 6x6 grid of subplots. The diagonal plots show histograms of individual variables, while the off-diagonal plots show scatter plots or density contours of pairs of variables.
> 
> *   **Histograms (Diagonal):** The diagonal subplots, from top-left to bottom-right, display histograms for variables X1, X2, X3, X4, X5, and X6 respectively. These histograms are represented by orange step functions with black outlines. The X-axis of each histogram represents the variable's value, and the Y-axis represents the frequency or density.
> 
> *   **Scatter/Density Plots (Off-Diagonal):** The off-diagonal subplots display the relationship between pairs of variables. The lower triangle of the grid shows scatter plots of the data points, which appear as a dense collection of tiny orange dots. The upper triangle shows density contours, which are black lines surrounding areas of high data point concentration, with the areas inside the contours filled with orange.
> 
> *   **Axes Labels:** The X-axis of each subplot is labeled with the corresponding variable name (X1, X2, X3, X4, X5, X6). The Y-axis of each subplot is labeled with the corresponding variable name (X2, X3, X4, X5, X6), but only on the leftmost column of subplots. The axes also have numerical tick marks indicating the range of values for each variable.
> 
> *   **Plot Titles:** Below each triangular grid plot, there is a title. The title of the top plot is "(c) VSBQ", and the title of the bottom plot is "(d) NFR".

Figure A.2: Multivariate Rosenbrock-Gaussian $(D=6)$ posterior visualization. The orange density contours and points in the sub-figures represent the posterior samples from different algorithms, while the black contours and points denote ground truth samples.

---

#### Page 33

> **Image description.** The image contains two triangular grid plots, one above the other, each displaying pairwise relationships between ten variables. The plots are arranged such that the diagonal elements show the marginal distributions of each variable, and the off-diagonal elements show the joint distributions of pairs of variables.
> 
> The top plot is labeled "(a) Laplace" and the bottom plot is labeled "(b) BBVI (10x)".
> 
> Each plot consists of a 10x10 grid of subplots, but only the lower triangle is populated, resulting in a triangular arrangement.
> 
> *   **Diagonal Subplots:** These subplots display histograms, which appear as step plots, showing the marginal distribution of each variable. The histograms are colored in orange and overlaid with a black line. The variables are labeled x1 through x10 along the diagonal.
> *   **Off-Diagonal Subplots:** These subplots display 2D density plots, or contour plots, showing the joint distribution of pairs of variables. The contours are represented by thin gray lines, with an orange filled area at the center. The x and y axes of these plots are not labeled with numerical values.
> *   **Text:** The variables are labeled x1 through x10 along the x and y axes of the grid. The x-axis labels are located at the bottom of the grid, while the y-axis labels are located on the left side.

---

#### Page 34

> **Image description.** The image contains two triangular grid plots showing posterior visualizations. The top plot is labeled "(c) VSBQ" and the bottom plot is labeled "(d) NFR".
> 
> Each plot consists of a 10x10 grid of subplots arranged in a lower triangular format.
> 
> *   **Diagonal Subplots:** The subplots along the diagonal display histograms. Each histogram is plotted with an orange line. The x-axis of each histogram is labeled with a variable name, ranging from x1 to x10.
> 
> *   **Off-Diagonal Subplots:** The off-diagonal subplots display 2D density contours. Each contour plot shows the relationship between two variables. The contours are plotted in orange, and the axes are labeled with the corresponding variable names.
> 
> The overall arrangement in both plots is identical, with the histograms on the diagonal and the 2D density contours in the lower triangle. The only difference between the two plots is the data used to generate the histograms and contours.

Figure A.3: Lumpy $(D=10)$ posterior visualization. The orange density contours and points in the sub-figures represent the posterior samples from different algorithms, while the black contours and points denote ground truth samples.

---

#### Page 35

> **Image description.** This image contains two triangular grid plots, one above the other. Each plot displays a correlation matrix of four parameters. The top plot is labeled "(a) BBVI (10x)", and the bottom plot is labeled "(b) VSBQ".
> 
> Each plot consists of a 4x4 grid of subplots. The diagonal subplots display histograms of the individual parameters. The subplots below the diagonal display 2D density plots showing the correlation between pairs of parameters. The upper triangle of the grid is empty.
> 
> The parameters are:
> *   ws (x-axis of the bottom row)
> *   wm (x-axis of the second row from the bottom, and y-axis of the top row)
> *   μp (x-axis of the third row from the bottom, and y-axis of the second row from the top)
> *   σp (x-axis of the fourth row from the bottom, and y-axis of the third row from the top)
> *   λ (y-axis of the bottom row)
> 
> The histograms on the diagonal show the distribution of each parameter. They are displayed as step plots, with both a black and an orange line indicating the distribution.
> 
> The 2D density plots show the correlation between each pair of parameters. The density is indicated by the color, ranging from light orange to dark orange, and by the presence of contour lines. The axes of these plots are labeled with the corresponding parameter names and values.

---

#### Page 36

> **Image description.** This image is a visualization of posterior samples from a Bayesian timing model, arranged as a matrix of plots. The diagonal plots are histograms, and the off-diagonal plots are scatter plots with density contours.
> 
> The matrix is 5x5, with each row and column representing a different parameter. The parameters, read from left to right and top to bottom, are: w_s, w_m, μ_p, σ_p, and λ.
> 
> *   **Diagonal Plots (Histograms):** Each diagonal plot shows a histogram of the posterior samples for a single parameter. There are two overlapping histograms in each plot, one in orange and one in black. The orange histogram represents the posterior samples from a specific algorithm, while the black histogram represents the ground truth samples.
> 
> *   **Off-Diagonal Plots (Scatter Plots with Density Contours):** Each off-diagonal plot shows a scatter plot of the posterior samples for two parameters. The x-axis represents the parameter corresponding to the column, and the y-axis represents the parameter corresponding to the row. The scatter plots are overlaid with density contours, also in orange and black, representing the posterior samples from the algorithm and the ground truth samples, respectively. The density of points is represented by a faint orange color, with darker orange and black lines showing the contour lines.
> 
> *   **Axis Labels:** Each plot has axis labels indicating the parameter being plotted. The x-axis labels are located below the bottom row of plots, and the y-axis labels are located to the left of the leftmost column of plots. The labels are: w_s, w_m, μ_p, σ_p, and λ. Numerical values are shown on each axis.
> 
> *   **Title:** Below the entire matrix of plots, there is the text "(c) NFR".

Figure A.4: Bayesian timing model $(D=5)$ posterior visualization. The orange density contours and points in the sub-figures represent the posterior samples from different algorithms, while the black contours and points denote ground truth samples.

---

#### Page 37

> **Image description.** This image contains two triangular grid plots, each displaying the posterior distributions and pairwise relationships of several parameters. The top plot is labeled "(a) Laplace" and the bottom plot is labeled "(b) BBVI (10x)".
> 
> Each plot consists of a grid of subplots arranged in a lower triangular fashion. The diagonal subplots display histograms of the marginal posterior distributions for each parameter. The off-diagonal subplots show contour plots representing the joint posterior distributions for pairs of parameters.
> 
> The parameters are labeled along the axes of the grid as follows: α, β, γ, δ, u(0), v(0), σu, and σv. The histograms on the diagonal show the distribution of each parameter, with both black and orange lines outlining the bars. The contour plots in the off-diagonal subplots show the relationships between pairs of parameters, with orange contours indicating higher density regions.

---

#### Page 38

> **Image description.** This image shows two triangular grid plots, one above the other, visualizing posterior distributions. Each grid plot consists of a series of subplots arranged in a triangular matrix.
> 
> Each subplot represents either a marginal distribution (along the diagonal) or a joint distribution (off-diagonal). The diagonal subplots contain histograms, while the off-diagonal subplots contain scatter plots with density contours overlaid. The histograms are step plots, with both black and orange lines. The scatter plots show a cloud of points, with orange density contours overlaid.
> 
> The axes of the plots are labeled with parameters: alpha, beta, gamma, delta, u(0), v(0), sigma_u, and sigma_v. The y-axis labels are on the left side of the plots and the x-axis labels are on the bottom.
> 
> The top grid plot is labeled "(c) VSBQ" at the bottom center. The bottom grid plot is labeled "(d) NFR" at the bottom center.

Figure A.5: Lotka-Volterra mode $(D=8)$ posterior visualization. The orange density contours and points in the sub-figures represent the posterior samples from different algorithms, while the black contours and points denote ground truth samples.

---

#### Page 39

> **Image description.** This image is a correlation plot, also known as a pair plot or scatterplot matrix, displaying the relationships between multiple variables. It's arranged in a triangular matrix format.
> 
> *   **Overall Structure:** The plot consists of a grid of subplots. The diagonal subplots display histograms, while the off-diagonal subplots show scatter plots. The lower triangle of the matrix is filled with these plots, while the upper triangle is empty.
> 
> *   **Diagonal Subplots (Histograms):** Each diagonal subplot represents a single variable (x1 to x12). These subplots show the distribution of each variable using a histogram. Each histogram has two lines, one black and one orange.
> 
> *   **Off-Diagonal Subplots (Scatter Plots):** Each off-diagonal subplot displays the relationship between two variables using a scatter plot. The x-axis represents one variable, and the y-axis represents another. The density of points indicates the strength of the correlation between the variables. The scatter plots are displayed as grey dots with black contour lines indicating density. Some scatter plots also have an orange area, indicating a higher density of points in that region.
> 
> *   **Axes Labels:** The x-axis of the bottom row of subplots is labeled with variable names: x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, and x12. The y-axis of the leftmost column of subplots is labeled with variable names: x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, and x12. Each axis also has numerical values to indicate the range of the variable.
> 
> *   **Text:** At the bottom of the plot, centered, is the text "(a) VSBQ".

---

#### Page 40

> **Image description.** This image is a triangular matrix of plots, visualizing a multisensory posterior distribution. Each plot represents the joint distribution of two variables, with histograms on the diagonal and scatter plots with density contours off-diagonal.
> 
> *   **Arrangement:** The plots are arranged in a lower triangular matrix. The x-axis variable increases from left to right, and the y-axis variable increases from top to bottom.
> 
> *   **Diagonal Plots:** The diagonal plots are histograms, showing the marginal distribution of each variable. Each histogram has two overlaid distributions: one in orange and one in black.
> 
> *   **Off-Diagonal Plots:** The off-diagonal plots are scatter plots with density contours. The scatter plots consist of many small orange points. Black contour lines are overlaid on the scatter plots, indicating the density of the points.
> 
> *   **Labels:** The axes are labeled with variables x1 through x12. The x-axis labels are at the bottom of the matrix, and the y-axis labels are on the left side of the matrix. Numerical values are displayed on the axes.
> 
> *   **Text:** The text "(b) NFR" is present at the bottom center of the image.
> 
> *   **Colors:** The image primarily uses black and orange. The density contours and one set of points are orange, while the ground truth samples are black.
> 
> The figure visualizes the posterior samples from different algorithms (orange) and compares them to ground truth samples (black) in a 12-dimensional space. The histograms on the diagonal show the marginal distributions, while the scatter plots with density contours show the joint distributions.

Figure A.6: Multisensory $(D=12)$ posterior visualization. The orange density contours and points in the sub-figures represent the posterior samples from different algorithms, while the black contours and points denote ground truth samples.