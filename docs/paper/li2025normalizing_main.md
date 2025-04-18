```
@article{2025,
title={Normalizing Flow Regression for Bayesian Inference with Offline Likelihood Evaluations},
author={Li, Chengkun and Huggins, Bobby and Mikkola, Petrus and Acerbi, Luigi},
journal={7th Symposium on Advances in Approximate Bayesian Inference (AABI) - Proceedings track},
year={2025}
}
```

---

#### Page 1

# Normalizing Flow Regression for Bayesian Inference with Offline Likelihood Evaluations

Chengkun Li ${ }^{1}$<br>Bobby Huggins ${ }^{2}$<br>Petrus Mikkola ${ }^{1}$<br>Luigi Acerbi ${ }^{1}$<br>${ }^{1}$ Department of Computer Science, University of Helsinki<br>${ }^{2}$ Department of Computer Science and Engineering, Washington University in St. Louis

#### Abstract

Bayesian inference with computationally expensive likelihood evaluations remains a significant challenge in many scientific domains. We propose normalizing flow regression (NFR), a novel offline inference method for approximating posterior distributions. Unlike traditional surrogate approaches that require additional sampling or inference steps, NFR directly yields a tractable posterior approximation through regression on existing log-density evaluations. We introduce training techniques specifically for flow regression, such as tailored priors and likelihood functions, to achieve robust posterior and model evidence estimation. We demonstrate NFR's effectiveness on synthetic benchmarks and real-world applications from neuroscience and biology, showing superior or comparable performance to existing methods. NFR represents a promising approach for Bayesian inference when standard methods are computationally prohibitive or existing model evaluations can be recycled.

## 1. Introduction

Black-box models of varying complexity are widely used in scientific and engineering disciplines for tasks such as parameter estimation, hypothesis testing, and predictive modeling (Sacks et al., 1989; Kennedy and O’Hagan, 2001). Bayesian inference provides a principled framework for quantifying uncertainty in both parameters and models by computing full posterior distributions and model evidence (Gelman et al., 2013). However, Bayesian inference is often analytically intractable, requiring the use of approximate methods like Markov chain Monte Carlo (MCMC; Brooks, 2011) or variational inference (VI; Blei et al., 2017). These methods typically necessitate repeated evaluations of the target density, and many require differentiability of the model (Neal, 2011; Kucukelbir et al., 2017). When model evaluations are computationally expensive - for instance, involving extensive numerical methods - these requirements make standard Bayesian approaches impractical.

Due to these computational demands, practitioners often resort to simpler alternatives such as maximum a posteriori (MAP) estimation or maximum likelihood estimation (MLE); ${ }^{1}$ see for example Wilson and Collins (2019); Ma et al. (2023). While these point estimates can provide useful insights, they fail to capture parameter uncertainty, potentially leading to overconfident or biased conclusions (Gelman et al., 2013). This limitation highlights the need for efficient posterior approximation methods that avoid the computational costs of standard inference techniques.

[^0]
[^0]: 1. In practice, MLE corresponds to MAP with flat priors.

---

#### Page 2

Recent advances in surrogate modeling present promising alternatives for addressing these challenges. Costly likelihood or posterior density functions are efficiently approximated via surrogates such as Gaussian processes (GPs; Rasmussen, 2003; Gunter et al., 2014; Acerbi, 2018, 2019; Järvenpää et al., 2021; Adachi et al., 2022; El Gammal et al., 2023). To mitigate the cost of standard GPs, both sparse GPs and deep neural networks have also served as surrogates for posterior approximation (Wang et al., 2022; Li et al., 2024). However, these approaches share a key limitation: the obtained surrogate model, usually of the log likelihood or log posterior, does not directly provide a valid probability distribution. Additional steps, such as performing MCMC or variational inference on the surrogate, are needed to yield tractable posterior approximations. Furthermore, many of these methods require active collections of new likelihood evaluations, which might be unfeasible or wasteful of existing evaluations.

To address these challenges, we propose using normalizing flows as regression models for directly approximating the posterior distribution from offline likelihood or density evaluations. While normalizing flows have been extensively studied for variational inference (Rezende and Mohamed, 2015; Agrawal et al., 2020), density estimation (Dinh et al., 2017), and simulation-based inference (Lueckmann et al., 2021; Radev et al., 2022), their application as regression models for posterior approximation remains largely unexplored. Unlike other surrogate methods, normalizing flows directly yield a tractable posterior distribution which is easy to evaluate and sample from. Moreover, unlike other applications of normalizing flows, our regression approach is offline, recycling existing log-density evaluations (e.g., from MAP optimizations as in Li et al., 2024) rather than requiring costly new evaluations from the target model.

The main contribution of this work consists of proposing normalizing flows as a regression model for surrogate-based, offline Bayesian inference, together with techniques for training them in this context, such as sensible priors over flows. We demonstrate the effectiveness of our method on challenging synthetic and real-world problems, showing that normalizing flows can accurately estimate posterior distributions and their normalizing constants through regression. This work contributes a new approach for Bayesian inference in settings where standard methods are computationally prohibitive, affording more robust and uncertainty-aware modeling across scientific and engineering applications.

# 2. Background

### 2.1. Normalizing flows

Normalizing flows construct flexible probability distributions by iteratively transforming a simple base distribution, typically a multivariate Gaussian distribution. A normalizing flow defines an invertible transformation $T_{\boldsymbol{\phi}}: \mathbb{R}^{D} \rightarrow \mathbb{R}^{D}$ with parameters $\boldsymbol{\phi}$. Let $\mathbf{u} \in \mathbb{R}^{D}$ be a random variable from the base distribution $p_{\mathbf{u}}$. For a random variable $\mathbf{x}=T_{\boldsymbol{\phi}}(\mathbf{u})$, the change of variables formula gives its density as:

$$
q_{\boldsymbol{\phi}}(\mathbf{x})=p_{\mathbf{u}}(\mathbf{u})\left|\operatorname{det} J_{T_{\boldsymbol{\phi}}}(\mathbf{u})\right|^{-1}, \quad \mathbf{u}=T_{\boldsymbol{\phi}}^{-1}(\mathbf{x})
$$

where $J_{T_{\boldsymbol{\phi}}}$ denotes the Jacobian matrix of the transformation. The transformation $T_{\boldsymbol{\phi}}(\mathbf{u})$ can be designed to balance expressive power with efficient computation of its Jacobian determinant. In this paper, we use the popular masked autoregressive flow (MAF; Papamakarios

---

#### Page 3

et al., 2017). MAF constructs the transformation through an autoregressive process, where each component $\mathbf{x}^{(i)}$ depends on previous components through:

$$
\mathbf{x}^{(i)}=g_{\text {scale }}\left(\alpha^{(i)}\right) \cdot \mathbf{u}^{(i)}+g_{\text {shift }}\left(\mu^{(i)}\right)
$$

Here, $g_{\text {scale }}$ is typically chosen as the exponential function to ensure positive scaling, while $g_{\text {shift }}$ is usually the identity function. The parameters $\alpha^{(i)}$ and $\mu^{(i)}$ are outputs of unconstrained scalar functions $h_{\alpha}$ and $h_{\mu}$ that take the preceding components as inputs:

$$
\alpha^{(i)}=h_{\alpha}\left(\mathbf{x}^{(1: i-1)}\right), \quad \mu^{(i)}=h_{\mu}\left(\mathbf{x}^{(1: i-1)}\right)
$$

where $h_{\alpha}$ and $h_{\mu}$ are usually parametrized by neural networks with parameters $\boldsymbol{\phi}$.
This autoregressive structure ensures invertibility of the transformation and enables efficient computation of the Jacobian determinant needed for the density calculation in Eq. 1 (Papamakarios et al., 2021). To accelerate computation, MAF is implemented in parallel via masking, using a neural network architecture called Masked AutoEncoder for Distribution Estimation (MADE; Germain et al. 2015).

# 2.2. Bayesian inference

Bayesian inference provides a principled framework for inferring unknown parameters $\mathbf{x}$ given observed data $\mathcal{D}$. From Bayes' theorem, the posterior distribution $p(\mathbf{x} \mid \mathcal{D})$ is:

$$
p(\mathbf{x} \mid \mathcal{D})=\frac{p(\mathcal{D} \mid \mathbf{x}) p(\mathbf{x})}{p(\mathcal{D})}
$$

where $p(\mathcal{D} \mid \mathbf{x})$ is the likelihood, $p(\mathbf{x})$ is the prior over the parameters, and $p(\mathcal{D})$ is the normalizing constant, also known as evidence or marginal likelihood, a quantity useful in Bayesian model selection (MacKay, 2003). Two widely used approaches for approximating this posterior are variational inference and Markov chain Monte Carlo (Gelman et al., 2013).

VI turns posterior approximation into an optimization problem by positing a family of parametrized distributions, such as normalizing flows ( $q_{\phi}$ in Section 2.1), and optimizing over the parameters $\boldsymbol{\phi}$. The objective to maximize is commonly the evidence lower bound (ELBO), which is equivalent to minimizing the Kullback-Leibler (KL) divergence between the approximate distribution and $p(\mathbf{x} \mid \mathcal{D})$ (Blei et al., 2017). When the likelihood $p(\mathcal{D} \mid \mathbf{x})$ is a black box, the estimated ELBO gradients can exhibit high variance, thus requiring many evaluations to converge (Ranganath et al., 2014). MCMC methods, such as MetropolisHastings, aim to draw samples from the posterior by constructing a Markov chain that converges to $p(\mathbf{x} \mid \mathcal{D})$. While MCMC offers asymptotic guarantees, it requires many likelihood evaluations. Due to the typically large number of required evaluations, both VI and MCMC are often infeasible for black-box models with expensive likelihoods.

## 3. Normalizing Flow Regression

We now present our proposed method, Normalizing Flow Regression (NFR) for approximate Bayesian posterior inference. In the following, we denote with $\mathbf{X}=\left(\mathbf{x}_{1}, \ldots, \mathbf{x}_{N}\right)$ a set of input locations where we have evaluated the target posterior, with corresponding unnormalized log-density evaluations $\mathbf{y}=\left(y_{1}, \ldots, y_{N}\right)$, where $\mathbf{x}_{n} \in \mathbb{R}^{D}$ and $y_{n} \in \mathbb{R}$. Evaluations

---

#### Page 4

have associated observation noise $\boldsymbol{\sigma}^{2}=\left(\sigma_{1}^{2}, \ldots, \sigma_{N}^{2}\right),{ }^{2}$ where we set $\sigma_{n}^{2}=\sigma_{\min }^{2}=10^{-3}$ for noiseless cases. We collect these into a training dataset $\boldsymbol{\Xi}=\left(\mathbf{X}, \mathbf{y}, \boldsymbol{\sigma}^{2}\right)$ for our flow regression model. Throughout this section, we use $p_{\text {target }}(\mathbf{x}) \equiv p(\mathcal{D} \mid \mathbf{x}) p(\mathbf{x})$ to denote the unnormalized target posterior density.

# 3.1. Overview of the regression model

We use a normalizing flow $T_{\boldsymbol{\phi}}$ with normalized density $q_{\boldsymbol{\phi}}(\mathbf{x})$ to fit $N$ observations of the log density of an unnormalized target $p_{\text {target }}(\mathbf{x})$, using the dataset $\boldsymbol{\Xi}=\left(\mathbf{X}, \mathbf{y}, \boldsymbol{\sigma}^{2}\right)$. Let $f_{\boldsymbol{\phi}}(\mathbf{x})=\log q_{\boldsymbol{\phi}}(\mathbf{x})$ be the flow's log-density at $\mathbf{x}$. The log-density prediction of our regression model is:

$$
f_{\boldsymbol{\psi}}(\mathbf{x})=f_{\boldsymbol{\phi}}(\mathbf{x})+C
$$

where $C$ is an additional free parameter accounting for the unknown (log) normalizing constant of the target posterior. The parameter set of the regression model is $\boldsymbol{\psi}=(\boldsymbol{\phi}, C)$.

We train the flow regression model itself via MAP estimation, by maximizing:

$$
\begin{aligned}
\mathcal{L}(\boldsymbol{\psi}) & =\log p\left(\mathbf{y} \mid \mathbf{X}, \boldsymbol{\sigma}^{2}, f_{\boldsymbol{\phi}}, C\right)+\log p(\boldsymbol{\phi})+\log p(C) \\
& =\sum_{n=1}^{N} \log p\left(y_{n} \mid f_{\boldsymbol{\psi}}\left(\mathbf{x}_{n}\right), \sigma_{n}^{2}\right)+\log p(\boldsymbol{\phi})+\log p(C)
\end{aligned}
$$

where $p\left(y_{n} \mid f_{\boldsymbol{\psi}}\left(\mathbf{x}_{n}\right), \sigma_{n}^{2}\right)$ is the likelihood of observing log-density value $y_{n},{ }^{3}$ while $p(\boldsymbol{\phi})$ and $p(C)$ are priors over the flow parameters and log normalizing constant, respectively.

Since we only have access to finite pointwise evaluations of the target log-density, $\boldsymbol{\Xi}=$ $\left(\mathbf{X}, \mathbf{y}, \boldsymbol{\sigma}^{2}\right)$, the choice of the likelihood function and priors for the regression model is crucial for accurate posterior approximation. We detail these choices in Sections 3.2 and 3.3.

### 3.2. Likelihood function for log-density observations

For each observation $y_{n}$, let $f_{n} \equiv p_{\text {target }}\left(\mathbf{x}_{n}\right)$ denote the true unnormalized log-density value, which our flow regression model aims to estimate via its prediction $f_{\boldsymbol{\psi}}\left(\mathbf{x}_{n}\right)$. We now discuss how to choose an appropriate likelihood function $p\left(y_{n} \mid f_{\boldsymbol{\psi}}\left(\mathbf{x}_{n}\right), \sigma_{n}^{2}\right)$ for these log-density observations. A natural first choice would be a Gaussian likelihood,

$$
p\left(y_{n} \mid f_{\boldsymbol{\psi}}\left(\mathbf{x}_{n}\right), \sigma_{n}^{2}\right)=\mathcal{N}\left(y_{n} \mid f_{\boldsymbol{\psi}}\left(\mathbf{x}_{n}\right), \sigma_{n}^{2}\right)
$$

However, this choice has a significant drawback emerging from the fact that maximizing this likelihood corresponds to minimizing the point-wise squared error $\left|y_{n}-f_{\boldsymbol{\psi}}\left(\mathbf{x}_{n}\right)\right|^{2} / \sigma_{n}^{2}$. Since log-density values approach negative infinity as density values approach zero, small errors in near-zero density regions of the target posterior would dominate the regression objective in Eq. 6. This would cause the normalizing flow to overemphasize matching these

[^0]
[^0]: 2. Log-density observations can be noisy when likelihood calculation involves simulation or Monte Carlo methods. Noise for each observation can then be quantified independently via bootstrap or using specific estimators (van Opheusden et al., 2020; Acerbi, 2020; Järvenpää et al., 2021). 3. Assuming conditionally independent noise on the log-density estimates, which holds trivially for noiseless observations and for many estimation methods (van Opheusden et al., 2020; Järvenpää et al., 2021).

---

#### Page 5

> **Image description.** The image consists of two plots side-by-side, illustrating the censoring effect on a target density. Both plots have a similar x-axis, labeled "x", ranging from -16 to 16 with tick marks at -16, -8, 0, 8, and 16. The left plot shows "Density" on the y-axis, ranging from 0.0 to 0.4, with tick marks at 0.0, 0.2, and 0.4. A curve peaks sharply around x=0 and approaches 0 at the extremes. Two shaded regions with a criss-cross pattern are present near x=-16 and x=16, where the density is close to zero. The right plot displays "Log density" on the y-axis, ranging from -100 to 0, with tick marks at -100, -50, and 0. The curve is bell-shaped, peaking at x=0. A dashed horizontal line is drawn at approximately y=-50, labeled "y_low". Shaded regions with a criss-cross pattern are present where the curve falls below the "y_low" line, near x=-16 and x=16.

Figure 1: Illustration of the censoring effect of the Tobit likelihood on a target density. The left panel shows the density plot, while the right panel displays the corresponding log-density values. The shaded region represents the censored observations with log-density values below $y_{\text {low }}$, where the density is near-zero.

near-zero density observations at the expense of accurately modeling the more important high-density regions.

To address this issue, we propose a more robust Tobit likelihood for flow regression, inspired by the Tobit model (Amemiya, 1984) and the noise shaping technique (Li et al., 2024). Let $f_{\max } \equiv \max _{\mathbf{x}} \log p(\mathbf{x})$ denote the maximum log-density value (i.e., at the distribution mode). The Tobit likelihood takes the form:

$$
p\left(y_{n} \mid f_{\boldsymbol{\psi}}\left(\mathbf{x}_{n}\right), \sigma_{n}^{2}\right)= \begin{cases}\mathcal{N}\left(y_{n} ; f_{\boldsymbol{\psi}}\left(\mathbf{x}_{n}\right), \sigma_{n}^{2}+s\left(f_{\max }-f_{n}\right)^{2}\right) & \text { if } y_{n}>y_{\text {low }} \\ \Phi\left(\frac{y_{\text {low }}-f_{\boldsymbol{\psi}}\left(\mathbf{x}_{n}\right)}{\sqrt{\sigma_{n}^{2}+s\left(f_{\max }-f_{n}\right)^{2}}}\right) & \text { if } y_{n} \leq y_{\text {low }}\end{cases}
$$

where $y_{\text {low }}$ represents a threshold below which we censor observed log-density values, $\Phi$ is the standard normal cumulative distribution function (CDF), and $s(\cdot)$ a noise shaping function, discussed below. When $y_{n} \leq y_{\text {low }}$, the Tobit likelihood only requires the model's prediction $f_{\boldsymbol{\psi}}\left(\mathbf{x}_{n}\right)$ to fall below $y_{\text {low }}$, rather than match $y_{n}$ exactly (see Figure 1). The function $s(\cdot)$ acts as a noise shaping mechanism (Li et al., 2024) that linearly increases observation uncertainty for lower-density regions, further retaining information from lowdensity observations without overfitting to them (see Appendix A. 1 for details).

# 3.3. Prior settings

The flow regression model's log-density prediction depends on both the flow parameters $\boldsymbol{\phi}$ and the log normalizing constant $C$ (Eq. 5), leading to a non-identifiability issue. Given a sufficiently expressive flow, alternative parameterizations $\left(\boldsymbol{\phi}^{\prime}, C^{\prime}\right)$ can yield identical predictions at observed points. While this suggests the necessity of informative priors for both the flow and the normalizing constant, setting a meaningful prior on $C$ is challenging since the target density evaluations are neither i.i.d. nor samples from the target distribution. Therefore, we focus on imposing sensible priors on the flow parameters $\boldsymbol{\phi}$, which indirectly regularize the normalization constant and avoid the pitfalls of complete non-identifiability.

A normalizing flow consists of a base distribution and transformation layers. The base distribution can incorporate prior knowledge about the target posterior's shape, for instance

---

#### Page 6

from a moment-matching approximation. In our case, the training data $\boldsymbol{\Xi}=\left(\mathbf{X}, \mathbf{y}, \boldsymbol{\sigma}^{2}\right)$ comes from MAP optimization runs on the target posterior. We use a multivariate Gaussian with diagonal covariance as the base distribution $p_{0}$, and estimate its mean and variance along each dimension using the sample mean and variance of observations with sufficiently high log-density values (see Appendix A. 1 for further details).

Specifying priors for the flow transformation layers is less straightforward since they are parameterized by neural networks (Fortuin, 2022). As a normalizing flow is itself a distribution, setting priors for its transformation layers means defining a distribution over distributions. Our approach is to ensure that the flow stays close to its base distribution a priori, unless the data strongly suggests otherwise. We achieve this by constraining the scaling and shifting transformations using the bounded $\tanh$ function:

$$
\begin{aligned}
& g_{\text {scale }}\left(\alpha^{(i)}\right)=\alpha_{\max }^{\tanh \left(\alpha^{(i)}\right)} \\
& g_{\text {shift }}\left(\mu^{(i)}\right)=\mu_{\max } \cdot \tanh \left(\mu^{(i)}\right)
\end{aligned}
$$

where $\alpha_{\max }$ and $\mu_{\max }$ cap the maximum scaling and shifting transformation, preventing extreme deviations from the base distribution. When the flow parameters $\phi=\mathbf{0}$, both $\alpha^{(i)}$ and $\mu^{(i)}$ are zero (Eq. 3), making $g_{\text {scale }}\left(\alpha^{(i)}\right)=1$ and $g_{\text {shift }}\left(\mu^{(i)}\right)=0$, thus yielding the identity transformation. We then place a Gaussian prior on the flow parameters, $p(\boldsymbol{\phi})=\mathcal{N}\left(\boldsymbol{\phi} ; \mathbf{0}, \sigma_{\boldsymbol{\phi}}^{2} \mathbf{I}\right)$, with $\sigma_{\boldsymbol{\phi}}$ chosen through prior predictive checks (see Section 4.1). $p(\boldsymbol{\phi})$, combined with our base distribution being moment-matched to the top observations, serves as a meaningful empirical prior that centers the flow in high-density regions of the target. Finally, we place an (improper) flat prior on the log normalization constant, $p(C)=1$.

# 3.4. Annealed optimization

Fitting a flow to a complex unnormalized target density $p_{\text {target }}(\mathbf{x})$ via direct regression on observations $\boldsymbol{\Xi}=\left(\mathbf{X}, \mathbf{y}, \boldsymbol{\sigma}^{2}\right)$ can be challenging due to both the unknown log normalizing constant and potential gradient instabilities during optimization. We found that a more robust approach is to gradually fit the flow to an annealed (tempered) target across training iterations $t=0, \ldots, T_{\max }$, using an inverse temperature parameter $\beta_{t} \in[0,1]$. The tempered target takes the following form (see Figure 2 for an illustration):

$$
\widehat{f}_{\beta_{t}}(\mathbf{x})=\left(1-\beta_{t}\right) \log p_{0}(\mathbf{x})+\beta_{t} \log p_{\text {target }}(\mathbf{x})
$$

where $p_{0}(\mathbf{x})$ is the flow's base distribution. This formulation has two key advantages: first, since the base distribution is normalized, we know the true log normalizing constant $C$ is zero when $\beta_{t}=0$. Second, by initializing the flow parameters near zero, the flow starts close to its base distribution $p_{0}$, providing a stable initialization point.

The tempered observations are defined as:

$$
\begin{aligned}
& \widetilde{\mathbf{X}}_{\beta_{t}}=\mathbf{X} \\
& \widetilde{\mathbf{y}}_{\beta_{t}}=\left(1-\beta_{t}\right) \log p_{0}(\mathbf{X})+\beta_{t} \mathbf{y} \\
& \widetilde{\boldsymbol{\sigma}}_{\beta_{t}}^{2}=\max \left\{\beta_{t}^{2} \boldsymbol{\sigma}^{2}, \sigma_{\min }^{2}\right\}
\end{aligned}
$$

where $p_{0}(\mathbf{X})=\left(p_{0}\left(\mathbf{x}_{1}\right), \ldots, p_{0}\left(\mathbf{x}_{N}\right)\right)$ denotes the base distribution evaluated at all observed points. We increase the inverse temperature $\beta_{t}$ according to a tempering schedule increasing

---

#### Page 7

> **Image description.** The image shows three separate plots arranged horizontally, each representing a stage in an annealed optimization strategy. The plots share a similar structure, depicting probability density functions. The plots are contained within a box with a hand-drawn style.
>
> - **Overall Structure:** Each plot has an x-axis labeled "x" and a y-axis labeled "Density." The y-axis ranges from 0.0 to 0.4. Each plot contains three curves representing different distributions: a "Base distribution" (orange), a "Tempered distribution" (green in the middle plot, blue in the right plot), and a "Target distribution" (dashed light orange). Additionally, each plot includes several gray dots labeled "Observations."
>
> - **Plot 1 (β=0):** The title above the plot is "β=0". The orange "Base distribution" curve is prominent, showing a bell-shaped curve centered around x=0. The light orange dashed "Target distribution" is flatter. The gray "Observations" dots are positioned along the "Base distribution" curve.
>
> - **Plot 2 (β=0.5):** The title above the plot is "β=0.5". The green "Tempered distribution" curve is now more pronounced and closer to the shape of the "Target distribution". The "Observations" dots are positioned along the "Tempered distribution" curve.
>
> - **Plot 3 (β=1):** The title above the plot is "β=1". The blue "Target distribution" curve is now the most prominent and closely matches the shape of the "Target distribution" (dashed light orange). The "Observations" dots are positioned along the "Target distribution" curve.
>
> - **Legend:** A legend is located below the plots, associating the colors with the distribution types and observations: orange for "Base distribution," green for "Tempered distribution," blue for "Target distribution," and gray dots for "Observations."

Figure 2: Annealed optimization strategy. The flow regression model is progressively fitted to a series of tempered observations, with the inverse temperature $\beta$ increasing over multiple training iterations, interpolating between the base and unnormalized target distributions.

from $\beta_{0}=0$ to $\beta_{t_{\text {end }}}=1$, where $t_{\text {end }} \leq T_{\text {max }}$ marks the end of tempering. After reaching $\beta=1$, we can perform additional optimization iterations if needed. By default, we use a linear tempering schedule: $\beta_{t}=\beta_{0}+\frac{t}{t_{\text {end }}}\left(1-\beta_{0}\right)$.

# 3.5. Normalizing flow regression algorithm

Having introduced the flow regression model and tempering approach, we now present the complete method in Algorithm 1, which returns the flow parameters $\boldsymbol{\phi}$ and the log normalizing constant $C$. We follow a two-step approach: first, we fix the flow parameters $\boldsymbol{\phi}$ and optimize the scalar parameter $C$ using, e.g., Brent's method (Brent, 1973), which is efficient as it requires only a single evaluation of the flow. Then, using this result as initialization, we jointly optimize both $\boldsymbol{\phi}$ and $C$ with L-BFGS (Liu and Nocedal, 1989). Further details, including optimization termination criteria, are provided in Appendix A.1.

```
Algorithm 1: Normalizing Flow Regression
Input: Observations \(\left(\mathbf{X}, \mathbf{y}, \boldsymbol{\sigma}^{2}\right)\), total number of tempered steps \(t_{\text {end }}\), maximum
    number of optimization iterations \(T_{\max }
```

Output: Flow $T_{\phi}$ approximating the target, log normalizing constant $C$
Compute and set the base distribution for the flow, using $\left(\mathbf{X}, \mathbf{y}, \boldsymbol{\sigma}^{2}\right)$ (Section 3.3);
for $t \leftarrow 0$ to $T_{\max }$ do
Set inverse temperature $\beta_{t} \in[0,1]$ according to tempering schedule $\left(\beta_{0}=0\right)$;
Update tempered observations $\left(\widetilde{\mathbf{X}}_{\beta_{t}}, \widetilde{\mathbf{y}}_{\beta_{t}}, \widetilde{\boldsymbol{\sigma}}_{\beta_{t}}^{2}\right)$ according to Eq. 11 ;
Fix $\boldsymbol{\phi}$ and optimize $C$ using fast $1 D$ optimization with objective in Eq. 6 ;
Optimize $(\boldsymbol{\phi}, C)$ jointly using L-BFGS with objective in Eq. 6 ;
end

---

#### Page 8

# 4. Experiments

We evaluate our normalizing flow regression (NFR) method through a series of experiments. First, we conduct prior predictive checks to select our flow's prior settings (see Section 3.3). We then assess NFR's performance on both synthetic and real-world problems. For all the experiments, we use a masked autoregressive flow architecture and adopt the same fixed hyperparameters for the NFR algorithm (see Appendix A. 1 for details). ${ }^{4}$

### 4.1. Prior predictive checks

As introduced in Section 3.3, we place a Gaussian prior $\mathcal{N}\left(\boldsymbol{\phi} ; \mathbf{0}, \sigma_{\phi}^{2} \mathbf{I}\right)$ on the flow parameters $\phi$. Since a normalizing flow represents a probability distribution, drawing parameters from this prior generates different realizations of possible distributions. We calibrate the prior variance $\sigma_{\phi}$ by visually inspecting these realizations, choosing a value that affords sufficient flexibility for the distributions to vary from the base distribution while maintaining reasonably smooth shapes. ${ }^{5}$ Figure 3 shows density contours and samples from flow realizations under three different prior settings: $\sigma_{\phi} \in\{0.02,0.2,2\}$. Based on this analysis, we set the prior standard deviation $\sigma_{\phi}=0.2$ for all subsequent experiments in the paper.

> **Image description.** This image contains three scatter plots arranged horizontally, each displaying density contours.
>
> Each plot has the same basic structure:
>
> - **Axes:** Each plot has x and y axes. The x-axis ranges vary between plots. The y-axis ranges are similar in the first and third plots, from approximately -4 to 4, while the second plot ranges from -5 to 2.5.
> - **Data points:** The plots contain a scattering of gray data points, with the density of points varying across the plots.
> - **Contours:** Overlaid on the data points are density contours, represented by nested lines of different colors (green, light blue, and dark blue). The contours visually represent areas of higher data point density.
> - **Titles:** Above each plot is a title indicating the distribution from which the data is drawn. The titles are of the form "$\phi \sim \mathcal{N}(0, \sigma^2)$", where $\sigma^2$ varies between plots.
>   - Plot (a) has title "$\phi \sim \mathcal{N}(0, 0.02^2)$".
>   - Plot (b) has title "$\phi \sim \mathcal{N}(0, 0.2^2)$".
>   - Plot (c) has title "$\phi \sim \mathcal{N}(0, 2^2)$".
> - **Plot labels:** Below each plot is a label in parentheses: (a), (b), and (c), respectively.
>
> The key difference between the plots is the spread and shape of the data points and contours, which is influenced by the variance in the Gaussian distribution specified in the title. Plot (a) shows a tight, circular distribution. Plot (b) shows a slightly more elongated and spread-out distribution. Plot (c) shows a more complex, multi-modal distribution with two distinct clusters.

Figure 3: Effect of prior variance on normalizing flow behavior, using a standard Gaussian as the base distribution. The panels show flow realizations with different prior standard deviations $\sigma_{\phi}$ : (a) The flow closely resembles the base distribution. (b) The flow exhibits controlled flexibility, allowing meaningful deviations while maintaining reasonable shapes. (c) The flow deviates significantly, producing complex and less plausible distributions.

### 4.2. Benchmark evaluations

We evaluate NFR on several synthetic and real-world problems, each defined by a blackbox log-likelihood function and a log-prior function, or equivalently the target log-density function. The black-box nature of the likelihood means its gradients are unavailable, and we allow evaluations to be moderately expensive and potentially noisy. We are interested

[^0]
[^0]: 4. The code implementation of NFR is available at github.com/acerbilab/normalizing-flow-regression. 5. This approach is a form of expert prior elicitation (Mikkola et al., 2024) about the expected shape of posterior distributions, leveraging our experience in statistical modeling.

---

#### Page 9

in the offline inference setting, under the assumption that practitioners would have already performed multiple optimization runs for MAP estimation. Thus, to obtain training data for NFR, we collect log-density evaluations from MAP optimization runs using two popular black-box optimizers: CMA-ES (Hansen, 2016) and BADS, a hybrid Bayesian optimization method (Acerbi and Ma, 2017; Singh and Acerbi, 2024). For each problem, we allocate $3000 D$ log-density evaluation where $D$ is the posterior dimension (number of model parameters). The details of the real-world problems are provided in Appendix A.3. For consistency, we present results from CMA-ES runs in the main text, with analogous BADS results and additional details in Appendix A.4. Example visualizations of the flow approximation and baselines are provided in Appendix A.9.

Baselines. We compare NFR against three baselines:

1. Laplace approximation (Laplace; MacKay, 2003), which constructs a Gaussian approximation using the MAP estimate and numerical computation of the Hessian, requiring additional log-density evaluations (Brodtkorb and D'Errico, 2022).
2. Black-box variational inference (BBVI; Ranganath et al., 2014), using the same normalizing flow architecture as NFR plus a learnable diagonal Gaussian base distribution. BBVI estimates ELBO gradients using the score function (REINFORCE) estimator with control variates, optimized using Adam (Kingma and Ba, 2014). We consider BBVI using both $3000 D$ and $10 \times 3000 D$ target density evaluations, with the latter being substantially more than NFR presented as a 'higher budget' baseline. Details on the implementation are provided in Appendix A.5.
3. Variational sparse Bayesian quadrature (VSBQ; Li et al., 2024), which like NFR uses existing evaluations to estimate the posterior. VSBQ fits a sparse Gaussian process to the log-density evaluations and runs variational inference on this surrogate with a Gaussian mixture model. We give VSBQ the same $3000 D$ evaluations as NFR.

NFR and VSBQ are directly comparable as surrogate-based offline inference methods. BBVI requires additional evaluations of the target log density during training and is included as a strong online black-box inference baseline. Laplace requires additional log-density evaluation for the Hessian and serves as a popular approximate inference baseline.

Metrics. We assess algorithm performance by comparing the returned solutions against ground-truth posterior samples and normalizing constants. We use three metrics: the absolute difference between the true and estimated log normalizing constant ( $\Delta \mathrm{LML}$ ); the mean marginal total variation distance (MMTV); and the "Gaussianized" symmetrized KL divergence (GsKL) between the approximate and the true posterior (Acerbi, 2020; Li et al., 2024). MMTV quantifies discrepancies between marginals, while GsKL evaluates the overall joint distribution. Following previous recommendations, we consider approximations successful when $\Delta \mathrm{LML}<1$, MMTV $<0.2$ and $\mathrm{GsKL}<\frac{1}{8}$, with lower values indicating better performance (see Appendix A.2). For the stochastic methods (BBVI, VSBQ, and NFR), we report median performance and bootstrapped $95 \%$ confidence intervals from ten independent runs. We report only the median for the Laplace approximation, which is deterministic. Statistically significant best results are bolded, and metric values exceeding the desired thresholds are highlighted in red. See Appendix A. 2 for further details.

---

#### Page 10

# 4.2.1. SYNTHETIC PROBLEMS

Multivariate Rosenbrock-Gaussian $(D=6)$. We first test NFR on a six-dimensional synthetic target density with known complex geometry (Li et al., 2024). The target density takes the form:

$$
p(\mathbf{x}) \propto e^{\mathcal{R}\left(x_{1}, x_{2}\right)} e^{\mathcal{R}\left(x_{3}, x_{4}\right)} \mathcal{N}\left(\left[x_{5}, x_{6}\right] ; \mathbf{0}, \mathbb{I}\right) \cdot \mathcal{N}\left(\mathbf{x} ; \mathbf{0}, 3^{2} \mathbb{I}\right)
$$

which combines two exponentiated Rosenbrock ('banana') functions $\mathcal{R}(x, y)$ and a twodimensional Gaussian density with an overall isotropic Gaussian prior.

From Figure 4 and Table 1, we see that both NFR and VSBQ perform well, with all metrics below the desired thresholds. Still, NFR consistently outperforms VSBQ across all metrics, achieving excellent approximation quality on this complex target. In contrast, BBVI suffers from slow convergence and potential local minima, with several metrics exceeding the thresholds even with a $10 \times$ budget. Unsurprisingly, the Laplace approximation fails to capture the target's highly non-Gaussian structure.

> **Image description.** The image consists of four scatter plots arranged horizontally, each representing a different method: Laplace, BBVI (10x), VSBQ, and NFR. All plots share the same axes, labeled x3 on the horizontal axis and x4 on the vertical axis.
>
> Each plot displays a distribution of gray dots, representing samples. Overlaid on these dots are contour lines, colored in shades of green and blue, indicating density levels.
>
> - **Laplace:** The scatter plot shows a U-shaped distribution of gray dots. The contour lines are elongated and tilted diagonally across the plot, not aligned with the main distribution of the dots.
> - **BBVI (10x):** The scatter plot shows a U-shaped distribution. The contour lines are more concentrated at the bottom of the U-shape, better aligned with the densest region of the dots.
> - **VSBQ:** The scatter plot shows a U-shaped distribution. The contour lines are tightly packed at the bottom of the U-shape, closely following the shape of the dot distribution.
> - **NFR:** The scatter plot shows a U-shaped distribution. The contour lines are also tightly packed at the bottom of the U-shape, similar to VSBQ, indicating a good fit to the data.
>
> The grid lines are visible in the background of each plot.

Figure 4: Multivariate Rosenbrock-Gaussian $(D=6)$. Example contours of the marginal density for $x_{3}$ and $x_{4}$, for different methods. Ground-truth samples are in gray.

Table 1: Multivariate Rosenbrock-Gaussian $(D=6)$.

|                    | $\Delta \mathbf{L M L}(\downarrow)$ |        MMTV $(\downarrow)$        |          GsKL $(\downarrow)$          |
| :----------------- | :---------------------------------: | :-------------------------------: | :-----------------------------------: |
| Laplace            |                 1.3                 |               0.24                |                 0.91                  |
| BBVI $(1 \times)$  |           $1.3[1.2,1.4]$            |         $0.23[0.22,0.24]$         |           $0.54[0.52,0.56]$           |
| BBVI $(10 \times)$ |           $1.0[0.72,1.2]$           |         $0.24[0.19,0.25]$         |           $0.46[0.34,0.59]$           |
| VSBQ               |          $0.20[0.20,0.20]$          |       $0.037[0.035,0.038]$        |         $0.018[0.017,0.018]$          |
| NFR                | $\mathbf{0 . 0 1 3}[0.0079,0.017]$  | $\mathbf{0 . 0 2 8}[0.026,0.030]$ | $\mathbf{0 . 0 0 4 2}[0.0024,0.0068]$ |

Lumpy $(D=10)$. Our second test uses a fixed instance of the lumpy distribution (Acerbi, 2018), a mildly multimodal density represented by a mixture of 12 partially overlapping multivariate Gaussian components in ten dimensions. For this target distribution, all methods except Laplace perform well with metrics below the target thresholds, and NFR again achieves the best performance. The Laplace approximation provides reasonable estimates

---

#### Page 11

of the normalizing constant and marginal distributions but struggles with the full joint distribution. Further details are provided in Appendix A.4.

# 4.2.2. REAL-WORLD PROBLEMS

Bayesian timing model $(D=5)$. Our first real-world application comes from cognitive neuroscience, where Bayesian observer models are applied to explain human time perception (Jazayeri and Shadlen, 2010; Acerbi et al., 2012; Acerbi, 2020). These models assume that participants in psychophysical experiments are themselves performing Bayesian inference over properties of sensory stimuli (e.g., duration), using Bayesian decision theory to generate percept responses (Pouget et al., 2013; Ma et al., 2023). To make the inference scenario more challenging and realistic, we include log-likelihood estimation noise with $\sigma_{n}=3$, similar to what practitioners would find if estimating the log likelihood via Monte Carlo instead of precise numerical integration methods (van Opheusden et al., 2020).

As shown in Table 2, NFR and VSBQ accurately approximate this posterior, while BBVI $(10 \times)$ shows slightly worse performance with larger confidence intervals. BBVI $(1 \times)$ fails to converge, with all metrics exceeding the thresholds. The Laplace approximation is not applicable here due to the likelihood noise preventing reliable numerical differentiation.

Table 2: Bayesian timing model $(D=5)$.

|                    | $\Delta \mathbf{L M L}(\downarrow)$ |        MMTV $(\downarrow)$        |          GsKL $(\downarrow)$          |
| :----------------- | :---------------------------------: | :-------------------------------: | :-----------------------------------: |
| BBVI $(1 \times)$  |           $1.6[1.1,2.5]$            |         $0.29[0.27,0.34]$         |           $0.77[0.67,1.0]$            |
| BBVI $(10 \times)$ |   $\mathbf{0 . 3 2}[0.036,0.66]$    |        $0.11[0.088,0.15]$         |          $0.13[0.052,0.23]$           |
| VSBQ               |    $\mathbf{0 . 2 1}[0.18,0.22]$    | $\mathbf{0 . 0 4 4}[0.039,0.049]$ | $\mathbf{0 . 0 0 6 5}[0.0059,0.0084]$ |
| NFR                |    $\mathbf{0 . 1 8}[0.17,0.24]$    | $\mathbf{0 . 0 4 9}[0.041,0.052]$ | $\mathbf{0 . 0 0 8 6}[0.0053,0.011]$  |

Lotka-Volterra model $(D=8)$. Our second real-world test examines parameter inference for the Lotka-Volterra predatory-prey model (Carpenter, 2018), a classic system of coupled differential equations that describe population dynamics. Using data from Howard (2009), we infer eight parameters governing the interaction rates, initial population sizes, and observation noise levels.

Table 3 shows that NFR significantly outperforms all baselines on this problem. BBVI, VSBQ, and the Laplace approximation achieve acceptable performance, with all metrics below the desired thresholds except for the GsKL metric in the Laplace approximation.

Bayesian causal inference in multisensory perception $(D=12)$. Our final and most challenging test examines a model of multisensory perception from computational neuroscience (Acerbi et al., 2018). The model describes how humans decide whether visual and vestibular (balance) sensory cues share a common cause - a fundamental problem in neural computation (Körding et al., 2007). The model's likelihood is mildly expensive ( $>3 \mathrm{~s}$ per evaluation), due to the numerical integration required for its computation.

The high dimensionality and complex likelihood of this model make it particularly challenging for several methods. Due to a non-positive-definite numerical Hessian, the Laplace

---

#### Page 12

Table 3: Lotka-Volterra model $(D=8)$.

|                    |  $\Delta$ LML $(\downarrow)$  |        MMTV $(\downarrow)$        |            GsKL $(\downarrow)$            |
| :----------------- | :---------------------------: | :-------------------------------: | :---------------------------------------: |
| Laplace            |             0.62              |               0.11                |                   0.14                    |
| BBVI $(1 \times)$  |       $0.47[0.42,0.59]$       |       $0.055[0.048,0.063]$        |           $0.029[0.025,0.034]$            |
| BBVI $(10 \times)$ |       $0.24[0.23,0.36]$       |       $0.029[0.025,0.039]$        |          $0.0087[0.0052,0.014]$           |
| VSBQ               |       $0.95[0.93,0.97]$       |       $0.085[0.084,0.089]$        |           $0.060[0.059,0.062]$            |
| NFR                | $\mathbf{0 . 1 8}[0.17,0.18]$ | $\mathbf{0 . 0 1 6}[0.015,0.017]$ | $\mathbf{0 . 0 0 0 6 6}[0.00056,0.00083]$ |

approximation is inapplicable. The likelihood's computational cost makes BBVI $(1 \times)$, let alone $10 \times$, impractical to benchmark and to use in practice. ${ }^{6}$ Thus, we focus on comparing NFR and VSBQ (Table 4). NFR performs remarkably well on this challenging posterior, with metrics near or just above our desired thresholds, while VSBQ fails to produce a usable approximation.

Table 4: Multisensory $(D=12)$.

|      |              $\Delta$ LML $(\downarrow)$              |      MMTV $(\downarrow)$      |                  GsKL $(\downarrow)$                  |
| :--- | :---------------------------------------------------: | :---------------------------: | :---------------------------------------------------: |
| VSBQ | $4.1 \mathrm{e}+2[3.0 \mathrm{e}+2,5.4 \mathrm{e}+2]$ |       $0.87[0.82,0.93]$       | $2.0 \mathrm{e}+2[1.1 \mathrm{e}+2,4.1 \mathrm{e}+4]$ |
| NFR  |             $\mathbf{0 . 8 2}[0.75,0.90]$             | $\mathbf{0 . 1 3}[0.12,0.14]$ |            $\mathbf{0 . 1 1}[0.091,0.16]$             |

# 5. Discussion

In this paper, we introduced normalizing flow regression as a novel method for performing approximate Bayesian posterior inference, using offline likelihood evaluations. Normalizing flows offer several advantages: they ensure proper probability distributions, enable easy sampling, scale efficiently with the number of likelihood evaluations, and can flexibly incorporate prior knowledge of posterior structure. While we demonstrated that our proposed approach works well, it has limitations which we discuss in Appendix A.6. For practitioners, we further provide an ablation study of our design choices in Appendix A. 7 and a discussion on diagnostics for detecting potential failures in the flow approximation in Appendix A.8.

In this work, we focused on using log-density evaluations from MAP optimization due to its widespread practice, but our framework can be extended to incorporate other likelihood evaluation sources. For example, it could include evaluations of pre-selected plausible parameter values, as seen in cosmology (Rizzato and Sellentin, 2023), or actively and sequentially acquire new evaluations based on the current posterior estimate (Acerbi, 2018; Greenberg et al., 2019). We leave these topics as future work.

[^0]
[^0]: 6. From partial runs, we estimated $>100$ hours per run for BBVI $(1 \times)$ on our computing setup.
