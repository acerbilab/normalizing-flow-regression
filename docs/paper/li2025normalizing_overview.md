# Normalizing Flow Regression for Bayesian Inference with Offline Likelihood Evaluations - Overview

Chengkun Li¹, Bobby Huggins², Petrus Mikkola¹, Luigi Acerbi¹

¹Department of Computer Science, University of Helsinki<br />
²Department of Computer Science and Engineering, Washington University in St. Louis

7th Symposium on Advances in Approximate Bayesian Inference (AABI) - Proceedings track, 2025

[Code](https://github.com/acerbilab/normalizing-flow-regression "View the paper codebase on GitHub") | [Paper](https://arxiv.org/abs/2504.11554 "Read the paper on arXiv") | [Social](https://arxiv.org/abs/2504.11554 "Read the paper thread") | [Markdown](https://github.com/acerbilab/normalizing-flow-regression/tree/main/docs/paper "Retrieve paper parts in Markdown (easy format for LLMs)")

### TL;DR

We propose **Normalizing Flow Regression (NFR)**, a novel offline inference method for approximating Bayesian posterior distributions using existing log-density evaluations. Unlike traditional surrogate approaches, NFR directly yields a tractable posterior approximation through regression on existing evaluations, without requiring additional sampling or inference steps. Our method performs well on both synthetic benchmarks and real-world applications from neuroscience and biology, offering a promising approach for Bayesian inference when standard methods are computationally prohibitive.

```bibtex
@article{li2025normalizing,
    title={Normalizing Flow Regression for Bayesian Inference with Offline Likelihood Evaluations},
    author={Li, Chengkun and Huggins, Bobby and Mikkola, Petrus and Acerbi, Luigi},
    journal={7th Symposium on Advances in Approximate Bayesian Inference (AABI) - Proceedings track},
    year={2025}
}
```

## Introduction

Bayesian inference with computationally expensive likelihood evaluations remains a significant challenge in scientific domains. When model evaluations involve extensive numerical methods or simulations, standard Bayesian approaches like MCMC or variational inference become impractical due to their requirement for numerous density evaluations.

Practitioners often resort to simpler alternatives like maximum a posteriori (MAP) estimation, but these point estimates fail to capture parameter uncertainty. Recent surrogate modeling approaches can approximate expensive density functions but typically don't directly yield valid probability distributions, requiring additional sampling or inference steps.

### Our Contribution

We propose **Normalizing Flow Regression (NFR)**, a novel offline inference method that directly yields a tractable posterior approximation through regression on existing log-density evaluations. Unlike other surrogate methods, NFR directly produces a posterior distribution that is easy to evaluate and sample from.

NFR efficiently recycles existing log-density evaluations (e.g., from MAP optimizations) rather than requiring costly new evaluations from the target model. This makes it particularly valuable in settings where standard Bayesian methods are computationally prohibitive.

## Background

### Normalizing Flows

Normalizing flows construct flexible probability distributions by transforming a simple base distribution (typically a multivariate Gaussian) through an invertible transformation $T\_{\\boldsymbol{\\phi}}: \\mathbb{R}^{D} \\rightarrow \\mathbb{R}^{D}$ with parameters $\\boldsymbol{\\phi}$.

For a random variable $\\mathbf{x}=T\_{\\boldsymbol{\\phi}}(\\mathbf{u})$ where $\\mathbf{u}$ follows base distribution $p\_{\\mathbf{u}}$, the change of variables formula gives its density as:

$$q_{\boldsymbol{\phi}}(\mathbf{x})=p_{\mathbf{u}}(\mathbf{u})\left|\operatorname{det} J_{T_{\boldsymbol{\phi}}}(\mathbf{u})\right|^{-1}, \quad \mathbf{u}=T_{\boldsymbol{\phi}}^{-1}(\mathbf{x})$$

We use masked autoregressive flow (MAF), which constructs the transformation through an autoregressive process:

$$ \mathbf{x}^{(i)}=g*{\text {scale }}\left(\alpha^{(i)}\right) \cdot \mathbf{u}^{(i)}+g*{\text {shift }}\left(\mu^{(i)}\right) $$

### Bayesian Inference

Bayesian inference uses Bayes' theorem to determine posterior distribution $p(\\mathbf{x} \\mid \\mathcal{D})$ of parameters $\\mathbf{x}$ given data $\\mathcal{D}$:

$$ p(\mathbf{x} \mid \mathcal{D})=\frac{p(\mathcal{D} \mid \mathbf{x}) p(\mathbf{x})}{p(\mathcal{D})} $$

Standard approximation approaches (VI and MCMC) require many likelihood evaluations, making them impractical for expensive models. While surrogate methods can approximate the target (log) density from limited evaluations, they don't directly yield proper probability distributions, requiring additional steps to obtain posterior samples. Our approach addresses this limitation by using normalizing flows as regression models that directly provide tractable posterior approximations.

## Normalizing Flow Regression (NFR)

### Key Innovation

NFR directly yields a tractable posterior approximation through regression on existing log-density evaluations, without requiring additional sampling or inference steps. The flow regression model provides both a normalized posterior density that's easy to evaluate and sample from, and an estimate of the normalizing constant (model evidence) for model comparison.

### Method Overview

We use a normalizing flow with normalized density $q\_{\\boldsymbol{\\phi}}(\\mathbf{x})$ to fit observations of the log density of an unnormalized target posterior. The log-density prediction of our regression model is:

$$ f*{\boldsymbol{\psi}}(\mathbf{x})=f*{\boldsymbol{\phi}}(\mathbf{x})+C $$

where $f\_{\\boldsymbol{\\phi}}(\\mathbf{x})=\\log q\_{\\boldsymbol{\\phi}}(\\mathbf{x})$ is the flow's log-density, and $C$ accounts for the unknown normalizing constant. We train the model via MAP estimation by maximizing:

$$ \mathcal{L}(\boldsymbol{\psi}) =\sum*{n=1}^{N} \log p\left(y*{n} \mid f*{\boldsymbol{\psi}}\left(\mathbf{x}*{n}\right), \sigma\_{n}^{2}\right)+\log p(\boldsymbol{\phi})+\log p(C) $$

### Robust Likelihood Function

Standard Gaussian likelihood for log-density observations would overemphasize near-zero density regions at the expense of high-density regions. We address this with a Tobit likelihood that censors observations below a threshold $y\_{\\text{low}}$.

> **Image description.** The image consists of two plots side-by-side, illustrating the censoring effect on a target density. Both plots have a similar x-axis, labeled "x", ranging from -16 to 16 with tick marks at -16, -8, 0, 8, and 16. The left plot shows "Density" on the y-axis, ranging from 0.0 to 0.4, with tick marks at 0.0, 0.2, and 0.4. A curve peaks sharply around x=0 and approaches 0 at the extremes. Two shaded regions with a criss-cross pattern are present near x=-16 and x=16, where the density is close to zero. The right plot displays "Log density" on the y-axis, ranging from -100 to 0, with tick marks at -100, -50, and 0. The curve is bell-shaped, peaking at x=0. A dashed horizontal line is drawn at approximately y=-50, labeled "y_low". Shaded regions with a criss-cross pattern are present where the curve falls below the "y_low" line, near x=-16 and x=16.
>
> **Caption:** **Illustration of the Tobit likelihood's censoring effect.** The shaded region represents censored observations with log-density values below $y\_{\\text{low}}$.

### Prior Specification and Optimization

We use a multivariate Gaussian base distribution estimated from high-density observations, and constrain flow transformations to stay reasonably close to this base distribution setting a prior over flows.

> **Image description.** This image contains three scatter plots arranged horizontally, each displaying density contours.
>
> Each plot has the same basic structure:
>
> - **Axes:** Each plot has x and y axes. The x-axis ranges vary between plots. The y-axis ranges are similar in the first and third plots, from approximately -4 to 4, while the second plot ranges from -5 to 2.5.
> - **Data points:** The plots contain a scattering of gray data points, with the density of points varying across the plots.
> - **Contours:** Overlaid on the data points are density contours, represented by nested lines of different colors (green, light blue, and dark blue). The contours visually represent areas of higher data point density.
> - **Titles:** Above each plot is a title indicating the distribution from which the data is drawn. The titles are of the form "$\\phi \\sim \\mathcal{N}(0, \\sigma^2)$", where $\\sigma^2$ varies between plots.
>   - Plot (a) has title "$\\phi \\sim \\mathcal{N}(0, 0.02^2)$".
>   - Plot (b) has title "$\\phi \\sim \\mathcal{N}(0, 0.2^2)$".
>   - Plot (c) has title "$\\phi \\sim \\mathcal{N}(0, 2^2)$".
> - **Plot labels:** Below each plot is a label in parentheses: (a), (b), and (c), respectively.
>
> The key difference between the plots is the spread and shape of the data points and contours, which is influenced by the variance in the Gaussian distribution specified in the title. Plot (a) shows a tight, circular distribution. Plot (b) shows a slightly more elongated and spread-out distribution. Plot (c) shows a more complex, multi-modal distribution with two distinct clusters.
>
> **Caption:** **Prior over flows.** Example flow realizations sampled from different priors over flow parameters $\\phi$, using a standard Gaussian as the base distribution. (a) Strong prior $\\rightarrow$ too rigid; (b) Intermediate prior $\\rightarrow$ reasonable shapes; (c) Weak prior $\\rightarrow$ ill-behaved distributions.

We follow an annealed optimization approach which gradually fits the flow to a tempered target across training iterations, providing stability.

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
>
> **Caption:** **Annealed optimization** progressively fits the flow to tempered observations, with inverse temperature $\\beta$ increasing over training iterations.

<details>
<summary>Show Algorithm Summary ▼</summary>

**Algorithm Summary**

**Input:** Observations $(\\mathbf{X}, \\mathbf{y}, \\boldsymbol{\\sigma}^{2})$, tempering steps $t\_{\\text{end}}$, max iterations $T\_{\\max}$

**Output:** Flow $T\_{\\phi}$ approximating the target, log normalizing constant $C$

1.  Compute base distribution from observations
2.  For $t \\leftarrow 0$ to $T\_{\\max}$ do:
    - Set inverse temperature $\\beta\_{t}$ according to tempering schedule
    - Update tempered observations
    - Optimize $C$ with fixed $\\boldsymbol{\\phi}$, then jointly optimize $(\\boldsymbol{\\phi}, C)$

</details>

## Experiments

We evaluate our normalizing flow regression (NFR) method through a series of experiments on both synthetic and real-world problems, comparing its performance against established baselines.

### Experimental Setup

For all experiments, we use log-density evaluations collected during maximum a posteriori (MAP) optimization runs, reflecting real-world settings where practitioners have already performed optimization to find parameter point estimates. We compare NFR against three baselines:

- **Laplace approximation**: A Gaussian approximation using the MAP estimate and numerical Hessian
- **Black-box variational inference (BBVI)**: Using the same flow architecture as NFR but trained through standard variational inference with up to ten times more likelihood evaluations ($10\\times$)
- **Variational sparse Bayesian quadrature (VSBQ)**: A state-of-the-art offline surrogate method using sparse Gaussian processes

We evaluate the methods using three metrics: the absolute difference between true and estimated log normalizing constant (ΔLML), the mean marginal total variation distance (MMTV), and the "Gaussianized" symmetrized KL divergence (GsKL).

### Results

We tested NFR on five benchmark problems of increasing complexity, from synthetic test cases to challenging real-world applications. The consolidated results demonstrate that NFR consistently outperforms baseline methods across problems of varying dimensionality and complexity.

| Problem (Dimension)  | Laplace (ΔLML↓) | Laplace (MMTV↓) | Laplace (GsKL↓) | BBVI (10×) (ΔLML↓) | BBVI (10×) (MMTV↓) | BBVI (10×) (GsKL↓) | VSBQ (ΔLML↓)  |  VSBQ (MMTV↓)  |  VSBQ (GsKL↓)   | NFR (ours) (ΔLML↓) | NFR (ours) (MMTV↓) | NFR (ours) (GsKL↓) |
| :------------------- | :-------------: | :-------------: | :-------------: | :----------------: | :----------------: | :----------------: | :-----------: | :------------: | :-------------: | :----------------: | :----------------: | :----------------: |
| Rosenbrock-G. (D=6)  |   _Poor_ 1.30   |   _Poor_ 0.24   |   _Poor_ 0.91   |    _Poor_ 1.00     |    _Poor_ 0.24     |    _Poor_ 0.46     |  _Good_ 0.20  |  _Good_ 0.037  |  _Good_ 0.018   |   **Best** 0.013   |   **Best** 0.028   |  **Best** 0.0042   |
| Lumpy (D=10)         |   _Good_ 0.81   |   _Good_ 0.15   |   _Poor_ 0.22   |    _Good_ 0.32     |    _Good_ 0.046    |    _Good_ 0.013    |  _Good_ 0.11  |  _Good_ 0.033  |  _Good_ 0.0070  |   **Best** 0.026   |   **Best** 0.022   |  **Best** 0.0020   |
| Timing Model (D=5)   |       N/A       |       N/A       |       N/A       |    _Good_ 0.32     |    _Good_ 0.11     |    _Good_ 0.13     | **Best** 0.21 | **Best** 0.044 | **Best** 0.0065 |   **Best** 0.18    |   **Best** 0.049   |  **Best** 0.0086   |
| Lotka-Volterra (D=8) |   _Good_ 0.62   |   _Good_ 0.11   |   _Poor_ 0.14   |    _Good_ 0.24     |    _Good_ 0.029    |   _Good_ 0.0087    |  _Good_ 0.95  |  _Good_ 0.085  |  _Good_ 0.060   |   **Best** 0.18    |   **Best** 0.016   |  **Best** 0.00066  |
| Multisensory (D=12)  |       N/A       |       N/A       |       N/A       |        N/A         |        N/A         |        N/A         | _Poor_ 4.1e+2 |  _Poor_ 0.87   |  _Poor_ 2.0e+2  |    _Good_ 0.82     |    _Good_ 0.13     |    _Good_ 0.11     |

**Results across all benchmark problems.** Lower values are better for all metrics. Best results are shown in **dark green (Best)**, acceptable results in _light green (Good)_, and poor results in _pink (Poor)_. N/A indicates that the method was not applicable or computationally infeasible for that problem.

### Key Results Highlights

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
>
> **Caption:** Example contours of the marginal density for the **Multivariate Rosenbrock-Gaussian** showing performance of different methods. Ground-truth samples are in gray.

**Synthetic Problems:** On the challenging Rosenbrock-Gaussian distribution (D=6) with its curved correlation structure, NFR achieves substantially better performance than all baselines, successfully capturing the complex posterior shape. The Laplace approximation fails to capture the non-Gaussian structure, while BBVI struggles with convergence issues.

For the mildly multimodal Lumpy distribution (D=10), NFR again shows the best overall performance, though all methods except Laplace perform reasonably well in this case.

### Real-World Applications

On real-world problems, NFR demonstrates particular strengths:

- **Bayesian Timing Model (D=5):** Both NFR and VSBQ accurately approximate this posterior, even with added log-likelihood estimation noise that makes the problem more challenging and realistic.
- **Lotka-Volterra Model (D=8):** NFR significantly outperforms all baselines on this problem with coupled differential equations, demonstrating its effectiveness on problems with moderate dimensionality and complex dynamics.
- **Multisensory Perception (D=12):** In our most challenging test, NFR performs remarkably well where the Laplace approximation is inapplicable and BBVI is computationally prohibitive. VSBQ completely fails to produce a usable approximation for this high-dimensional problem.

## Discussion and Conclusions

We introduced normalizing flow regression (NFR), a novel offline inference method that directly yields a tractable posterior approximation through regression on existing log-density evaluations, unlike traditional surrogate approaches that require additional sampling or inference steps.

Normalizing flows offer key advantages for this task: they ensure proper probability distributions, enable easy sampling, scale efficiently with evaluations, and flexibly incorporate prior knowledge. Our empirical evaluation demonstrates that NFR effectively approximates both synthetic and real-world posteriors, excelling particularly in challenging scenarios where standard methods are computationally prohibitive.

In the paper's appendix, we also discuss diagnostics to detect failures of the method, from simple visualizations to Pareto-smoothed importance sampling.

### Limitations and Future Work

While NFR shows promising results, it has limitations: it requires sufficient coverage of probability mass regions (challenging in dimensions D \> 15-20), depends on evaluations adequately exploring the posterior landscape, and needs careful prior specification. Future research directions include:

- Incorporating additional likelihood evaluation sources beyond MAP optimization
- Developing active learning strategies for sequential evaluation acquisition
- Exploring advanced flow architectures for higher-dimensional problems

NFR represents a promising approach for Bayesian inference in computationally intensive settings, offering robust, uncertainty-aware modeling across scientific applications.

> **Acknowledgments:** This work was supported by Research Council of Finland (grants 358980 and 356498), and by the Flagship programme: [Finnish Center for Artificial Intelligence FCAI](https://fcai.fi/). The authors wish to thank the Finnish Computing Competence Infrastructure (FCCI) for supporting this project with computational and data storage resources.

## References

1.  Acerbi, L. (2018). Variational Bayesian Monte Carlo. _Advances in Neural Information Processing Systems_, 31:8222-8232.
2.  Acerbi, L. (2020). Variational Bayesian Monte Carlo with noisy likelihoods. _Advances in Neural Information Processing Systems_, 33:8211-8222.
3.  Blei, D.M., Kucukelbir, A., and McAuliffe, J.D. (2017). Variational inference: A review for statisticians. _Journal of the American Statistical Association_, 112(518):859-877.
4.  Dinh, L., Sohl-Dickstein, J., and Bengio, S. (2017). Density estimation using Real NVP. _International Conference on Learning Representations_.
5.  Gelman, A., Carlin, J.B., Stern, H.S., Dunson, D.B., Vehtari, A., and Rubin, D.B. (2013). _Bayesian Data Analysis_ (3rd edition). CRC Press.
6.  Li, C., Clarté, G., Jørgensen, M., and Acerbi, L. (2024). Fast post-process Bayesian inference with variational sparse Bayesian quadrature. _arXiv preprint_ arXiv:2303.05263.
7.  Papamakarios, G., Pavlakou, T., and Murray, I. (2017). Masked autoregressive flow for density estimation. _Advances in Neural Information Processing Systems_, 30.
8.  Papamakarios, G., Nalisnick, E., Rezende, D.J., Mohamed, S., and Lakshminarayanan, B. (2021). Normalizing flows for probabilistic modeling and inference. _Journal of Machine Learning Research_, 22(57):1-64.
9.  Ranganath, R., Gerrish, S., and Blei, D. (2014). Black box variational inference. _Artificial Intelligence and Statistics_, 814-822.
10. Rezende, D.J. and Mohamed, S. (2015). Variational inference with normalizing flows. _Proceedings of the 32nd International Conference on Machine Learning_, 1530-1538.

---

© 2025 Chengkun Li, Bobby Huggins, Petrus Mikkola, Luigi Acerbi

Find more information at: [https://github.com/acerbilab](https://github.com/acerbilab)
