# Normalizing flow regression

Bayesian inference with computationally expensive likelihood evaluations remains a significant challenge in many scientific domains. We propose normalizing flow regression (NFR), a novel offline inference method for approximating posterior distributions. Unlike traditional surrogate approaches that require additional sampling or inference steps, NFR directly yields a tractable posterior approximation through regression on existing log-density evaluations. See [our paper](https://openreview.net/pdf?id=lR0BGbw6hq) for more details.

## Set up

```bash
conda env create -f environment.yml
conda activate nfr
# install kernel for jupyter notebook
python -m ipykernel install --user --name nfr
```

See `demo.ipynb` for an example of using NFR.

## Citation
To appear in 7th Symposium on Advances in Approximate Bayesian Inference (AABI 2025, proceedings track).

> Li, C., Huggins, B., Mikkola, P., & Acerbi, L. (2025). Normalizing Flow Regression for Bayesian Inference with Offline Likelihood Evaluations. In 7th Symposium on Advances in Approximate Bayesian Inference.

### BibTeX
```bibtex
@inproceedings{liNormalizingFlowRegression2025,
  title = {Normalizing Flow Regression for {B}ayesian Inference with Offline Likelihood Evaluations},
  booktitle = {7th Symposium on Advances in Approximate Bayesian Inference},
  author = {Li, Chengkun and Huggins, Bobby and Mikkola, Petrus and Acerbi, Luigi},
  year = {2025},
  note = {To appear},
  url = {https://approximateinference.org/2025/},
}
```

## Acknowledgements

This repository includes code adapted from the `nflows` library: https://github.com/bayesiains/nflows, originally developed by Conor Durkan, Artur Bekasov, Iain Murray, and George Papamakarios.

We have modified `nflows/transforms/autoregressive.py` such that:
- When neural network parameters are zeros, the flow becomes the identity transform.
- The scale and shift transformation is constrained to a specified range.

<!--
## Notation

- target original/constrained space: the space where the target posterior is defined, potentially constrained by Cartesian product of intervals
- target inference/unconstrained space: obtained by applying a transformation (e.g., probit transform) to the target original space
- flow base space: the space where the flow base distribution is defined, unconstrained. `flow.transform`: target inference space -> flow base space, `flow.inverse_transform`: flow base space -> target inference space -->
