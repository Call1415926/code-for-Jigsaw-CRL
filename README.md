This repository contains the MATLAB implementation for the **Jigsaw-CRL** model.

## Main File

The main script is:

`main_final_submit.m`

This file serves as the entry point of the project and includes **three different causal network settings** with varying scales, as presented in the paper.

Additional implementation details and configurable parameters can be found in the inline comments of the script.

## Important Parameter: `is_n_infinity`

This parameter controls whether the algorithm uses the **theoretical precision matrix** (infinite-sample setting) or the **sample-based precision matrix** (finite-sample setting).

- `is_n_infinity = 0`  
  Use the finite-sample setting.  
  In this case, users need to specify the sample size `n` and tune the hyperparameter `tau` to obtain the final estimation results.

- `is_n_infinity = 1`  
  Use the infinite-sample setting.  
  The algorithm directly uses the true theoretical precision matrix to verify the theoretical results in the paper.

## Output

Running `main_final_submit.m` will produce the results of the **three evaluation metrics** under the selected setting.

## Appendix N.6: Sensitivity and Scalability

`main_appendix_N6.m` implements the sensitivity and scalability
experiments described in Appendix N.6. It generates random
Erdős–Rényi DAGs and client subsets, and evaluates three parameter sweeps:

- Latent dimension: `p = [10, 20, 30, 40]`, with `K = 15`, `q = 0.6`.
- Number of clients: `K = [5, 10, 15, 20]`, with `p = 20`, `q = 0.6`.
- Client coverage: `q = [0.5, 0.6, 0.7, 0.8]`, with `p = 20`, `K = 15`.

Run from the repository root in MATLAB:

```matlab
main_appendix_N6('full')        % All finite-sample and population settings
```

The full experiment uses sample sizes
`[1e4, 5e4, 1e5, 5e5, 1e6, Inf]`.
Results, including the three evaluation metrics and estimation runtime,
are saved under `results_N6/full/`.

The script uses an explicit random seed and the repository's existing
estimation and evaluation functions.
