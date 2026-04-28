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
