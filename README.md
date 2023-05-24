# Unemployment in a Production Network

This repository hosts the calibration code and data accompanying the paper [Unemployment in a Production Network](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4449027). In particular, we calibrate the production network model presented in the paper to the US economy under different labor market specifications, and examine the first-order response of macroeconomic variables and disaggregated economic variables to technology shocks, fixed factor shocks, and labor supply shocks.

## Data

The `data` folder hosts the raw and cleaned versions of the datasets we use. These datasets are publicly available. We obtain them from the Bureau of Economic Analysis, the Bureau of Labor Statistics, and the Current Population Survey. The paper has a detailed description of where the datasets are from and how we cleaned them. Also, the cleaning process is replicable through our Jupyter notebooks in `code/clean_data/`. The clean datasets are stored in `data/clean/`

## Code

The `code` folder houses the code we use to clean the raw data, as well as code that we use to calibrate and compute the first-order responses of our model. In particular, the `calib` notebooks in the `code` folder produces our main calibration results, and compute responses to different types of shocks under different labor market assumptions. The `code/clean_data` folder contains the code we use to clean the raw data. The `code/functions` folder contains the code for plotting, as well as computing an array of endogenous variables and equilibrium objects.

## Output

The `output` folder contains our results, which are a collection of figures of how sectoral and aggregate variables respond to different shocks.