# Data Description

This file contains the description for the cleaned csv files for the computation of optimal unemployment with and without production linkages.

## Input and Labor Shares

`A.csv` contains the share of input used by sector A from sector B. Each row documents the input usage of a sector from all other sectors. Labor share is excluded and stored separately in `labor_share.csv`. This is calculated from data in 2007.

## Key Parameters

We calibrate the key parameters from sector and aggregate outputs, as well as info shares. Our key parameters include the sales share γ, the preference parameter θ, the sector-specific matching parameter φ, and the preference-weighted sectoral importance parameter λ. They are all in the file `params.csv`.

## JOLTs Data

`labor_market_monthly.csv` and `labor_market_yearly.csv` contain JOLTs data on vacancy and unemployment. It also contains data on employment. The data is available both at the sector level and the aggregate level (denoted by capital letters).