# To-Dos for the Project

## Computing Mismatch Index

- [x] Download input-output table for years 2005 - 2011
- [x] Numerical solution for optimal unemployment allocation
- [x] Compute $\theta = (I-A)\gamma$, where $\gamma$ is the ratio of sector output to GDP.

## July 8th Update

- [x] Sectoral breakdown of mismatch
- [ ] Check Sahin et al calculation with diminishing return to scale labor.
- [ ] Equalize marginal benefit of labor across sectors and compare with Sahin et al
- [ ] Incorporate incomplete markets

## Jul 13th Notes

The key issue with checking Sahin et al calculations with DRS for labor is that some of the $\theta$s are negative. This comes from the fact that $\theta$'s are calibrated from relationships derived from assuming a network structure. This permits the existence of negative $\theta$'s, which happens mostly for raw material production sectors, such as mining. If we use this $\theta$ for calculating the mismatch index for Sahin et al, this simply means that consumers have no preference for mining outputs.

Alternatively, we can try computing $\theta$s from consumer expenditure on sectoral outputs. It might still be the case that $\theta$s are close to 0 for intermediary sectors. Specifically, we want to compute $\theta$ as

$$\theta_i = \frac{p_iC_i}{G},$$

where 

$$p_iC_i = p_i(Y_i - \sum_j x_{ij})$$
