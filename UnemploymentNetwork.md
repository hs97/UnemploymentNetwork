# Unemployment in Production Networks

## Research Questions

- What is the impact of sector-specific productivity shocks on the aggregate labor market?
- How do production linkages amplify the impact of these shocks?
- How do sectoral unemployment rates respond to idiosyncratic shocks and comove based on input-output linkages?

## Approach

1. Production network model with sector-specific labor market and searching/matching.
2. Labor is immobile across sectors.
3. Productivity shocks propagate through the economy via direct input-output linkages that affects labor demand. **What other propagation mechanisms can we think of? Is there a household demand story through unemployment? How do we incorporate it to the model?**
4. Derive a Leontief-inverse-type relationship for productivity shocks and unemployment or labor market tightness.
5. Estimate sector-specific matching elasticities, differential job-finding rates, and separation rates. Simulate shocks to obtain propagation patterns. Match the predictions with empirical data. In particular, can see whether oil shocks generate unemployment propagation similar to our calibrated model.
6. Alternatively, compute sector-specific productivity shocks (with GIV) and estimate the tightness Leontief inverse.
7. Use input-output table to quantify cross sector linkages.
8. Quantify aggregate effect using total effects including spillovers to other industries - a Holten's theorem parallel for labor market tightness, unemployment, etc.

## Toy Model

There are two sectors $1$ and $2$, each with production function:

$$y_i = A_i N_i^{\alpha_i} \prod_{j\neq i} X_{ij}^{\omega_{ij}}.$$

For simplicity, we assume a vertical economy:

$$y_1 = A_1 N_1^{\alpha_1},\\ y_2 = A_2N_2^{\alpha_2} y_1^{\beta_2}.$$

The sector recruits production workers with recruiters $R_i$, who can each produce $\frac{1}{\kappa}$ vacancies $V_i$, and pay wage $w_i$ to all employees $L^d_i = N_i + R_i$. Letting $\tau_i$ denote the recruiter-producer ration, $\frac{N_i}{R_i}$, sector 1 solves
$$\max_{N_1} A_1 N_1^{\alpha_1} - w_1(1+\tau_1) N_1. $$

The first order condition governing the sector's optimal employment choice is
$$ \alpha_1 A_1N_1^{\alpha_1-1} = w_1(1+\tau_1).$$

This implies that the demand for production workers is
$$
N_1^d = \left(\frac{\alpha_1 A_1}{w_1 (1+\tau_1)}\right)^{\frac{1}{1-\alpha_1}}.
$$
Which implies that
$$
    L_1^d = (1+\tau_1)N^d_1 = (1+\tau_1) \left(\frac{\alpha_1 A_1}{w_1 (1+\tau_1)}\right)^{\frac{1}{1-\alpha_1}} = \left(\frac{\alpha_1 A_1}{w_1 (1+\tau_1)^{\alpha_1}}\right)^{\frac{1}{1-\alpha_1}}
$$

Assuming balanced flows, we have that
$$
    s_1 L^d_1 = q(\theta_1) V_1
$$
Using $R_1 = \kappa V_1$ and $L^d_1 = (1+\tau_t)N^d_1$
$$
    s (1+\tau_t) N^d_t = q(\theta_t) \frac{1}{\kappa} R_t
$$
Which pins down $\tau_t = \frac{R_t}{N_t}$ as
$$
    \kappa s (1+\tau_1) = q(\theta_1) \tau_1 \Rightarrow \tau_1 = \frac{\kappa s}{q(\theta_1)-\kappa s}.
$$

Similarly, for sector 2, we have that the following maximization problem:

$$\max_{N_2} A_2 N_2^{\alpha_2}A_1^{\beta_2}N_1^{\alpha_1\beta_2} - w_2(1+\tau_2) N_2 - y_1. $$

The first order condition governing the sector's optimal employment choice is

$$ \alpha_2 A_2N_2^{\alpha_2-1}A_1^{\beta_2}N_1^{\alpha_1\beta_2} = w_2(1+\tau_2).$$

This implies that the demand for production workers is
$$
N_2^d = \left(\frac{\alpha_2 A_2A_1^{\beta_2}N_1^{\alpha_1\beta_2}}{w_2 (1+\tau_2)}\right)^{\frac{1}{1-\alpha_2}}.
$$
Which implies that
$$
    L_2^d = (1+\tau_2)N^d_2 = (1+\tau_2) \left(\frac{\alpha_2 A_2 A_1^{\beta_2}N_1^{\alpha_1\beta_2}}{w_2 (1+\tau_2)}\right)^{\frac{1}{1-\alpha_2}} = \left(\frac{\alpha_2A_2A_1^{\beta_2}N_1^{\alpha_1\beta_2}}{w_2 (1+\tau_2)^{\alpha_2}}\right)^{\frac{1}{1-\alpha_2}}
$$

Substituting in the value for $N_1$, we have that:

$$L_2^d = \left(\frac{\alpha_1 A_1^{\frac{1}{\alpha_1}}}{w_1 (1+\tau_1)}\right)^{\frac{\alpha_1}{1-\alpha_1}\frac{\beta_2}{1-\alpha_2}}\left(\frac{\alpha_2A_2}{w_2 (1+\tau_2)^{\alpha_2}}\right)^{\frac{1}{1-\alpha_2}}.$$

The above equation demonstrates how productivity shocks to sector 1 can propagate to the labor demand for sector 2 through input-output linkages. Combining linkages and labor demand allows us to look at how productivity shocks propagate through sectoral labor markets and unemployment (once we close the model with a simple labor supply condition).

## Contribution to the Literature

- No explicit treatment of unemployment in production networks. Arguably a very important channel.
  
## Policy Implications

- Fiscal policy should take into labor market characteristics and upstreamness into account
  - should target upstream sectors since upstream shocks propagate further
  - should target sectors with high elasticity of tightness wrt productivity

## Alternative ideas
- A labor market model with heterogenous frictions across different skill levels + differential access to credit
- Estimating hysteresis effects in across european countries using similar method to Furlanetto et al. (2021). Ask how much of the differential unemployment rates across european countries these shocks can explain.
