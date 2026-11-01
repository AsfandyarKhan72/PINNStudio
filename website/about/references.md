# Scientific references

The works below are those PINNStudio's methods and examples are based on. Each entry says where it is used. Other
pages cite them by author and year with a link to the entry's anchor, for example
`references.md#raissi2019`. When you use one of these methods or examples in your work, please cite the original
publication as well as PINNStudio ([Citing PINNStudio](cite.md)).

DOIs are given where they could be confirmed; entries without a DOI give the full bibliographic reference or a stable
URL instead.

## Software

<a id="lu2021"></a>

### Lu et al. 2021 {#lu2021deepxde}

Lu, L., Meng, X., Mao, Z., & Karniadakis, G. E. (2021). DeepXDE: A deep learning library for solving differential
equations. *SIAM Review*, 63(1), 208–228. <https://doi.org/10.1137/19M1274067>

DeepXDE is the library every script PINNStudio generates runs on. Its documented examples are among PINNStudio's
examples, each linked to its original script, and the paper's L-shaped Poisson problem is the example
*Poisson 2D – L-shape*. The residual-based adaptive refinement (RAR) offered under **Adaptive training** follows
this paper.

## Physics-informed neural networks and benchmark problems

### Raissi et al. 2019 {#raissi2019}

Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework
for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of
Computational Physics*, 378, 686–707. <https://doi.org/10.1016/j.jcp.2018.10.045>

The PINN method. The example *Burgers 1D – Raissi et al.* follows the paper's Burgers problem (its *Default* settings
use the paper's network, points and L-BFGS training); its inverse mode recovers both coefficients of the paper's
data-driven discovery form. DeepXDE's Burgers and Schrödinger examples study problems from this paper.

### Wight & Zhao 2021 {#wight2021}

Wight, C. L., & Zhao, J. (2021). Solving Allen-Cahn and Cahn-Hilliard equations using the adaptive physics informed
neural networks. *Communications in Computational Physics*, 29(3), 930–954.
<https://doi.org/10.4208/cicp.OA-2020-0086>

The Allen-Cahn (1D, 2D, 3D) and Cahn-Hilliard (1D, 2D) examples *– Wight & Zhao*, and the time-adaptive training
approaches (time windows, a growing time range, resampling in each window).

### Mattey & Ghosh 2022 {#mattey2022}

Mattey, R., & Ghosh, S. (2022). A novel sequential method to train physics informed neural networks for Allen-Cahn
and Cahn-Hilliard equations. *Computer Methods in Applied Mechanics and Engineering*, 390, 114474.
<https://doi.org/10.1016/j.cma.2021.114474>

The Allen-Cahn and Cahn-Hilliard examples *– Mattey & Ghosh*, and the backward-compatible time windows (bc-PINN) of
time-adaptive training.

### Mathias et al. 2022 {#mathias2022}

Mathias, M. S., de Almeida, W. P., de Barros, M. R., Coelho, J. F., et al. (2022). Augmenting a physics-informed
neural network for the 2D Burgers equation by addition of solution data points. In *Intelligent Systems (BRACIS
2022)*, Lecture Notes in Computer Science 13654. Springer. <https://doi.org/10.1007/978-3-031-21689-3_28>
([arXiv:2301.07824](https://arxiv.org/abs/2301.07824))

The example *Burgers 2D – Mathias et al.*

### Jin et al. 2021 {#jin2021}

Jin, X., Cai, S., Li, H., & Karniadakis, G. E. (2021). NSFnets (Navier-Stokes flow nets): Physics-informed neural
networks for the incompressible Navier-Stokes equations. *Journal of Computational Physics*, 426, 109951.
<https://doi.org/10.1016/j.jcp.2020.109951>

The Navier-Stokes examples from DeepXDE (*Navier-Stokes 2D – Kovasznay flow*, *Navier-Stokes 3D – Beltrami flow*)
follow problems studied in this paper.

### Ethier & Steinman 1994 {#ethier1994}

Ethier, C. R., & Steinman, D. A. (1994). Exact fully 3D Navier-Stokes solutions for benchmarking. *International
Journal for Numerical Methods in Fluids*, 19(5), 369–375.

The exact three-dimensional unsteady solution of the incompressible Navier-Stokes equations used by the Beltrami
flow example.

### Haghighat et al. 2021 {#haghighat2021}

Haghighat, E., Raissi, M., Moure, A., Gomez, H., & Juanes, R. (2021). A physics-informed deep learning framework for
inversion and surrogate modeling in solid mechanics. *Computer Methods in Applied Mechanics and Engineering*, 379,
113741. <https://doi.org/10.1016/j.cma.2021.113741>

DeepXDE's examples *Linear elasticity 2D – plate* and *Linear elasticity 2D – plate, soft BCs* follow problems
studied in this paper.

### Pang et al. 2019 {#pang2019}

Pang, G., Lu, L., & Karniadakis, G. E. (2019). fPINNs: Fractional physics-informed neural networks. *SIAM Journal on
Scientific Computing*, 41(4), A2603–A2626. <https://doi.org/10.1137/18M1229845>

DeepXDE's example *Fractional Poisson 1D* follows a problem studied in this paper. Fractional terms `frac(...)` in
PINNStudio's equations use DeepXDE's discretization of the fractional operator (Grünwald-Letnikov), as in fPINNs.

### Yu et al. 2022 {#yu2022}

Yu, J., Lu, L., Meng, X., & Karniadakis, G. E. (2022). Gradient-enhanced physics-informed neural networks for forward
and inverse PDE problems. *Computer Methods in Applied Mechanics and Engineering*, 393, 114823.
<https://doi.org/10.1016/j.cma.2022.114823>

DeepXDE's examples *Poisson 1D – hard BCs*, *Diffusion-reaction 1D* and the inverse examples
*Brinkman-Forchheimer 1D – unknown ν_e, K* and *Diffusion-reaction 1D – unknown rate field* follow problems studied in
this paper.

### Escapil-Inchauspé & Ruz 2023 {#escapil2023}

Escapil-Inchauspé, P., & Ruz, G. A. (2023). Hyper-parameter tuning of physics-informed neural networks: Application
to Helmholtz problems. *Neurocomputing*, 561. ([arXiv:2205.06704](https://arxiv.org/abs/2205.06704))

DeepXDE's examples *Helmholtz 2D – Dirichlet* and *Helmholtz 2D – hyperparameter search* follow problems studied in
this paper. The **Bayesian search** mode of **Parameter sweep** uses the same kind of search as the hyperparameter
search example (a Gaussian process model with expected improvement).

## Sampling and training methods

<a id="wu2023"></a>

### Wu et al. 2023 {#wu2023rar}

Wu, C., Zhu, M., Tan, Q., Kartha, Y., & Lu, L. (2023). A comprehensive study of non-adaptive and residual-based
adaptive sampling for physics-informed neural networks. *Computer Methods in Applied Mechanics and Engineering*, 403,
115671. <https://doi.org/10.1016/j.cma.2022.115671>

DeepXDE's examples *Diffusion 1D* and *Diffusion 1D – hard IC and BCs* follow problems studied in this paper,
which compares fixed and residual-based adaptive sampling of the collocation points.

### Anagnostopoulos et al. 2024 {#anagnostopoulos2024}

Anagnostopoulos, S. J., Toscano, J. D., Stergiopulos, N., & Karniadakis, G. E. (2024). Residual-based attention in
physics-informed neural networks. *Computer Methods in Applied Mechanics and Engineering*, 421, 116805.
<https://doi.org/10.1016/j.cma.2024.116805> ([arXiv:2307.00379](https://arxiv.org/abs/2307.00379))

**Loss weights** › per-point weights, **RBA**.

### McClenny & Braga-Neto 2023 {#mcclenny2023}

McClenny, L. D., & Braga-Neto, U. M. (2023). Self-adaptive physics-informed neural networks. *Journal of
Computational Physics*, 474, 111722. <https://doi.org/10.1016/j.jcp.2022.111722>
([arXiv:2009.04544](https://arxiv.org/abs/2009.04544))

**Loss weights** › per-point weights, **SA-PINN**.

<a id="wang2024"></a>

### Wang, Sankaran & Perdikaris 2024 {#wang2024causal}

Wang, S., Sankaran, S., & Perdikaris, P. (2024). Respecting causality for training physics-informed neural networks.
*Computer Methods in Applied Mechanics and Engineering*, 421, 116813. <https://doi.org/10.1016/j.cma.2024.116813>
([arXiv:2203.07404](https://arxiv.org/abs/2203.07404))

**Loss weights** › **Causal training**.

### Lagaris et al. 1998 {#lagaris1998}

Lagaris, I. E., Likas, A., & Fotiadis, D. I. (1998). Artificial neural networks for solving ordinary and partial
differential equations. *IEEE Transactions on Neural Networks*, 9(5), 987–1000.
<https://doi.org/10.1109/72.712178> ([arXiv:physics/9705023](https://arxiv.org/abs/physics/9705023))

Exact (hard) initial conditions built into the network output.

### Dong & Ni 2021 {#dong2021}

Dong, S., & Ni, N. (2021). A method for representing periodic functions and enforcing exactly periodic boundary
conditions with deep neural networks. *Journal of Computational Physics*, 435, 110242.
<https://doi.org/10.1016/j.jcp.2021.110242> ([arXiv:2007.07442](https://arxiv.org/abs/2007.07442))

Exact periodicity (periodic inputs \( \cos(2\pi x/L) \), \( \sin(2\pi x/L) \)).

<a id="wang2021"></a>

### Wang, Wang & Perdikaris 2021 {#wang2021fourier}

Wang, S., Wang, H., & Perdikaris, P. (2021). On the eigenvector bias of Fourier feature networks: From regression to
solving multi-scale PDEs with physics-informed neural networks. *Computer Methods in Applied Mechanics and
Engineering*, 384, 113938. <https://doi.org/10.1016/j.cma.2021.113938>

The multi-scale Fourier feature networks **MsFFN** and **STMsFFN** (**Neural network** › network type), and the
examples that use them.

### Rathore et al. 2024 {#rathore2024}

Rathore, P., Lei, W., Frangella, Z., Lu, L., & Udell, M. (2024). Challenges in training PINNs: A loss landscape
perspective. *Proceedings of the 41st International Conference on Machine Learning (ICML)*, PMLR 235.
<https://proceedings.mlr.press/v235/rathore24a.html>

The NNCG optimizer (Nyström-preconditioned Newton conjugate gradient) offered for a training phase.

### Wang, Teng & Perdikaris 2021 {#wang2021gradient}

Wang, S., Teng, Y., & Perdikaris, P. (2021). Understanding and mitigating gradient flow pathologies in
physics-informed neural networks. *SIAM Journal on Scientific Computing*, 43(5), A3055–A3081.
<https://doi.org/10.1137/20M1318043>

Gradient-norm loss balancing. It was tested in the project's training methods study and is not offered in
PINNStudio; it is listed because the study refers to it.

## Numerical methods for the reference solutions

Some of the reference solutions in `reference_data/` are computed by the scripts in the repository's `tools/` folder
with a Fourier pseudo-spectral method in space and fourth-order exponential time differencing (ETDRK4) in time (see
[Reference data](../reproducibility/reference-data.md)).

### Cox & Matthews 2002 {#cox2002}

Cox, S. M., & Matthews, P. C. (2002). Exponential time differencing for stiff systems. *Journal of Computational
Physics*, 176(2), 430–455. <https://doi.org/10.1006/jcph.2002.6995>

The exponential time differencing Runge-Kutta scheme ETDRK4.

### Kassam & Trefethen 2005 {#kassam2005}

Kassam, A.-K., & Trefethen, L. N. (2005). Fourth-order time-stepping for stiff PDEs. *SIAM Journal on Scientific
Computing*, 26(4), 1214–1233. <https://doi.org/10.1137/S1064827502410633>

The numerically stable evaluation of ETDRK4 used for the phase-field reference solutions.
