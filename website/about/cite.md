# Citing PINNStudio

If PINNStudio is useful in your work, please cite the software. The repository's
[`CITATION.cff`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/CITATION.cff) holds the same information,
and GitHub's **Cite this repository** button formats it for you.

```bibtex
@software{khan2026pinnstudio,
  author  = {Khan, Asfandyar and Mamivand, Mahmood},
  title   = {PINNStudio: A graphical platform for forward and inverse physics-informed neural networks},
  year    = {2026},
  url     = {https://github.com/AsfandyarKhan72/PINNStudio}
}
```

If you used a released version, add its number, for example `version = {1.5.0}`. You can see the installed version
with `pip show pinnstudio`.

## The paper

A paper describing PINNStudio is in preparation. It has not been published yet;
this page will give its full reference and DOI once it is. Until then, please cite the software as above.

## Methods and examples

PINNStudio implements methods and reproduces benchmark problems from the literature. When you use one of them, please
also cite the original work. Each method page of the [feature reference](../reference/index.md) and each
[example](../examples/index.md) names its source, and the full list is on [Scientific references](references.md).
In particular, every script PINNStudio generates runs on DeepXDE:

> Lu, L., Meng, X., Mao, Z., & Karniadakis, G. E. (2021). DeepXDE: A deep learning library for solving differential
> equations. *SIAM Review*, 63(1), 208–228. <https://doi.org/10.1137/19M1274067>
