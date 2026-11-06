# Citing PINNStudio

If PINNStudio is useful in your work, please cite the version you used. PINNStudio 1.6.0 is archived on Zenodo with
the DOI [10.5281/zenodo.24023305](https://doi.org/10.5281/zenodo.24023305):

```bibtex
@software{khan2026pinnstudio,
  author    = {Khan, Asfandyar and Mamivand, Mahmood},
  title     = {PINNStudio: A graphical platform for forward and inverse physics-informed neural networks},
  version   = {1.6.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24023305},
  url       = {https://doi.org/10.5281/zenodo.24023305}
}
```

The repository's [`CITATION.cff`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/CITATION.cff) has the same
authors, title, version, release date and DOI, and GitHub's **Cite this repository** button formats it for you. You can
see the version you have in PINNStudio under **Help › About PINNStudio**, or with `pip show pinnstudio`.

## The paper

A paper describing PINNStudio is in preparation. It has not been published yet;
this page will give its full reference once it is. Until then, please cite the software as above. Once the paper is
out, cite the paper for the methods and their validation, and the archived software version for the exact
implementation you used.

## Methods and examples

PINNStudio implements methods and reproduces benchmark problems from the literature. When you use one of them, please
also cite the original work. Each method page of the [feature reference](../reference/index.md) and each
[example](../examples/index.md) names its source, and the full list is on [Scientific references](references.md).
In particular, every script PINNStudio generates runs on DeepXDE:

> Lu, L., Meng, X., Mao, Z., & Karniadakis, G. E. (2021). DeepXDE: A deep learning library for solving differential
> equations. *SIAM Review*, 63(1), 208–228. <https://doi.org/10.1137/19M1274067>
