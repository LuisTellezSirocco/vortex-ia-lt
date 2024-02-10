# Vortex-IA-LT attack
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
<!-- [![All Contributors](https://img.shields.io/badge/all_contributors-6-orange.svg?style=flat-square)](#contributors-) -->
<!-- ALL-CONTRIBUTORS-BADGE:END -->

# Instalation

We will be using `conda` to manage the environment, so first we need to install it. You can download it from [here](https://docs.conda.io/en/latest/miniconda.html).

Once you have installed `conda`, you can create the environment by running the following command:

```bash
conda create -n vortexia python=3.10
````
Then, you can activate the environment by running:

```bash
conda activate vortexia
```

Now, you can install the packages by running:

```bash
pip install -e .
```

Create notebook kernel
```bash
python -m ipykernel install --user --name vortexia --display-name "vortex-ia-lt"
```

This will install the package in editable mode, so you can modify the code and test it without having to reinstall the package. And also it will install the required packages.


## Contributors ✨

Thanks goes to:

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/LuisTellezSirocco"><img src="https://avatars.githubusercontent.com/u/110382845?s=96&v=4" width="100px;" alt=""/><br /><sub><b>Luis Téllez</b></sub></a><br /><a href="https://github.com/LuisTellezSirocco" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/gcastro-98"><img src="https://avatars.githubusercontent.com/u/83754427?v=4" width="100px;" alt=""/><br /><sub><b>Gerard Castro</b></sub></a><br /><a href="https://github.com/gcastro-98" title="Code">💻</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
