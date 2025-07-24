# SSEC-JHU flfm

[![CI](https://github.com/ssec-jhu/flfm/actions/workflows/ci.yml/badge.svg)](https://github.com/ssec-jhu/flfm/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/flfm/badge/?version=latest)](https://flfm.readthedocs.io/en/latest/)
[![codecov](https://codecov.io/gh/ssec-jhu/flfm/graph/badge.svg?token=l3ND0EA9dE)](https://codecov.io/gh/ssec-jhu/flfm)
[![Security](https://github.com/ssec-jhu/flfm/actions/workflows/security.yml/badge.svg)](https://github.com/ssec-jhu/flfm/actions/workflows/security.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15353540.svg)](https://doi.org/10.5281/zenodo.15353540)


![SSEC-JHU Logo](docs/_static/SSEC_logo_horiz_blue_1152x263.png)

# Fourier Light Field Microscopy (FLFM)

Fourier Light Field Microscopy (FLFM) is a technique for scan-free volumetric imaging.
FLFM utilizes an array of lenses to simultaneously acquire several images from different viewpoints. These images are
then post-processed to generate a 3D volume using a Richardson-Lucy-based deconvolution algorithm. This enables
volumetric imaging at the exposure time of the camera, a speed unmatched by conventional volumetric scanning.
This fast imaging is particularly useful for samples with transient signals, where the time spent scanning will miss
relevant information. For example, FLFM is ideal for capturing the transient activity of point-like neurons in 3D.


# Quickstart

```term
pip install git+https://github.com/ssec-jhu/flfm.git
```

See [Usage](#usage) for quick and easy usage instructions.


# Installation, Build, & Run instructions

### Git LFS:

 * This repo uses git Large File Storage (git-lfs) for tracking data files, i.e., image file. To download these data
   files git-lfs is required. To install git-lfs please follow these [git-lfs instructions](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage).  

### Conda:

For additional cmds see the [Conda cheat-sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf).

 * Download and install either [miniconda](https://docs.conda.io/en/latest/miniconda.html#installing) or [anaconda](https://docs.anaconda.com/free/anaconda/install/index.html).
 * Create new environment (env) and install ``conda create -n <environment_name>``
 * Activate/switch to new env ``conda activate <environment_name>``
 * ``cd`` into repo dir.
 * Install ``python`` and ``pip`` ``conda install python=3.11 pip``
 * Install all required dependencies (assuming local dev work), there are two ways to do this
   * If working with tox (recommended) ``pip install -r requirements/dev.txt``.
   * If you would like to setup an environment with all requirements to run outside of tox ``pip install -r requirements/all.txt``.

### Build:

  #### with Python ecosystem:
  * ``cd`` into repo dir.
  * ``conda activate <environment_name>``
  * Build and install package in <environment_name> conda env: ``pip install .``
  * Do the same but in dev/editable mode (changes to repo will be reflected in env installation upon python kernel restart)
  >[!NOTE]
  > This is the preferred installation method for dev work. ``pip install -e .``.
  
  >[!NOTE]
  > If you didn't install dependencies from ``requirements/dev.txt``, you can install a looser constrained set of deps
  > using: ``pip install -e .[dev]``.
  
  >[!NOTE]
  > For GPU acceleration either PyTorch or JAX can be re-installed with their accelerator options.
  > For PyTorch see the [PyTorch installation docs](https://pytorch.org/get-started/locally/).
  > E.g., ``pip install --force -r requirements/pytorch.txt --index-url https://download.pytorch.org/whl/cu126``.
  > For JAX see the [JAX installation docs](https://docs.jax.dev/en/latest/installation.html#installation).
  > E.g., ``pip install --force jax["cuda12"]``. Since both are installed via ``requirements/prd.txt``, ``--force``
  > or ``--upgrade`` must  be used to re-install the accelerator versions.  ``--force`` is preferable as it will error
  > if the distribution is not available at the given url index, however ``--upgrade`` may not.

  #### with Docker (C++ version only):
  * Download & install Docker - see [Docker install docs](https://docs.docker.com/get-docker/).
  * ``cd`` into repo dir.
    * OpneCV with CUDA support: ``docker buildx build -f docker/Dockerfile.opencv --platform linux/amd64 . -t opencv_image`` 
    * flfm.exe: ``docker buildx build -f docker/Dockerfile.flfm --platform linux/amd64 . -t flfm``
  * Run container interactively: ``docker run --platform linux/amd64 -it flfm  sh``

### Usage

Follow the above [Quickstart](#quickstart) or [Build with Python ecosystem instructions](#with-python-ecosystem).

Using the command line interface (i.e., from a terminal prompt):
```term
python flfm/cli.py main flfm/tests/data/yale/light_field_image.tif flfm/tests/data/yale/measured_psf.tif reconstructed_image.tiff --normalize_psf=True --lens_radius=230 --lens_center="(1000,980)" --backend=torch
```

_NOTE: The above data files are only present when cloning the repo and not when pip installing the package.

Within a Python session or Jupyter notebook:
```python
import flfm.util
from flfm.backends import reload_backend

# The following can be pre-set in ``flfm/settings.py`` prior to import, or prior to invoking the notebook when using env vars,
# e.g., ``FLFM_BACKEND=jax jupyter notebook``. However, to dynamically change the backend the following code snippet
# can be used.
reload_backend("jax")  # or "torch"

import flfm.io
import flfm.restoration

# Read in images. NOTE: These data files are only present when cloning the repo and not when pip installing the package.
image = flfm.io.open(flfm.util.find_package_location()  / "tests" / "data" / "yale" / "light_field_image.tif")
psf = flfm.io.open(flfm.util.find_package_location()  / "tests" / "data" / "yale" / "measured_psf.tif")

# Normalize PSF.
psf_norm = psf / flfm.restoration.sum(psf)

# Compute reconstruction.
reconstruction = flfm.restoration.reconstruct(image, psf_norm)

# Clip image to view only the central lens perspective.
cropped_reconstruction = flfm.util.crop_and_apply_circle_mask(reconstruction, center=(1000, 980), radius=230)

# Save cropped reconstruction to file.
flfm.io.save("reconstructed_image.tif", cropped_reconstruction)
```

# Starting the UI app

Follow the above [Build with Python ecosystem instructions](#with-python-ecosystem). Then install the additional application
dependencies:

```terminal
pip install -r requirements/app.txt
```

Start the app from a terminal with:
```terminal
python flfm/app/main.py
```

The app should then be reachable from a browser at ``127.0.0.1:8080``.

The host IP and port number are set and can be edited in ``flfm/settings.py``. The following environmental variables can also be used.
 * ``FLFM_APP_HOST``
 * ``FLFM_APP_PORT``

For example, to change the port number before starting the app you can use:
```terminal
FLFM_APP_PORT=8000 python flfm/app/main.py
```

### From a Jupyter notebook

Run the following in a notebook cell.

```python
from flfm.app.main import dash_app
dash_app.run()
```

See [dash in jupyter](https://dash.plotly.com/dash-in-jupyter) for further options and details.

### Manually using uvicorn/gunicorn
Run either of the following from a terminal:
```terminal
FLFM_APP_WEB_API=fastapi uvicorn flfm.app.main:app --host=127.0.0.1 --port=8080
```

```terminal
FLFM_APP_WEB_API=flask gunicorn flfm.app.main:dash_server -b 127.0.0.1:8080
```

# ImageJ Plugin

A plugin for [ImageJ](https://imagej.net/ij/) is avaliable from [ssec-jhu/flfm-ij-plugin](https://github.com/ssec-jhu/flfm-ij-plugin).

# Testing
_NOTE: The following steps require ``pip install -r requirements/dev.txt``._

## Using tox

* Run tox ``tox``. This will run all of linting, security, test, docs and package building within tox virtual environments.
* To run an individual step, use ``tox -e {step}`` for example, ``tox -e test``, ``tox -e build-docs``, etc.

Typically, the CI tests run in github actions will use tox to run as above. See also [ci.yml](https://github.com/ssec-jhu/flfm/blob/main/.github/workflows/ci.yml).

## Outside of tox:

The below assume you are running steps without tox, and that all requirements are installed into a conda environment, e.g. with ``pip install -r requirements/all.txt``.

_NOTE: Tox will run these for you, this is specifically if there is a requirement to setup environment and run these outside the purview of tox._

### Linting:
Facilitates in testing typos, syntax, style, and other simple code analysis tests.
  * ``cd`` into repo dir.
  * Switch/activate correct environment: ``conda activate <environment_name>``
  * Run ``ruff .``
  * This can be automatically run (recommended for devs) every time you ``git push`` by installing the provided
    ``pre-push`` git hook available in ``./githooks``.
    Instructions are in that file - just ``cp ./githooks/pre-push .git/hooks/;chmod +x .git/hooks/pre-push``.

### Security Checks:
Facilitates in checking for security concerns using [Bandit](https://bandit.readthedocs.io/en/latest/index.html).
 * ``cd`` into repo dir.
 * ``bandit --severity-level=medium -r flfm``

### Unit Tests:
Facilitates in testing core package functionality at a modular level.
  * ``cd`` into repo dir.
  * Run all available tests: ``pytest .``
  * Run specific test: ``pytest tests/test_util.py::test_base_dummy``.

### Regression tests:
Facilitates in testing whether core data results differ during development.
  * WIP

### Smoke Tests:
Facilitates in testing at the application and infrastructure level.
  * WIP

### Build Docs:
Facilitates in building, testing & viewing the docs.
 * ``cd`` into repo dir.
 * ``pip install -r requirements/docs.txt``
 * ``cd docs``
 * ``make clean``
 * ``make html``
 * To view the docs in your default browser run ``open docs/_build/html/index.html``.
