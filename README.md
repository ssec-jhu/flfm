# SSEC-JHU flfm

[![CI](https://github.com/ssec-jhu/flfm/actions/workflows/ci.yml/badge.svg)](https://github.com/ssec-jhu/flfm/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/flfm/badge/?version=latest)](https://flfm.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/ssec-jhu/flfm/graph/badge.svg?token=l3ND0EA9dE)](https://codecov.io/gh/ssec-jhu/flfm)
[![Security](https://github.com/ssec-jhu/flfm/actions/workflows/security.yml/badge.svg)](https://github.com/ssec-jhu/flfm/actions/workflows/security.yml)
<!---[![DOI](https://zenodo.org/badge/<insert_ID_number>.svg)](https://zenodo.org/badge/latestdoi/<insert_ID_number>) --->


![SSEC-JHU Logo](docs/_static/SSEC_logo_horiz_blue_1152x263.png)

# Fourier Light Field Microscopy (FLFM)

Fourier Light Field Microscopy (FLFM) is a technique for scan-free volumetric imaging.
FLFM utilizes an array of lenses to simultaneously acquire several images from different viewpoints. These images are
then post-processed to generate a 3D volume using a Richardson-Lucy-based deconvolution algorithm. This enables
volumetric imaging at the exposure time of the camera, a speed unmatched by conventional volumetric scanning.
This fast imaging is particularly useful for samples with transient signals, where the time spent scanning will miss
relevant information. For example, FLFM is ideal for capturing the transient activity of point-like neurons in 3D.

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
  > For Nvidia GPU utilization install ``jax["cuda12"]``, e.g., ``pip install jax["cuda12"]``.
  > See the [JAX installation docs](https://docs.jax.dev/en/latest/installation.html#installation) for further details
  > on supported hardware accelerator architectures and operating systems.

  #### with Docker (C++ version only):
  * Download & install Docker - see [Docker install docs](https://docs.docker.com/get-docker/).
  * ``cd`` into repo dir.
    * OpneCV with CUDA support: ``docker buildx build -f docker/Dockerfile.opencv --platform linux/amd64 . -t opencv_image`` 
    * flfm.exe: ``docker buildx build -f docker/Dockerfile.flfm --platform linux/amd64 . -t flfm``
  * Run container interactively: ``docker run --platform linux/amd64 -it flfm  sh``

### Usage

Follow the above [Build with Python ecosystem instructions](#with-python-ecosystem).

Using the command line interface (i.e., from a terminal prompt):
```term
python flfm/cli.py data/yale/light_field_image.tif data/yale/measured_psf.tif reconstructed_image.tiff --lens_radius=230 --lens_center="(1000,980)"
```

Within a Python session or Jupyter notebook:
```python
import jax.numpy as jnp

import flfm.io
import flfm.restoration
import flfm.util

# Read in images.
image = flfm.io.open("/Users/jamienoss/repos/adkins-flfm/flfm/data/yale/light_field_image.tif")
psf = flfm.io.open("/Users/jamienoss/repos/adkins-flfm/flfm/data/yale/measured_psf.tif")

# Normalize PSF.
psf_norm = psf / jnp.sum(psf, axis=(1,2), keepdims=True)

# Compute reconstruction.
reconstruction = flfm.restoration.richardson_lucy(image, psf_norm)

# Clip image to view only central lens perspective.
cropped_reconstruction = flfm.util.crop_and_apply_circle_mask(reconstruction, center=(1000, 980), radius=230)

# Save cropped reconstruction to file.
flfm.io.save("reconstructed_image.tif", cropped_reconstruction)
```


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
