Usage
=====

.. _installation:

Installation
------------

This package can be installed from PyPI using pip:

.. code-block:: console

   $ pip install flfm

To work with the development version, clone the repository and install it in editable mode:

.. code-block:: console

   $ git clone https://github.com/ssec-jhu/flfm.git
   $ cd flfm
   $ pip install -e .[dev]

For GPU acceleration, you can reinstall PyTorch or JAX with the appropriate options. For example:

.. code-block:: console

   # For PyTorch with CUDA
   $ pip install --force -r requirements/pytorch.txt --index-url https://download.pytorch.org/whl/cu126

   # For JAX with CUDA
   $ pip install --force jax["cuda12"]

Quick Start
-----------

Here is a quick example of how to use `flfm` in a Python script:

.. code-block:: python

   import flfm.util
   from flfm.backend import reload_backend

   # To dynamically change the backend the following code snippet can be used.
   reload_backend("jax")  # or "torch"

   import flfm.io
   import flfm.restoration

   # Read in images.
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

Command-Line Interface
----------------------

`flfm` provides a command-line interface for running reconstructions. Here is an example:

.. code-block:: console

   $ flfm.cli main path/to/image.tif path/to/psf.tif output_image.tiff --normalize_psf=True --lens_radius=230 --lens_center="(1000,980)" --backend=torch

GUI
---

`flfm` also includes a web-based user interface for interactive reconstructions. To use it, first install the application dependencies:

.. code-block:: console

    $ pip install -r requirements/app.txt

Then, start the application:

.. code-block:: console

    $ python flfm/cli.py app --host=127.0.0.1 --port=8080

The application will be available at http://127.0.0.1:8080.

.. note:: See :doc:`gui` for further details.

ImageJ Plugin
-------------

A plugin for `ImageJ <https://imagej.net/ij/>`_ is available from `ssec-jhu/flfm-ij-plugin <https://github.com/ssec-jhu/flfm-ij-plugin>`_.
