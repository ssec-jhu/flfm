Welcome to flfm's documentation!
==========================================

Fourier Light Field Microscopy (FLFM) is a technique for scan-free volumetric
imaging. FLFM utilizes an array of lenses to simultaneously acquire several
images from different viewpoints. These images are then post-processed to
generate a 3D volume using a Richardson-Lucy-based deconvolution algorithm. This
enables volumetric imaging at the exposure time of the camera, a speed unmatched
by conventional volumetric scanning. This fast imaging is particularly useful
for samples with transient signals, where the time spent scanning will miss
relevant information. For example, FLFM is ideal for capturing the transient
activity of point-like neurons in 3D.

Check out the :doc:`usage` section for further information, including
how to :ref:`install <installation>` the project.

.. note::

   This project is under active development.

Contents
--------

.. toctree::
   :maxdepth: 2

   usage
