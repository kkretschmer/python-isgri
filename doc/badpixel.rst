Bad Pixels
==========

Introduction
------------

ISGRI has a *noisy pixel handling system* which can automatically
detect and temporarily disable pixels which cause excessively many
events. The standard data processing performs further
filtering. Still, the resulting data sets contain some pixels that
behave in an anomalous way that can be identified.

Once identified these outlier pixels can either be excluded from the
analysis or their count rate can be estimated from other, related
pixels.

Identifying anomalous pixels
----------------------------

The method to identify outlier pixels as implemented in :ref:`bgcube`
looks at the statistical distribution of pixel counts compared to a
reference count rate image. when applied to empty field observations,
this identifies anomalously *dark* and *hot* pixels.

.. figure:: images/badpixel_counts_scatter.png
   :alt: scatter plot of actual vs. expected counts

   Distribution of expected and observed counts.

   The dashed lines show observed = expected ± {1, 2, 3} σ. The
   bullets show the inverse cumulative Poisson distribution/survival
   functions corresponding to the probability of a deviation of 4 σ.

.. figure:: images/badpixel_logcdf_sort.png
   :alt: sorted log(cdf) per pixel for a set of SCWs

   Pixel probability distributions

   This plot shows the sorted logarithms of the cumulative Poisson
   distribution function (blue) and the survival function (red) for a
   set of science windows. In the case of purely statistical scatter,
   a straight line starting at :math:`-\log 128^2 \approx -9.7` would
   be expected. Pixels lying significantly below are outliers.

   The dotted lines show the raw distribution functions while the
   solid lines are normalised to the 64\ :sup:`th` pixel to eliminate
   the probability decrease introduced by a systematic mismatch of the
   background intensity distribution.
