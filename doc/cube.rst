Cube
====

Introduction
------------

The “cube” in the context of ISGRI is a pre-processed data set,
usually corresponding to a single science window, which is useful by
avoiding the repeated execution of the early steps of the data
processing pipeline.

The pre-processing consists of binning the events by reconstructed
energy and per detector pixel, resulting in a three-dimensional array
of counts and an associated array of efficiencies. Currently there are
two different types of cubes in use:

IDL cube
  This is the first form of cube used, all the processing, starting
  from the data as distributed by ISDC, is done in IDL. There is one
  FITS file per science window, containing four FITS images:
  
  +-------+----------------+-----------------+--------------------------+
  | Index | Extension      | Dimension       | Description              |
  +=======+================+=================+==========================+
  |     0 | Primary        | 128 × 128 × 256 | counts                   |
  +-------+----------------+-----------------+--------------------------+
  |     1 | Efficiency map | 128 × 128       | pixel efficiency         |
  +-------+----------------+-----------------+--------------------------+
  |     2 | LT Cube        | 128 × 128 × 256 | low threshold efficiency |
  +-------+----------------+-----------------+--------------------------+
  |     3 | Valid map      | 128 × 128       | pixel validity           |
  +-------+----------------+-----------------+--------------------------+

  The MDU efficiencies an dead time fractions are stored in the header
  of the Primary HDU, along with many keywords containing housekeeping
  data. They can be used to calculate the total efficiency as a
  function of pixel and energy bin.

  The energy binning uses 256 bins with a bin width approximately
  proportional to :math:`\sqrt{E}`, rounded to the nearest 0.5 keV.

OSA cube
  Built using the standard OSA tools, these consist of two FITS files
  per science window: one for the counts and one for the efficiencies.

  They use a different energy binning: The bin boundaries are a
  superset of the ones used by the IDL cubes and a set of “round”
  energy values, yielding 336 bins in total.
