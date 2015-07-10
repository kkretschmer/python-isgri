# ISGRI with Python

Copyright (C) 2015  [Karsten Kretschmer](https://www.researchgate.net/profile/Karsten_Kretschmer2) <kkretsch@apc.univ-paris7.fr>

These are tools for working with [INTEGRAL](http://www.cosmos.esa.int/web/integral)/ISGRI data in Python.

* **cube.py:** read and work with “cubes”, i.e. preprocessed 256×128×128 detector plane images
* **bgcube.py:** read, create and write “background cubes”, i.e. cubes intended to model the instrumental background rate
* **build_background:** build background cubes
* **isotropic_response.py:** read the response of ISGRI to isotropic incoming radiation into an numpy array
* **srcsig.py:** collect information about the significance of source detection per science window
