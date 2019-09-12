# numPy-Sculpture
(Manipulating Numpy 3D arrays)
Complete this solution program so that all Doctests work properly and it solves the problems described. Some empty functions are included to help guide you. You may add or modify their parameters and may create additional functions as well.

The Scenario:
Suppose we are trying to carve a lasting monument from a "marble" block, and its intended final 3D shape is given with a 3D array, containing just 1s and 0s to indicate where it will be solid vs empty (removed).

We also have a large rectangular prism of floats in a 3D array. These numbers represent local densities within a block of marble, perhaps having been measured externally using some process like NMR (nuclear magnetic resonance) from 3 axes.

Before we 'carve' or 'sculpt' a shape from one of these expensive marble blocks, we want to determine which is the best orientation (rotation) to use.

Which block & orientation results in the highest average density, AFTER being 'carved'. I gave you just two block data files, but write your program so that it could read and evaluate MANY such files -- break your habits of copy-and-paste programming now! :-) Your output can display the answer by the block's filename and
For each possible orientation, determine if the sculpture would be stable & balanced. So, we need to compute where the center of mass would be and make sure it's within the boundaries of its base.
These Khan Academy pages on Center of Mass are helpful to understand problem 2:

• https://www.khanacademy.org/science/physics/linear-momentum/center-of-mass/v/center-of-mass-equation

• https://www.khanacademy.org/science/physics/linear-momentum/center-of-mass/a/what-is-center-of-mass (the "Topple Limit" section in this page helps explain problem 2 above. But you are not required to compute tipping angles)

Note though the SciPy package already has a function that can compute the center of mass for any Numpy array, including multi-dimensional ones like we have here.

https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.measurements.center_of_mass.html

