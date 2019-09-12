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

Steps to Get Started (mostly shown in class):
Do this assignment individually (no teams, so you won't have any problems with git merge conflicts.)
Create one and only one fork of this repository.
In the GitHub website, immediately go to your new repository,
click "Settings" tab
click "Collaborators & Teams"
click the X to REMOVE the "All_Students" team from having access to your repository. Warning! We'll deduct 20% from your score if you don't.
Return to the <Code> tab in YOUR new repository in GitHub.
Click "Clone or Download" to get the link you need for PyCharm.
As shown in the video and in class session 4, create a new PyCharm project from your repository.
Now in PyCharm, write the code needed to make the functions and tests work properly.
COMMIT AND PUSH revisions frequently as you work on it, until complete.
INCLUDE the output text files in your GitHub repository to make checking your calculations easier.
BONUS CREDIT!
You can get extra credit on this assignment by doing either of these things:

Writing enough Doctests in your program so that the Doctests With Coverage report in PyCharm shows at least 95% for the numpy_marble_solution.py file.
Configuring TravisCI to work on your repository and showing proof, such taking a screenshot of your build history and committing that screen shot image into your GitHub repository where we can see it. Remember you get TravisCI free as part of the GitHub "StudentPack"
