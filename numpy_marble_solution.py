"""
590PR Spring 2019. Instructor: John Weible  jweible@illinois.edu
Assignment on Numpy: "High-Tech Sculptures"

See assignment instructions in the README.md document.
"""

import numpy as np
np.set_printoptions(threshold=np.inf)
from scipy.ndimage import center_of_mass
from typing import List
import math


def get_orientations_possible(block: np.ndarray) -> List[List[dict]]:
    """Given a 3D numpy array, look at its shape to determine how many ways it
    can be rotated in each axis to end up with a (theoretically) different array
    that still has the SAME shape.

    if all three dimensions are different sizes, then we have 3 more
    orientations, excluding the original, which are all 180-degree rotations.

    if just two dimensions match size, we have 7 plus original. 90-degree
    rotations are around the unique-length axis.

    if all three dimensions match (a cube), then we have 23 plus original.

    :param block: a numpy array of 3 dimensions.
    :return: a list of the ways we can rotate the block. Each is a list of dicts containing parameters for rot90()

    >>> a = np.arange(64, dtype=int).reshape(4, 4, 4)  # a cube
    >>> rotations = get_orientations_possible(a)
    >>> len(rotations)
    23
    >>> rotations  # doctest: +ELLIPSIS
    [[{'k': 1, 'axes': (0, 1)}], ... [{'k': 3, 'axes': (1, 2)}, {'k': 3, 'axes': (0, 2)}]]
    >>> a = a.reshape(2, 4, 8)
    >>> len(get_orientations_possible(a))
    3
    >>> a = a.reshape(16, 2, 2)
    >>> len(get_orientations_possible(a))
    7
    >>> get_orientations_possible(np.array([[1, 2], [3, 4]]))
    Traceback (most recent call last):
    ValueError: array parameter block must have exactly 3 dimensions.
    >>> marble_block_1 = np.load(file='data/marble_block_1.npy')
    >>> len(get_orientations_possible(marble_block_1))
    7
    """

    if len(block.shape) != 3:
        raise ValueError('array parameter block must have exactly 3 dimensions.')

    # Create list of the 23 possible 90-degree rotation combinations -- params to call rot90():
    poss = [
        [{'k': 1, 'axes': (0, 1)}],  # 1-axis rotations:
        [{'k': 2, 'axes': (0, 1)}],
        [{'k': 3, 'axes': (0, 1)}],
        [{'k': 1, 'axes': (0, 2)}],
        [{'k': 2, 'axes': (0, 2)}],
        [{'k': 3, 'axes': (0, 2)}],
        [{'k': 1, 'axes': (1, 2)}],
        [{'k': 2, 'axes': (1, 2)}],
        [{'k': 3, 'axes': (1, 2)}],
        [{'k': 1, 'axes': (0, 1)}, {'k': 1, 'axes': (0, 2)}],  # 2-axis rotations:
        [{'k': 1, 'axes': (0, 1)}, {'k': 2, 'axes': (0, 2)}],
        [{'k': 1, 'axes': (0, 1)}, {'k': 3, 'axes': (0, 2)}],
        [{'k': 2, 'axes': (0, 1)}, {'k': 1, 'axes': (0, 2)}],
        [{'k': 2, 'axes': (0, 1)}, {'k': 3, 'axes': (0, 2)}],
        [{'k': 3, 'axes': (0, 1)}, {'k': 1, 'axes': (0, 2)}],
        [{'k': 3, 'axes': (0, 1)}, {'k': 2, 'axes': (0, 2)}],
        [{'k': 3, 'axes': (0, 1)}, {'k': 3, 'axes': (0, 2)}],
        [{'k': 1, 'axes': (1, 2)}, {'k': 1, 'axes': (0, 2)}],
        [{'k': 1, 'axes': (1, 2)}, {'k': 2, 'axes': (0, 2)}],
        [{'k': 1, 'axes': (1, 2)}, {'k': 3, 'axes': (0, 2)}],
        [{'k': 3, 'axes': (1, 2)}, {'k': 1, 'axes': (0, 2)}],
        [{'k': 3, 'axes': (1, 2)}, {'k': 2, 'axes': (0, 2)}],
        [{'k': 3, 'axes': (1, 2)}, {'k': 3, 'axes': (0, 2)}],
        ]

    # consider the 3-tuple shape of axes numbered 0, 1, 2 to represent (height, width, depth)
    (height, width, depth) = block.shape

    if height == width == depth:
        return poss  # return all possibilities, it's a cube

    elif height != width != depth:
        poss_1 = [poss[1], poss[4], poss[7]]
        return poss_1

    elif height != width == depth:
        poss_2 = [poss[1], poss[4], poss[6], poss[7], poss[8], poss[18], poss[21]]
        return poss_2

    elif height == width != depth:
        poss_3 = [poss[0], poss[1], poss[2], poss[4], poss[7], poss[10], poss[15]]
        return poss_3

    elif height == depth != width:
        poss_4 = [poss[1], poss[3], poss[4], poss[5], poss[7], poss[12], poss[13]]
        return poss_4


    # TODO: Complete this function for the other situations...
    # Hint, the results will be parts of the 23-item list above, read the Docstring!


def carve_sculpture_from_density_block(shape: np.ndarray, block: np.ndarray) -> np.ndarray:
    """Given two numpy array with same shape, multiply the two numbers in the same position
    and return a new numpy array.

    :param shape: The target shape which is a binary 3d numpy array.
    :param block: Marble block material which is a 3d numpy array.
    :return: A new numpy array whose each position is the multiplication of the shape and block at same position.

    >>> a = np.zeros((4, 4))
    >>> a[0][1] = 1
    >>> b = np.arange(16).reshape(4, 4)
    >>> result = carve_sculpture_from_density_block(a, b)
    >>> result[0][1]
    1.0
    """

    return np.multiply(shape, block)

    # TODO: write the code for this function, which could be as short as one line of code!
    # TODO: Add Doctests for good coverage.


def are_rotations_unique(list_of_rotations: List[List[dict]], verbose=False) -> bool:
    """Given a list of list of 3D rotation combinations suitable for using with np.rot90()
    and as returned from the get_orientations_possible() function, determine whether any
    of the rotations are equivalent, and discard the duplicates.

    The purpose is to detect situations where a combination of rotations would produce either
    the original unmodified array or the same orientation as any previous one in the list.

    :param list_of_rotations: a list, such as returned by get_orientations_possible()
    :param verbose: if True, will print details to console, otherwise silent.
    :return: True, if all listed rotation combinations produce distinct orientations.

    >>> x = [[{'k': 4, 'axes': (0, 1)}]]  # 4x90 degrees is a full rotation
    >>> are_rotations_unique(x)
    False
    >>> x = [[{'k': 2, 'axes': (0, 1)}, {'k': 2, 'axes': (0, 1)}]]  # also a full rotation
    >>> are_rotations_unique(x)
    False
    >>> y1 = [[{'k': 3, 'axes': (1, 2)}], [{'k': 1, 'axes': (0, 1)}, {'k': 1, 'axes': (2, 0)}]]
    >>> are_rotations_unique(y1)
    True
    >>> y2 = y1 + [[{'k': 1, 'axes': (1, 2)}, {'k': 3, 'axes': (1, 0)}]]  # equiv. to earlier
    >>> are_rotations_unique(y2, verbose=True)
    combination #1: [{'k': 3, 'axes': (1, 2)}] ok.
    combination #2: [{'k': 1, 'axes': (0, 1)}, {'k': 1, 'axes': (2, 0)}] ok.
    combination #3: [{'k': 1, 'axes': (1, 2)}, {'k': 3, 'axes': (1, 0)}] not unique.
    it results in the same array as combination 2
    False
    """
    # create a small cube to try all the input rotations. It has unique values so that
    #  no distinct rotations could create an equivalent array by accident.
    cube = np.arange(0, 27).reshape((3, 3, 3))

    # Note: In the code below, the arrays must be appended to the orientations_seen
    #  list in string form, because Numpy would otherwise misunderstand the intention
    #  of the if ... in orientations_seen expression.

    orientations_seen = [cube.tostring()]  # record the original

    count = 0
    for combo in list_of_rotations:
        count += 1
        if verbose:
            print('combination #{}: {}'.format(count, combo), end='')

        r = cube  # start with a view of cube unmodified for comparison
        for r90 in combo:  # apply all the rotations given in this combination
            r = np.rot90(r, k=r90['k'], axes=r90['axes'])
        if r.tostring() in orientations_seen:
            if verbose:
                print(' not unique.')
                if r.tostring() == cube.tostring():
                    print('it results in the original 3d array.')
                else:
                    print('it results in the same array as combination',
                          orientations_seen.index(r.tostring()))
            return False
        else:
            if verbose:
                print(' ok.')
        orientations_seen.append(r.tostring())
    return True


def after_rotations_results(block: np.ndarray) -> List[np.ndarray]:
    """
    Given a 3d numpy array, get possible rotation methods for it.
    And use each rotation to change the array then store the result
    into a list.
    :param block: A 3d numpy array for the material marble block.
    :return: A list of all the rotations result.
    """
    possibles = get_orientations_possible(block)
    if are_rotations_unique(possibles):
        result = [block]
        for i in possibles:
            if len(i) == 2:
                result_1 = np.rot90(block, k=i[0]['k'], axes=i[0]['axes'])
                result_2 = np.rot90(result_1, k=i[1]['k'], axes=i[1]['axes'])
                result.append(result_2)
            else:
                result_3 = np.rot90(block, k=i[0]['k'], axes=i[0]['axes'])
                result.append(result_3)
    return result


def max_average_density(carved_result_list: List[np.ndarray]) -> List:
    """Give a list of 3d numpy arrays which represent the after carved
    sculptures, calculate each one's average density and return the max
    one and its index in a list.

    :param carved_result_list:a list of 3d numpy arrays which represent
    the after carved sculptures.
    :return: a list of the index of the max average density and the value
    of the max average density.

    """
    average_density_list = []
    for carved_result in carved_result_list:
        nan_carved_result = np.where(carved_result == 0, np.nan, carved_result)
        average_density_list.append(np.nanmean(nan_carved_result))
    return [average_density_list.index(max(average_density_list)), max(average_density_list)]


def get_center_of_mass_projection(carved_result_list: List[np.ndarray]) -> List[List]:
    """Given a list of carved_result numpy array, calculate the carved_result of
    each of it.

    :param carved_result_list:a list of carved_result numpy array
    :return: a list of center_of_mass_projection on the base
    """
    center_of_mass_projection = []
    for i in carved_result_list:
        cm = list(center_of_mass(i))
        cm[2] = 0
        center_of_mass_projection.append(cm)
    return center_of_mass_projection


def get_base_array(carved_result_list: List[np.ndarray]) -> List[np.ndarray]:
    """Give a list of carved_result 3d numpy array, get the base 2d array.
    :param carved_result_list: a list of carved_result 3d numpy array
    :return:a list of the base 2d array
    """
    base_array = []
    for i in carved_result_list:
        base_array.append(i[-1])
    return base_array


def array_exist_nonzero(part_base_array: np.ndarray) ->bool:
    """
    Given a 2d numpy array, judge whether it has nonzero value or not.
    :param part_base_array: A 2d numpy array
    :return: Boolean value

    >>> a = np.arange(16).reshape(4, 4)
    >>> array_exist_nonzero(a)
    True
    >>> b = np.zeros((4,4))
    >>> array_exist_nonzero(b)
    False

    """

    jud = np.nonzero(part_base_array)
    if len(jud[0]) > 0:
        return True
    else:
        return False


def judge_stable_or_not(center_of_mass_projection: List[List], base_array: List[np.ndarray]) -> List:
    """
    Given a list of mass of center projection on the base and a list of the 2d numpy array of base, cut the base into
    4 pieces to see if each piece have nonzero value, if true, it is stable, otherwise it is instable.
    :param center_of_mass_projection: A list of mass of center projection on the base
    :param base_array: a list of the 2d numpy array of base
    :return: a list of stable judgement "stable" or "instable"
    """
    judge_result = []
    i = 0
    while i < 8:
        left_top = base_array[i][:math.floor(center_of_mass_projection[i][0])+1, :math.floor(center_of_mass_projection[i][1])+1]
        right_top = base_array[i][:math.floor(center_of_mass_projection[i][0])+1, math.ceil(center_of_mass_projection[i][1]):]
        left_bot = base_array[i][math.ceil(center_of_mass_projection[i][0]):, :math.floor(center_of_mass_projection[i][1])+1]
        right_bot = base_array[i][math.ceil(center_of_mass_projection[i][0]):, math.ceil(center_of_mass_projection[i][1]):]
        if array_exist_nonzero(left_top)&array_exist_nonzero(right_top)&array_exist_nonzero(left_bot)&array_exist_nonzero(right_bot):
            judge_result.append("stable")
        else:
            judge_result.append("instable")
        i += 1
    return judge_result



marble_block_1 = np.load(file='data/marble_block_1.npy').astype("float64")
marble_block_2 = np.load(file='data/marble_block_2.npy').astype("float64")
shape_1 = np.load(file='data/shape_1.npy').astype("float64")

after_carved_result_1 = []
for rot_result_1 in after_rotations_results(marble_block_1):
    after_carved_result_1.append(carve_sculpture_from_density_block(shape_1, rot_result_1))

after_carved_result_2 = []
for rot_result_2 in after_rotations_results(marble_block_2):
    after_carved_result_2.append(carve_sculpture_from_density_block(shape_1, rot_result_2))


rotation_method = {0: "no rotation", 1: "{'k': 2, 'axes': (0, 1)}", 2: "{'k': 2, 'axes': (0, 2)}",
                   3: "{'k': 1, 'axes': (1, 2)}", 4: "{'k': 2, 'axes': (1, 2)}", 5:"{'k': 3, 'axes': (1, 2)}",
                   6: "{'k': 1, 'axes': (1, 2)}, {'k': 2, 'axes': (0, 2)}",
                   7: "{'k': 3, 'axes': (1, 2)}, {'k': 2, 'axes': (0, 2)}"}

if max_average_density(after_carved_result_1)[1] > max_average_density(after_carved_result_2)[1]:
    print("marble block 1 and rotation:" + str(rotation_method[max_average_density(after_carved_result_1)[0]])
          + " have highest average density:" + str(max_average_density(after_carved_result_1)[1]))
else:
    print("marble block 2 and rotation:" + str(rotation_method[max_average_density(after_carved_result_2)[0]])
          + " have highest average density:" + str(max_average_density(after_carved_result_2)[1]))

m = 0
while m < len(after_carved_result_1):

    print("marble_block_1, rotation: "+rotation_method[m]+",  " +
          judge_stable_or_not(get_center_of_mass_projection(after_carved_result_1), get_base_array(after_carved_result_1))[m])
    m += 1
n = 0
while n < len(after_carved_result_1):
    print("marble_block_2, rotation: "+rotation_method[n]+",  " +
          judge_stable_or_not(get_center_of_mass_projection(after_carved_result_2), get_base_array(after_carved_result_2))[n])
    n += 1



if __name__ == '__main__':

    # Load the sample "blocks" of variable-density marble:
    marble_block_1 = np.load(file='data/marble_block_1.npy')
    marble_block_2 = np.load(file='data/marble_block_2.npy')

    # Load the array describing the 3D shape of the sculpture we want to carve from marble:
    shape_1 = np.load(file='data/shape_1.npy')
