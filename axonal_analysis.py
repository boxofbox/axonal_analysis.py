__author__ = "Johnathan G. Lyon"
__copyright__ = "Copyright 2016"
__credits__ = ["Johnathan G. Lyon", "Nassir Mokarram"]
__license__ = "*FLOSS" # NOTE: may be insufficient given imported libraries
__version__ = "1.0.0"
__maintainer__ = "Johnathan G. Lyon"
__email__ = "jgl@gatech.edu"
__status__ = "Production"


"""
This file is used for analyzing a folder of rgb .tif images for various parameters related
to axonal segments of peripheral nerve. 

The images will be interpreted thusly: 
    GREEN: axons 
    RED: myelin
    BLUE: nuclei
    
The two primary forms of measurement are: 
    1) The overall coverage of a given color above a specified threshold
    2) The thickness of myelin around each axon
    
Axons are first detected as particles and then using the individual centers, a vector is 
swept from that center to measure any myelination with a specified proximity of the axon's
edge. Using these vectors, axons are grouped as either myelinated or unmyelinated based on 
a specified percentage of valid vectors per axon. For myelinated axons, myelin thickness 
is calculated and reported as an average per axon.

"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import ndimage, misc
from matplotlib.patches import Circle
from scipy.spatial import distance

# PARAMETERS
green_threshold = 5  # minimum radius of green particles (axons)
blue_threshold = 4   # minimum radius of blue particles (nuclei)
red_threshold = 4  # minimum radius of red particles (myelin)
area_calc_border_tolerance = 0  # border proximity limit for area calculations
intensity_threshold = 100  # minimum pixel value
green_intensity_threshold = 30  # green (axonal) pixel value minimum threshold
blue_intensity_threshold = 50  # blue (nuclear) pixel value minimum threshold
red_intensity_threshold = 40  # red (myelin) pixel value minimum threshold
n_angles = 360  # number of angles to test around each axon for detecting red thickness
um_per_pixel = 0.106  # pixel to micron conversion factor
red_to_green_maximum_dist = 25  # maximum distance between edge of axon and onset of myelin for valid myelinated axon
red_circle_percent_minimum = 0.40   # minimum percent valid vectors around a axon to not reject
                                    # (between 0.0 and 1.0)
maximum_myelin_radius = 75      # maximum extent of myelin on a vector or else invalid
                                # also rejects an axon if border proximity to particle center is below this value

um2_per_pixel_area = um_per_pixel ** 2

# file locations
in_folder = "input"  # folder containing input .tif files
out_folder = "output"  # folder to output process & measurement images, and parameter and data files to (must exist)
parameter_filename = "parameters.txt"  # filename for parameter file
data_filename = "out.tsv"  # filename for tab-separated data file


def threshold(img_array, thresh):
    # helper function to threshold img_array into binary based on a threshold
    return np.where(img_array > thresh, 1, 0)


def count_factors(img_array):
    # helper function to make dictionary of counts for each unique value in img_array
    output = dict()
    for factor in img_array.flatten():
        output[factor] = output.get(factor, 0) + 1
    return output


def sum_dict(a, exclude=None):
    # helper function to count values of dictionary a with optional list of excluded keys
    # 0 is a default excluded key
    if exclude is None or len(exclude) == 0:
        exclude = [0]
    output = 0
    for k, value in a.iteritems():
        if k in exclude:
            continue
        output += value
    return output

# build input file list
img_files = []
for fil in os.listdir(in_folder):
    if fil[-4:] == ".tif":
        img_files.append(fil)

data_out_file = open(out_folder + "/" + data_filename, 'w')
parameter_out_file = open(out_folder + "/" + parameter_filename, 'w')

# data file header
header_str = "filename\ttotal axonal area (um^2)\t# of axons\taverage axon area (um^2)"
header_str += "\ttotal nuclei area (um^2)\t# of nuclei\taverage nuclei area (um^2)\ttotal myelinated area (um^2)"
header_str += "\taxon:myelin ratio\taxons measured\tmean myelin width\tsd myelin width\t# rejected"
header_str += "\tpercent unmyelinated\taxonal density\tnuclear density\ttotal tissue area\n"
data_out_file.write(header_str)

img_count = 1
total_img = len(img_files)
for img_file in img_files:
    print("Processing: " + str(img_count) + "/" + str(total_img) + "\t" + img_file)  # progress to stdout
    img_count += 1
    img = misc.imread(in_folder + "/" + img_file)
    temp_img = img.copy()

    # calculate borders for area calculations
    x_min = area_calc_border_tolerance
    y_min = area_calc_border_tolerance
    x_max = img.shape[0]-area_calc_border_tolerance
    y_max = img.shape[1]-area_calc_border_tolerance

    # GREEN (axons) AREA CALCULATION
    green_ch = temp_img[:, :, 1]
    green_threshed = threshold(green_ch, green_intensity_threshold)
    green_eroded = ndimage.binary_erosion(green_threshed, iterations=green_threshold)
    green_dilated = ndimage.binary_dilation(green_eroded, iterations=green_threshold)
    green_labels, n_green = ndimage.label(green_dilated)
    green_centers = ndimage.center_of_mass(green_dilated, green_labels, range(1, n_green+1))
    green_masked = green_threshed * green_labels

    # check if within borders
    green_centers_include = []
    green_centers_exclude = set()
    green_centers_exclude.update(count_factors(green_masked[0:x_min, :]).keys())
    green_centers_exclude.update(count_factors(green_masked[x_max:img.shape[0], :]).keys())
    green_centers_exclude.update(count_factors(green_masked[x_min:x_max, 0:y_min]).keys())
    green_centers_exclude.update(count_factors(green_masked[x_min:x_max, y_max:img.shape[1]]).keys())
    for c in green_centers:
        x, y = round(c[0]), round(c[1])
        if green_labels[x, y] not in green_centers_exclude:
            green_centers_include.append((x, y))
    green_factors = count_factors(green_masked)
    green_total_valid_area = sum_dict(green_factors, green_centers_exclude)
    n_green = len(green_centers_include)
    # remove axons within border proximity limits
    green_masked_bordered = green_masked.copy()
    for ind, val in np.ndenumerate(green_masked):
        if val in green_centers_exclude:  # skip 0's
            green_masked_bordered[ind] = 0
    green_binary = threshold(green_masked_bordered, 0)

    # BLUE (nuclei) AREA CALCULATION
    blue_ch = temp_img[:, :, 2]
    blue_threshed = threshold(blue_ch, blue_intensity_threshold)
    blue_eroded = ndimage.binary_erosion(blue_threshed, iterations=blue_threshold)
    blue_dilated = ndimage.binary_dilation(blue_eroded, iterations=blue_threshold)
    blue_labels, n_blue = ndimage.label(blue_dilated)
    blue_centers = ndimage.center_of_mass(blue_dilated, blue_labels, range(1, n_blue+1))
    blue_masked = blue_threshed * blue_labels

    # check if within borders
    blue_centers_include = []
    blue_centers_exclude = set()
    blue_centers_exclude.update(count_factors(blue_masked[0:x_min, :]).keys())
    blue_centers_exclude.update(count_factors(blue_masked[x_max:img.shape[0], :]).keys())
    blue_centers_exclude.update(count_factors(blue_masked[x_min:x_max, 0:y_min]).keys())
    blue_centers_exclude.update(count_factors(blue_masked[x_min:x_max, y_max:img.shape[1]]).keys())
    for c in blue_centers:
        x, y = round(c[0]), round(c[1])
        if blue_labels[x, y] not in blue_centers_exclude:
            blue_centers_include.append((x, y))
    blue_factors = count_factors(blue_masked)
    blue_total_valid_area = sum_dict(blue_factors, blue_centers_exclude)
    n_blue = len(blue_centers_include)
    # remove nuclei within border proximity limits
    blue_masked_bordered = blue_masked.copy()
    for ind, val in np.ndenumerate(blue_masked):
        if val in blue_centers_exclude:  # skip 0's
            blue_masked_bordered[ind] = 0
    blue_binary = threshold(blue_masked_bordered, 0)

    # RED (myelin) AREA CALCULATION
    red_ch = temp_img[:, :, 0]
    red_threshed = threshold(red_ch, red_intensity_threshold)
    red_eroded = ndimage.binary_erosion(red_threshed, iterations=red_threshold)
    red_dilated = ndimage.binary_dilation(red_eroded, iterations=red_threshold)
    red_labels, n_red_all = ndimage.label(red_dilated)
    red_centers = ndimage.center_of_mass(red_dilated, red_labels, range(1, n_red_all + 1))
    red_masked = red_threshed * red_labels

    # check if within borders
    red_centers_include = []
    red_centers_exclude = set()
    red_centers_exclude.update(count_factors(red_masked[0:x_min, :]).keys())
    red_centers_exclude.update(count_factors(red_masked[x_max:img.shape[0], :]).keys())
    red_centers_exclude.update(count_factors(red_masked[x_min:x_max, 0:y_min]).keys())
    red_centers_exclude.update(count_factors(red_masked[x_min:x_max, y_max:img.shape[1]]).keys())
    for c in red_centers:
        x, y = round(c[0]), round(c[1])
        if red_labels[x, y] not in red_centers_exclude:
            red_centers_include.append((x, y))
    red_factors = count_factors(red_masked)
    red_total_valid_area = sum_dict(red_factors, red_centers_exclude)
    n_red = len(red_centers_include)
    # remove myelin segments within border proximity limits
    red_masked_bordered = red_masked.copy()
    for ind, val in np.ndenumerate(red_masked):
        if val in red_centers_exclude:  # skip 0's
            red_masked_bordered[ind] = 0
    red_binary = threshold(red_masked_bordered, 0)

    # MYELINATE AXON ANALYSIS
    # array for storing myelin & axon information as it is used
    used = img.copy()
    used[:, :, :] = 0
    # array for storing rejected axons (due to lack of valid myelination)
    rejected = img.copy()
    rejected[:, :, :] = 0
    # array for storing myelin & axonal pixels that have yet to be analysed
    unused = img.copy()
    unused[:, :, 0] = red_binary * 255.0
    unused[:, :, 1] = green_binary * 255.0
    unused[:, :, 2] = 0

    # set border tolerances to maximum_myelin_radius
    x_min = maximum_myelin_radius
    y_min = maximum_myelin_radius
    x_max = img.shape[0]-maximum_myelin_radius
    y_max = img.shape[1]-maximum_myelin_radius

    # build vectors for sweeping around axons
    angle_step = 2.0 * np.pi / n_angles
    vectors = [[] for i in range(n_angles)]
    for i in range(n_angles):
        angle = i * angle_step
        x_step = np.cos(angle)
        y_step = np.sin(angle)
        x_sign = 1
        y_sign = 1
        if (angle > np.pi / 2.0) and (angle <= np.pi):
            x_sign = -1
        elif (angle > np.pi) and (angle <= 3.0 * np.pi / 2.0):
            y_sign = -1
            x_sign = -1
        elif angle > 3.0*np.pi/2.0:
            y_sign = -1
        if y_step == 0.0:
            for j in range(maximum_myelin_radius):
                vectors[i].append((j * x_sign, 0))
        elif x_step == 0.0:
            for j in range(maximum_myelin_radius):
                vectors[i].append((0, j * y_sign))
        elif abs(x_step) > abs(y_step):
            adj = y_step / x_step
            for j in range(maximum_myelin_radius):
                x = j*x_sign
                y = round(x * adj)
                # if distance exceeds, break
                if distance.euclidean((0, 0), (x, y)) > maximum_myelin_radius:
                    break
                vectors[i].append((x, y))
        else:
            adj = x_step / y_step
            for j in range(maximum_myelin_radius):
                y = j * y_sign
                x = round(y * adj)
                if distance.euclidean((0, 0), (x, y)) > maximum_myelin_radius:
                    break
                vectors[i].append((x, y))

    valid_centers = []
    valid_widths = []
    valid_inner_radii = []
    n_skipped = 0
    n_rejected = 0
    # for each green particle (axon) center
    for cx, cy in green_centers_include:
        # blank working image
        working = img.copy()
        working[:, :, :] = 0
        # check if too close to edge
        if cx < x_min or cx > x_max or cy < y_min or cy > y_max:
            n_skipped += 1
        else:
            c_valid_widths = []
            c_n_zero_widths = 0
            c_valid_inner_radii = []
            for v in vectors:
                # crawl vector for axon init and terminal (store step # terminal)
                green_init = None
                green_terminus = None
                green_terminus_index = None
                ind = 0
                for vx, vy in v:
                    x = cx + vx
                    y = cy + vy
                    if green_init is None:
                        if unused[x, y, 1] > 0:
                            green_init = (x, y)
                    else:
                        if unused[x, y, 1] == 0:
                            green_terminus = (x, y)
                            green_terminus_index = ind
                            break
                    ind += 1

                if green_init is None:
                    green_terminus = (cx, cy)
                elif green_terminus is None:
                    green_terminus = v[-1]

                # crawl vector for red init and terminal (store step # init and terminal)
                red_init = None
                red_terminus = None
                red_init_index = None
                ind = 0
                for vx, vy in v:
                    x = cx + vx
                    y = cy + vy
                    if red_init is None:
                        if unused[x, y, 0] > 0:
                            red_init = (x, y)
                            red_init_index = ind
                    else:
                        if unused[x, y, 0] == 0:
                            red_terminus = (x, y)
                            break
                    ind += 1

                if red_init is None:
                    for vx, vy in v:
                        x = cx + vx
                        y = cy + vy
                        working[x, y, 1] = unused[x, y, 1]
                        if (x, y) == green_terminus:
                            break
                    c_n_zero_widths += 1    # width defined zero, but
                    continue                # inner radii is undefined so skip
                elif red_terminus is None:
                    continue
                else:
                    d = distance.euclidean(red_init, green_terminus)
                    w = distance.euclidean(red_init, red_terminus)
                    if (red_init_index <= green_terminus_index) or (d <= red_to_green_maximum_dist):
                        for vx, vy in v:
                            x = cx + vx
                            y = cy + vy
                            working[x, y, 0] = unused[x, y, 0]
                            working[x, y, 1] = unused[x, y, 1]
                            if (x, y) == red_terminus:
                                break
                        c_valid_widths.append(w)
                        c_valid_inner_radii.append(distance.euclidean((cx, cy), red_init))
                    else:
                        c_n_zero_widths += 1        # width defined zero, but
                        continue                    # inner radii is undefined so skip
            n_valid = len(c_valid_widths) * 1.0
            if n_valid / n_angles >= red_circle_percent_minimum:
                used = used + working
                unused = unused - working
                # update centers, widths, radii
                valid_centers.append((cx, cy))
                valid_widths.append(np.mean(c_valid_widths))
                valid_inner_radii.append(np.median(c_valid_inner_radii))
            else:
                n_rejected += 1
                rejected = rejected + working
    unused = unused - rejected

    # PROCESS AND MEASUREMENT FIGURE GENERATION
    temp_img3 = temp_img.copy()
    temp_img3[:, :, 0] = red_binary * 255.0
    temp_img3[:, :, 1] = green_binary * 255.0
    temp_img3[:, :, 2] = blue_binary * 255.0

    f, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(10, 10))

    ax0.imshow(img)
    ax0.set_title("original")
    ax1.imshow(unused)
    ax1.set_title("unused")
    ax2.imshow(rejected)
    ax2.set_title("rejected")
    ax3.imshow(used)
    ax3.set_title("used")
    plt.title(img_file[:-4] + "_process")
    plt.savefig(out_folder + "/" + img_file[:-4] + "_process.tif")
    plt.close(f)

    f, ax4 = plt.subplots(1, 1, figsize=(10, 10))

    ax4.imshow(temp_img3)

    for i in range(len(valid_centers)):
        c = valid_centers[i]
        inner_radii = valid_inner_radii[i]
        width = valid_widths[i]
        xy = (c[1], c[0])  # canvas is flipped matrix coords!
        circ1 = Circle(xy, 3, fill=True, color='w', alpha=1.0)
        circ2 = Circle(xy, inner_radii, fill=False, color='b')
        circ3 = Circle(xy, inner_radii+width, fill=False, color='b')
        ax4.add_patch(circ1)
        ax4.add_patch(circ2)
        ax4.add_patch(circ3)
    plt.title(img_file[:-4] + "_measurements")
    plt.savefig(out_folder + "/" + img_file[:-4] + "_measurements.tif")
    plt.close(f)

    total_tissue = green_binary | blue_binary | red_binary
    total_tissue_area = sum(sum(total_tissue))
    print total_tissue_area * um2_per_pixel_area

    # UPDATE DATA FILE
    out = img_file + "\t"
    out += str(green_total_valid_area * um2_per_pixel_area) + "\t"
    out += str(n_green) + "\t"
    out += str(um2_per_pixel_area * green_total_valid_area * 1.0 / n_green) + "\t"
    out += str(blue_total_valid_area * um2_per_pixel_area) + "\t"
    out += str(n_blue) + "\t"
    out += str(um2_per_pixel_area * blue_total_valid_area * 1.0 / n_blue) + "\t"
    out += str(um2_per_pixel_area * red_total_valid_area) + "\t"
    out += str(green_total_valid_area*1.0/red_total_valid_area) + "\t"
    out += str(len(valid_centers)) + "\t"
    out += str(np.mean(valid_widths) * um_per_pixel) + "\t"
    out += str(np.std(valid_widths) * um_per_pixel) + "\t"
    out += str(n_rejected) + "\t"
    if len(valid_centers) == 0:
        out += "inf\t"
    else:
        out += str(n_rejected * 100.0 / (len(valid_centers) + n_rejected)) + "\t"
    out += str(n_green * 1.0 / (total_tissue_area * um2_per_pixel_area)) + "\t"
    out += str(n_blue * 1.0 / (total_tissue_area * um2_per_pixel_area)) + "\t"
    out += str(total_tissue_area * um2_per_pixel_area) + "\t"
    data_out_file.write(out + "\n")

# OUTPUT PARAMETER FILE
parameter_out_file.write("green_threshold: " + str(green_threshold) + "\n")
parameter_out_file.write("blue_threshold: " + str(blue_threshold) + "\n")
parameter_out_file.write("red_threshold: " + str(red_threshold) + "\n")
parameter_out_file.write("area_calc_border_tolerance: " + str(area_calc_border_tolerance) + "\n")
parameter_out_file.write("intensity_threshold: " + str(intensity_threshold) + "\n")
parameter_out_file.write("green_intensity_threshold: " + str(green_intensity_threshold) + "\n")
parameter_out_file.write("blue_intensity_threshold: " + str(blue_intensity_threshold) + "\n")
parameter_out_file.write("red_intensity_threshold: " + str(red_intensity_threshold) + "\n")
parameter_out_file.write("n_angles: " + str(n_angles) + "\n")
parameter_out_file.write("um_per_pixel: " + str(um_per_pixel) + "\n")
parameter_out_file.write("red_to_green_maximum_dist: " + str(red_to_green_maximum_dist) + "\n")
parameter_out_file.write("red_circle_percent_minimum: " + str(red_circle_percent_minimum) + "\n")
parameter_out_file.write("maximum_myelin_radius: " + str(maximum_myelin_radius))

# DATA FILE CLEANUP
data_out_file.close()
parameter_out_file.close()
