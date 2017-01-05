# Finding Lane Lines on the Road

The original readme before this repository was forked can be seen
[here](README_ori.md). For details about the assignment please
refer to that readme. The following text provides reflections on the project.

## Reflections

### Current Pipeline

The current pipeline identifies lines on a road as follows.

  1. Takes an image that contains a road with lines marking the lanes.
  2. Converts the image to gray scale.
  3. Blurs the image in order to reduce the number or sharp edges.
  4. Detects edges in the image using the canny algorithm.
  5. Masks out edges that will not likely be part of the lane.
  6. Detect lines using the edges in the mask using the hough line detect algorithm.
  7. Use the detected lines to determine a single left and right lane line.
     1. Use slope to detect right and left lane lines.
     2. Use the multiple lines detected and extrapolate a single line for the left and right.
  8. Overlay the detected left and right lane lines on the original image.
  9. Repeat the above steps for multiple images (aka video).

### Shortcomings

Some of the shortcomings are:

  - The algorithm will fail when the road is not within the mask.
  - The algorithm that I provided failed to consistently detect lanes.
    This was noted when multiple images (aka video) was processed.

### Improvements

Some improvements that could be made are:

  - Some how make a "intelligent mask" that always knows where the road is.
  - Improve that lane detection algorithm so that it is more consistent without
    increasing the computing load significantly.
