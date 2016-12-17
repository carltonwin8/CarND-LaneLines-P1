import math
import cv2
import numpy as np

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, single_line, vertices):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    if single_line:
        lines = reduce_lines(lines, vertices)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def reduce_lines(lines, vertices):
    
    ybl = vertices[0][0][1]
    ytl = vertices[0][1][1]
    xtl = vertices[0][1][0]
    xtr = vertices[0][2][0]
    xmiddle = (xtr + xtl)/2
     
    def printTuple(t): # for debug
        x1, y1, x2, y2, m, b, x, y = t[0][0], t[0][1], t[0][2], t[0][3], t[1], t[2], t[3], t[4]
        print(str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2)
            + ',' + '{: 5.2f}'.format(m) + ',' + '{: 5.1f}'.format(b)
            + ',' + str(x) + ',' + str(y))
        
    def getXY(p,y):
        return([int(round((y-p[1])/p[0])),y])

    def getPoint(p):
        points = []
        points.extend(getXY(p,ybl))
        points.extend(getXY(p,ytl))
        return [points]
                      
    # l2r, r2l = [], [] # during debug
    xl2r, yl2r, xr2l, yr2l = [], [], [], []
    for line in lines:
        x1, y1, x2, y2 = line[0][0], line[0][1], line[0][2], line[0][3]
        if (x1 != x2) and (y1 != y2): # eliminate vertical and horizonal lines
            m = (y1 -y2)/(x1 - x2)
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            if m < 0: # check slope for LHS or RHS
                if x < xmiddle: # make sure on LHS or RHS
                    #l2r.append((line[0],m,b,x,y)) # used during debug
                    xl2r.append(x)
                    yl2r.append(y)
            else:
                if x > xmiddle:
                    #r2l.append((line[0],m,b,x,y)) # used during debug
                    xr2l.append(x)
                    yr2l.append(y)
    #for lr in l2r: # used during debug
    #    printTuple(lr)
    #    print(lr)
 
    #for rl in r2l: # used during debug
    #    printTuple(rl)
    #    print(lr)

    poly1 = np.polyfit(xl2r,yl2r,1)
    poly2 = np.polyfit(xr2l,yr2l,1)
    
    lines2 = [getPoint(poly1)]
    lines2.append(getPoint(poly2))
    print(lines2)
    #print(lines)
    return lines2
                      