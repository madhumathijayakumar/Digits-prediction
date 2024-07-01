import numpy as np
from itertools import combinations


## working markers
# markers = [(142, 556), (2647, 552), (150, 1894), (2642, 1892)]
# # markers = [(176, 390), (2678, 388), (172, 1731), (2668, 1728)]
# markers = [(134, 241), (2628, 238), (126, 1584), (2625, 1580)]
# markers = [(176, 390), (2678, 388), (172, 1731), (2668, 1728)]
# markers = [(134, 241), (2628, 238), (126, 1584), (2625, 1580)]
# markers = [(136, 304), (2630, 282), (142, 1645), (2641, 1623)]

### not workingss...
# markers = [(142, 556), (2647, 552), (150, 1894), (2137, 1532)]

######################################################################
# avg _height = 1341                                                 #
# avg_width = 2498                                                   #
#                                                                    #
# 80% of avg_width and avg_height is min_width and min_height        #
######################################################################
min_width = 1998
min_height = 1072

avg_image_w=2498
avg_image_h=1341






def find_outermost_rectangle(coords):
    # Convert coordinates to numpy array for easier manipulation
    coords = np.array(coords)

    # Initialize variables to store the best rectangle found
    best_rectangle = None
    bigger_area = 0
    best_variance = float('inf')

    # Get all combinations of 4 points from the given coordinates
    point_combinations = combinations(coords, 4)
    print("point_combinations = ", point_combinations)
    
    for points in point_combinations:
        print("points = ", points)
        
        # converted = [tuple(point) for point in points]
        # print(converted)

        points = sorted(points, key=lambda p: (p[1], p[0]))  # Sort by y, then by x
        print("points after sorting: ", points)
        
        # Manually assign top-left, top-right, bottom-left, bottom-right
        top_left, top_right = sorted(points[:2], key=lambda p: p[0])
        bottom_left, bottom_right = sorted(points[2:], key=lambda p: p[0])
        # markers = [top_left, top_right, bottom_left, bottom_right]
        
        print(top_left)
        print(top_right)
        print(bottom_left)
        print(bottom_right)
    
        # Calculate potential rectangle properties
        top_left = np.array(top_left)
        top_right = np.array(top_right)
        bottom_left = np.array(bottom_left)
        bottom_right = np.array(bottom_right)
        
        print("after np array")
        print(top_left)
        print(top_right)
        print(bottom_left)
        print(bottom_right)
        
        
        converted = [tuple(top_left), tuple(top_right), tuple(bottom_left), tuple(bottom_right)]

        print(converted)
        st = is_skewed(converted)
        print("st = ",st)
        # exit(0)
        if not st:
            continue
        
        # width = np.linalg.norm(top_right - top_left)
        # height = np.linalg.norm(bottom_left - top_left)
        
        width = np.linalg.norm(bottom_right - bottom_left)
        height = np.linalg.norm(bottom_right - top_right)
        
        print("w,h = ",width, height)
        area = width * height
        print("area = ", area)
        print("bigger_area = ", bigger_area)
        if area > bigger_area:
            bigger_area = area
            best_rectangle = converted
        
        if best_rectangle:
            best_rectangle = sorted(best_rectangle, key=lambda p: (p[1], p[0]))  # Sort by y, then by x
            # Manually assign top-left, top-right, bottom-left, bottom-right
            top_left, top_right = sorted(best_rectangle[:2], key=lambda p: p[0])
            bottom_left, bottom_right = sorted(best_rectangle[2:], key=lambda p: p[0])
            best_rectangle = [top_left, top_right, bottom_left, bottom_right]
            print("sorted marker = ", best_rectangle)
    return best_rectangle
    

def find_fourth_point(markers):
    point1 = markers[0]
    point2 = markers[1]
    point3 = markers[2]
    tolerance = 50
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3
    
    # Check for tolerance in both x and y directions
    def within_tolerance(val1, val2):
        return abs(val1 - val2) <= tolerance
    
    # Determine the fourth point
    if within_tolerance(x1, x2):
        x4 = x3
    elif within_tolerance(x1, x3):
        x4 = x2
    elif within_tolerance(x2, x3):
        x4 = x1
    else:
        return None  # No valid rectangle could be formed
    
    if within_tolerance(y1, y2):
        y4 = y3
    elif within_tolerance(y1, y3):
        y4 = y2
    elif within_tolerance(y2, y3):
        y4 = y1
    else:
        return None  # No valid rectangle could be formed
    
    point4 = x4,y4
    point4 = 2796,531
    new_markers = markers
    new_markers.append(point4)
    print("new_markers = ", new_markers)
    return new_markers
    # return (x4, y4)
    

def distance(point1, point2):
    return ((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2) ** 0.5

def find_diagonal_points(points):
    if len(points) != 3:
        raise ValueError("The function expects exactly three points.")

    point1, point2, point3 = points

    # Calculate distances between each pair of points
    dist12 = distance(point1, point2)
    dist13 = distance(point1, point3)
    dist23 = distance(point2, point3)

    # Identify the two points that form the diagonal (longest distance)
    if dist12 >= dist13 and dist12 >= dist23:
        return point1, point2
    elif dist13 >= dist12 and dist13 >= dist23:
        return point1, point3
    else:
        return point2, point3

def find_fourth_point_v2(points):
    try:
        if len(points) != 3:
            raise ValueError("The function expects exactly three points.")

        # Find the points that form the diagonal
        diag_point1, diag_point2 = find_diagonal_points(points)
        
        # Identify the remaining point
        remaining_point = [p for p in points if p not in [diag_point1, diag_point2]][0]
        
        # Calculate the fourth point
        x4 = diag_point1[0] + diag_point2[0] - remaining_point[0]
        y4 = diag_point1[1] + diag_point2[1] - remaining_point[1]

        print(x4,y4)
        return (x4, y4)
    # point4 = x4, y4
    # new_markers = markers.copy()
    # new_markers.append(point4)
    # print("new_markers = ", new_markers)
    # return new_markers
    except Exception as e:
        print("Error in finding fourth point v2: ",e)
    


    
    
def dimension_checker(markers):
    print("inside dimension_chcker...")
    tl = markers[0]
    tr = markers[1]
    bl = markers[2]
    br = markers[3]
    width = abs(tl[0] - tr[0])
    print("width = ", width)
    height = abs(tr[1] - br[1])
    print("height = ", height)
    
    width2 = abs(bl[0] - br[0])
    print("width2 = ", width2)
    height2 = abs(tl[1] - bl[1])
    print("height2 = ", height2)
    
    return width > min_width and width2 > min_width and height > min_height and height2 > min_height
    # return width > min_width and height > min_height
    
    
    

def is_skewed(markers):
    top_ref = markers[0][0] if markers[0][0] >0 else 1
    left_ref = markers[0][1] if markers[0][1] >0 else 1
    right_ref = markers[1][0] if markers[1][0] >0 else 1
    bottom_ref = markers[2][1] if markers[2][1] >0 else 1
    r = []
    variance = 20
    
    print("inside skewed validator...")
    print(markers)
    t = abs(markers[2][0]-top_ref)
    print("t=",t)
    t = (t/top_ref)*100
    print("t = ",t)
    # Hack: left marker should be atleast 150px in x direction
    # In this case, we go for best 3 of 4
    if t<variance or top_ref < 150:
        r.append(True)
    
    t = abs(markers[1][1]-left_ref)
    print("t=",t)
    t = (t/left_ref)*100
    print("t = ",t)
    if t<variance:
        r.append(True)
    
    t = abs(markers[3][0]-right_ref)
    print("t=",t)
    t = (t/right_ref)*100
    print("t = ",t)
    if t<variance:
        r.append(True)
    
    t = abs(markers[3][1]-bottom_ref)
    print("t=",t)
    t = (t/bottom_ref)*100
    print("t = ",t)
    if t<variance:
        r.append(True)
    
    if len(r) == 4:
        return True
    else:
        return False


def get_quadrant (image_w, image_h, marker):
    if len(marker) != 2:
        return None

    try:
        center_x = image_w / 2
        center_y = image_h / 2
        print("Center x : ",center_x)
        print("Center y : ",center_y)
        marker_x = marker[0]
        marker_y = marker[1]
        print("Marker x : ",marker_x)
        print("Marker y : ",marker_y)

        if marker_x > center_x and marker_y < center_y:
            return 2
        elif marker_x < center_x and marker_y > center_y:
            return 3
        elif marker_x > center_x and marker_y > center_y:
            return 4
        elif marker_x < center_x and marker_y < center_y:
            return 1
        else:
            return None
    except Exception as err:
        print("Error occured in get_quadrant : ",err)
        return None


def get_markers_direction (image_w, image_h, m):
    q1=get_quadrant (image_w, image_h, m[0])
    q2=get_quadrant (image_w, image_h, m[1])
    
    if not q1 or not q2:
        return None

    if not (q1 in range (1, 5)):
        return None

    if not (q2 in range (1, 5)):
        return None

    sorted_q1, sorted_q2 = sorted ([q1, q2])
    q1=sorted_q1
    q2=sorted_q2

    sum_q = q1+q2
    if sum_q == 3:
        return "top"
    elif sum_q == 6:
        return "right"
    elif sum_q == 7:
        return "bottom"
    elif sum_q == 4:
        return "left"
    elif sum_q == 5:
        if q1==1 and q2==4:
            return "diag-back"
        elif q1==2 and q2==3:
            return "diag-front"
        else:
            return None

def calc_var (x, ref):
    ref = ref if ref >0 else 1
    return (abs(x-ref)/ref)*100

# def extrapolate_for_direction_right (markers):
#     if len(markers) != 2:
#         return None

#     markers = sorted (markers)
#     print (markers)
#     tr = markers[0]
#     br = markers[1]

#     tl_x = tr[0] - avg_image_w
#     bl_x = br[0] - avg_image_w

#     h = br[1]-tr[1]

#     var_tr = calc_var (br[0], tr[0])

#     tl_y = None
#     if tr[0] < br[0]:
#         tl_y = int(var_tr+tr[1])
#     else:
#         tl_y = int(tr[1]-var_tr)

#     bl_y = tl_y + h

#     return [(tl_x, tl_y), markers[0], (bl_x, bl_y), markers[1]]

# def extrapolate_for_direction_right (markers):
#     if len(markers) != 2:
#         return None

#     markers = sorted (markers)
#     print (markers)
#     tr = markers[0]
#     br = markers[1]
#     print(f"tr: {tr}, br: {br}")
#     tl_x = tr[0] - avg_image_w
#     bl_x = br[0] - avg_image_w
#     print(f"tl_x: {tl_x}, bl_x: {bl_x}")
#     h = br[1]-tr[1]
#     print(f"h: {h}")
#     var_tr = calc_var (br[0], tr[0])
#     print(f"var_tr: {var_tr}")
#     per_px = (var_tr/100) * tr[0]
#     print("perc_px: ", per_px)
#     tl_y = None
#     if tr[0] < br[0]:
#         tl_y = int(per_px+tr[1])
#     else:
#         tl_y = int(tr[1]-per_px)

#     bl_y = tl_y + h
#     print(f"tl_y: {tl_y}")
#     print(f"bl_y: {bl_y}")

#     return [(tl_x, tl_y), markers[0], (bl_x, bl_y), markers[1]]

# # def extrapolate_for_direction_bottom (markers):
# #     if len(markers) != 2:
# #         return None

# #     markers = sorted (markers)
# #     print (markers)
# #     bl = markers[0]
# #     br = markers[1]
# #     print(f"bl: {bl}, br: {br}")
# #     tl_y = abs(bl[1] - avg_image_h)
# #     tr_y = abs(br[1] - avg_image_h)
# #     print(f"tl_y: {tl_y}, tr_y: {tr_y}")
    
# #     tl_x = bl[0]
# #     tr_x = br[0]
# #     print(f"tl_x: {tl_x}, tr_x: {tr_x}")
   

# #     return [(tl_x, tl_y), (tr_x, tr_y), markers[0], markers[1]]

# def extrapolate_for_direction_left (markers):
#     if len(markers) != 2:
#         return None

#     markers = sorted (markers)
#     print (markers)
#     tl = markers[0]
#     bl = markers[1]
#     print(f"tl: {tl}, bl: {bl}")
#     tr_x = abs(tl[0] + avg_image_w)
#     br_x = abs(bl[0] + avg_image_w)
#     print(f"tr_x: {tr_x}, br_x: {br_x}")
#     h = bl[1]-tl[1]
#     print(f"h: {h}")
    
#     var_tr = calc_var (bl[0], tl[0])
#     print(f"var_tr: {var_tr}")
#     tr_y = None
#     if tl[0] < bl[0]:
#         tr_y = int(var_tr+tl[1])
#     else:
#         tr_y = int(tl[1]-var_tr)

#     br_y = tr_y + h
    
#     print(f"tr_y: {tr_y}")
#     print(f"br_y: {br_y}")
    
#     return [markers[0], (tr_x, tr_y), markers[1], (br_x,br_y)]

# def extrapolate_for_direction_diag_back (markers):
#     if len(markers) != 2:
#         return None

#     markers = sorted (markers)
#     print (markers)
#     tl = markers[0]
#     br = markers[1]
#     print(f"tl: {tl}, br: {br}")
#     bl_x = tl[0]
#     bl_y = br[1]
    
#     tr_x = br[0]
#     tr_y = tl[1]
#     print(f"bl_x: {bl_x}, tr_x: {tr_x}")
    
#     print(f"bl_y: {bl_y}, tr_y: {tr_y}")
   

#     return [markers[0], (tr_x, tr_y), (bl_x,bl_y), markers[1]]

# # def extrapolate_for_direction_top (markers):
# #     if len(markers) != 2:
# #         return None

# #     markers = sorted (markers)
# #     print (markers)
# #     tl = markers[0]
# #     tr = markers[1]
# #     print(f"tl: {tl}, tr: {tr}")
# #     bl_y = abs(tl[1] + avg_image_h)
# #     br_y = abs(tr[1] + avg_image_h)
# #     print(f"bl_y: {bl_y}, br_y: {br_y}")
    
# #     bl_x = tl[0]
# #     br_x = tr[0]
    
# #     print(f"bl_x: {bl_x}, br_x: {br_x}")
   
# #     return [markers[0], markers[1], (bl_x,bl_y), (br_x, br_y)]

# def extrapolate_for_direction_diag_front (markers):
#     if len(markers) != 2:
#         return None

#     markers = sorted (markers)
#     print (markers)
#     bl = markers[0]
#     tr = markers[1]
#     print(f"tr: {tr}, bl: {bl}")
#     tl_x = bl[0]
#     tl_y = tr[1]
    
#     br_x = tr[0]
#     br_y = bl[1]
#     print(f"tl_x: {tl_x}, br_x: {br_x}")
    
#     print(f"tl_y: {tl_y}, tr_y: {br_y}")
   

#     return [ (tl_x, tl_y), markers[0], markers[1], (br_x, br_y)]


# def extrapolate_for_direction_bottom (markers):
#     if len(markers) != 2:
#         return None

#     markers = sorted (markers)
#     print (markers)
#     bl = markers[0]
#     br = markers[1]
#     print(f"bl: {bl}, br: {br}")
#     tl_y = bl[1] - avg_image_h
#     tr_y = br[1] - avg_image_h
#     print(f"tl_y: {tl_y}, tr_y: {tr_y}")
    
#     w = br[0]-bl[0]
#     print(w)

#     var_tl = calc_var (br[1], bl[1])
#     print(var_tl)

#     tl_x = None
#     if bl[1] < br[1]:
#         tl_x = int(bl[0]-var_tl)
#     else:
#         tl_x = int(var_tl+bl[0])

#     tr_x = tl_x + w
    
#     return [(tl_x, tl_y),(tr_x, tr_y),markers[0], markers[1]]
    
# def extrapolate_for_direction_top (markers):
#     if len(markers) != 2:
#         return None

#     markers = sorted (markers)
#     print (markers)
#     tl = markers[0]
#     tr = markers[1]
#     print(f"tl: {tl}, tr: {tr}")
#     bl_y = abs(tl[1] + avg_image_h)
#     br_y = abs(tr[1] + avg_image_h)
#     print(f"bl_y: {bl_y}, br_y: {br_y}")
    
#     w = tr[0]-tl[0]
#     print(w)

#     var_tl = calc_var (tr[1], tl[1])
#     print(var_tl)

#     bl_x = None
#     if tl[1] < tr[1]:
#         bl_x = int(tl[0]-var_tl)
#     else:
#         bl_x = int(var_tl+tl[0])

#     br_x = bl_x + w
   
#     return [markers[0], markers[1], (bl_x,bl_y), (br_x, br_y)]


def calc_var_in_pixels (x, var_x):
    try:
        return int((x*var_x)/100)
    except Exception as e:
        print("error in calc_var_in_pixels",e)
        return 0
    
def extrapolate_for_direction_right (markers):
    if len(markers) != 2:
        return None

    markers = sorted (markers)
    print (markers)
    tr = markers[0]
    br = markers[1]

    tl_x = tr[0] - avg_image_w
    bl_x = br[0] - avg_image_w

    h = br[1]-tr[1]

    var_tr = calc_var (br[0], tr[0])

    tl_y = None
    var_tr_pixels = calc_var_in_pixels (tr[1], var_tr)
    if tr[0] < br[0]:
        tl_y = int(var_tr_pixels+tr[1])
    else:
        tl_y = int(tr[1]-var_tr_pixels)

    bl_y = tl_y + h

    return [(tl_x, tl_y), markers[0], (bl_x, bl_y), markers[1]]

def extrapolate_for_direction_top (markers):
    if len(markers) != 2:
        return None

    markers = sorted (markers)
    print (markers)
    tl = markers[0]
    tr = markers[1]

    bl_y = tl[1] + avg_image_h
    br_y = tr[1] + avg_image_h

    w = tr[0]-tl[0]

    var_tl = calc_var (tr[1], tl[1])

    bl_x = None
    var_tl_pixels = calc_var_in_pixels (tl[0], var_tl)
    if tl[1] < tr[1]:
        bl_x = int(tl[0]-var_tl_pixels)
    else:
        bl_x = int(var_tl_pixels+tl[0])

    br_x = bl_x + w

    return [markers[0], markers[1], (bl_x, bl_y), (br_x, br_y)]

# def extrapolate_for_direction_left (markers):
#     if len(markers) != 2:
#         return None

#     markers = sorted (markers)
#     print (markers)
#     tl = markers[0]
#     bl = markers[1]
#     print(f"tl: {tl}, bl: {bl}")
#     tr_x = abs(tl[0] + avg_image_w)
#     br_x = abs(bl[0] + avg_image_w)
#     print(f"tr_x: {tr_x}, br_x: {br_x}")
#     h = bl[1]-tl[1]
#     print(f"h: {h}")
    
#     var_tl = calc_var (bl[0], tl[0])
#     print(f"var_tl: {var_tl}")
#     tr_y = None
#     var_tl_pixels = calc_var_in_pixels (tl[1], var_tl)
#     if tl[0] < bl[0]:
#         tr_y = int(var_tl_pixels+tl[1])
#     else:
#         tr_y = int(tl[1]-var_tl_pixels)

#     br_y = tr_y + h
    
#     print(f"tr_y: {tr_y}")
#     print(f"br_y: {br_y}")
    
#     return [markers[0], (tr_x, tr_y), markers[1], (br_x,br_y)]

def extrapolate_for_direction_bottom (markers):
    if len(markers) != 2:
        return None

    markers = sorted (markers)
    print (markers)
    bl = markers[0]
    br = markers[1]
    print(f"bl: {bl}, br: {br}")
    tl_y = bl[1] - avg_image_h
    tr_y = br[1] - avg_image_h
    print(f"tl_y: {tl_y}, tr_y: {tr_y}")
    
    w = br[0]-bl[0]
    print(w)

    var_bl = calc_var (br[1], bl[1])
    print(var_bl)

    tl_x = None
    var_bl_pixels = calc_var_in_pixels (bl[0], var_bl)
    if bl[1] < br[1]:
        tl_x = int(var_bl_pixels+bl[0])
    else:
        tl_x = int(bl[0]-var_bl_pixels)

    tr_x = tl_x + w
    
    return [(tl_x, tl_y),(tr_x, tr_y),markers[0], markers[1]]


def sort_left (markers):
    m0 = markers[0]
    m1 = markers[1]

    return [m0, m1] if m0[1] < m1[1] else [m1, m0]

def calc_left_pad(markers):
    # when x>150px accuracy is good
    # Hence we using min_pad = 200
    min_pad = 400
    
    m0 = markers[0]
    m1 = markers[1]
    
    return min_pad - m0[0] if m0[0]<m1[0] else min_pad - m1[0]
    
        
    
def extrapolate_for_direction_left (markers):
    pad = 0
    
    if len(markers) != 2:
        return None

    #Suj: Custom sort as it fails for left...
    markers = sort_left (markers)
    print (markers)
    tl = markers[0]
    bl = markers[1]
    print(f"tl: {tl}, bl: {bl}")

    if tl[0]<150 or bl[0]<150: pad = calc_left_pad(markers)

    print(f"pad: {pad}")
    tl_list = list(tl)
    bl_list = list(bl)

    tl_list[0] = tl_list[0] + pad
    bl_list[0] = bl_list[0] + pad

    tl = tuple(tl_list)
    bl = tuple(bl_list)
    print(f" after tl: {tl}, bl: {bl}")
    tr_x = tl[0] + avg_image_w
    br_x = bl[0] + avg_image_w
    print(f"tr_x: {tr_x}, br_x: {br_x}")
    h = bl[1]-tl[1]
    print(f"h: {h}")

    var_tl = calc_var (bl[0], tl[0])
    print(f"var_tl: {var_tl}")
    tr_y = None
    var_tl_pixels = calc_var_in_pixels (tl[1], var_tl)
    if tl[0] < bl[0]:
        tr_y = int(tl[1]-var_tl_pixels)
    else:
        tr_y = int(var_tl_pixels+tl[1])

    br_y = tr_y + h

    print(f"tr_y: {tr_y}")
    print(f"br_y: {br_y}")

    return [markers[0], (tr_x-pad, tr_y), markers[1], (br_x-pad,br_y)]