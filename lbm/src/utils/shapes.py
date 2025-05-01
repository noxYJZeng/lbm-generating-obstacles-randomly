# Generic imports
import os
import os.path
import PIL
import math
import scipy.special
import matplotlib
import numpy             as np
import matplotlib.pyplot as plt
import random


### ************************************************
### Class defining shape object
class shape:
    ### ************************************************
    ### Constructor
    def __init__(self,
                name,
                position,
                control_pts,
                n_control_pts,
                n_sampling_pts,
                radius,
                edgy,
                output_dir,
                rotation_angle=0.0):  # 🆕 新增角度参数

        self.name           = name
        self.position       = position
        self.control_pts    = control_pts
        self.n_control_pts  = n_control_pts
        self.n_sampling_pts = n_sampling_pts
        self.curve_pts      = np.array([])
        self.area           = 0.0
        self.size_x         = 0.0
        self.size_y         = 0.0
        self.radius         = radius
        self.edgy           = edgy
        self.output_dir     = output_dir
        self.rotation_angle = rotation_angle



    ### ************************************************
    ### Reset object
    def reset(self):

        # Reset object
        self.name           = 'shape'
        self.control_pts    = np.array([])
        self.n_control_pts  = 0
        self.n_sampling_pts = 0
        self.radius         = np.array([])
        self.edgy           = np.array([])
        self.curve_pts      = np.array([])
        self.area           = 0.0

    ### ************************************************
    ### Build shape
    def build(self):

        # Center set of points
        center = np.mean(self.control_pts, axis=0)
        self.control_pts -= center

        # Sort points counter-clockwise
        control_pts, radius, edgy  = ccw_sort(self.control_pts,
                                              self.radius,
                                              self.edgy)

        local_curves = []
        delta        = np.zeros([self.n_control_pts,2])
        radii        = np.zeros([self.n_control_pts,2])
        delta_b      = np.zeros([self.n_control_pts,2])

        # Compute all informations to generate curves
        for i in range(self.n_control_pts):
            # Collect points
            prv  = (i-1)
            crt  = i
            nxt  = (i+1)%self.n_control_pts
            pt_m = control_pts[prv,:]
            pt_c = control_pts[crt,:]
            pt_p = control_pts[nxt,:]

            # Compute delta vector
            diff         = pt_p - pt_m
            diff         = diff/np.linalg.norm(diff)
            delta[crt,:] = diff

            # Compute edgy vector
            delta_b[crt,:] = 0.5*(pt_m + pt_p) - pt_c

            # Compute radii
            dist         = compute_distance(pt_m, pt_c)
            radii[crt,0] = 0.5*dist*radius[crt]
            dist         = compute_distance(pt_c, pt_p)
            radii[crt,1] = 0.5*dist*radius[crt]

        # Generate curves
        for i in range(self.n_control_pts):
            crt  = i
            nxt  = (i+1)%self.n_control_pts
            pt_c = control_pts[crt,:]
            pt_p = control_pts[nxt,:]
            dist = compute_distance(pt_c, pt_p)
            smpl = math.ceil(self.n_sampling_pts*math.sqrt(dist))

            local_curve = generate_bezier_curve(pt_c,           pt_p,
                                                delta[crt,:],   delta[nxt,:],
                                                delta_b[crt,:], delta_b[nxt,:],
                                                radii[crt,1],   radii[nxt,0],
                                                edgy[crt],      edgy[nxt],
                                                smpl)
            local_curves.append(local_curve)

        curve          = np.concatenate([c for c in local_curves])
        x, y           = curve.T
        z              = np.zeros(x.size)
        self.curve_pts = np.column_stack((x,y,z))
        self.curve_pts = remove_duplicate_pts(self.curve_pts)

        # Center set of points
        center            = np.mean(self.curve_pts, axis=0)
        self.curve_pts   -= center
        self.control_pts[:,0:2] -= center[0:2]

        # Reprocess to position
        self.control_pts[:,0:2] += self.position[0:2]
        self.curve_pts  [:,0:2] += self.position[0:2]

    ### ************************************************
    ### Write image
    def generate_image(self, *args, **kwargs):
        # Optional arguments
        plot_pts = kwargs.get('plot_pts', True)

        # Max view size to avoid generating overly large PNGs
        MAX_VIEW_SIZE = 2.5
        MARGIN = 0.2  # padding around shape

        # If any axis range is not provided, calculate from curve_pts
        if 'xmin' in kwargs:
            xmin = kwargs['xmin']
        else:
            xmin = max(np.min(self.curve_pts[:, 0]) - MARGIN, -MAX_VIEW_SIZE)
        if 'xmax' in kwargs:
            xmax = kwargs['xmax']
        else:
            xmax = min(np.max(self.curve_pts[:, 0]) + MARGIN, MAX_VIEW_SIZE)

        if 'ymin' in kwargs:
            ymin = kwargs['ymin']
        else:
            ymin = max(np.min(self.curve_pts[:, 1]) - MARGIN, -MAX_VIEW_SIZE)
        if 'ymax' in kwargs:
            ymax = kwargs['ymax']
        else:
            ymax = min(np.max(self.curve_pts[:, 1]) + MARGIN, MAX_VIEW_SIZE)

        # Plot shape
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        plt.axis('off')
        plt.gca().set_aspect('equal', adjustable='box')

        # Background fill
        plt.fill([xmin, xmax, xmax, xmin],
                [ymin, ymin, ymax, ymax],
                color=(0.784, 0.773, 0.741),
                linewidth=2.5,
                zorder=0)

        # Shape fill
        plt.fill(self.curve_pts[:, 0],
                self.curve_pts[:, 1],
                'black',
                linewidth=0,
                zorder=1)

        # Plot control points if requested
        if plot_pts:
            colors = matplotlib.cm.ocean(np.linspace(0, 1, self.n_control_pts))
            plt.scatter(self.control_pts[:, 0],
                        self.control_pts[:, 1],
                        color=colors,
                        s=16,
                        zorder=2,
                        alpha=0.5)

        # Save image
        filename = os.path.join(self.output_dir, self.name + '.png')
        plt.savefig(filename, dpi=200)
        plt.close(plt.gcf())
        plt.cla()
        trim_white(filename)


    ### ************************************************
    ### Write csv
    def write_csv(self):
        filename = os.path.join(self.output_dir, self.name + '.csv')
        with open(filename,'w') as file:
            # Write header including rotation angle
            file.write('{} {} {:.6f}\n'.format(self.n_control_pts,
                                            self.n_sampling_pts,
                                            self.rotation_angle))

            # Write control points coordinates
            for i in range(0,self.n_control_pts):
                file.write('{} {} {} {}\n'.format(self.control_pts[i,0],
                                                self.control_pts[i,1],
                                                self.radius[i],
                                                self.edgy[i]))


    ### ************************************************
    ### Read csv and initialize shape with it
    def read_csv(self, filename, *args, **kwargs):
        # Handle optional argument
        keep_numbering = kwargs.get('keep_numbering', False)

        if (not os.path.isfile(filename)):
            print('I could not find csv file: '+filename)
            print('Exiting now')
            exit()

        self.reset()
        sfile  = filename.split('.')
        sfile  = sfile[-2]
        sfile  = sfile.split('/')
        name   = sfile[-1]

        if (keep_numbering):
            sname = name.split('_')
            name  = sname[0]

        x      = []
        y      = []
        radius = []
        edgy   = []

        with open(filename) as file:
            header         = file.readline().split()
            n_control_pts  = int(header[0])
            n_sampling_pts = int(header[1])

            for i in range(0,n_control_pts):
                coords = file.readline().split()
                x.append(float(coords[0]))
                y.append(float(coords[1]))
                radius.append(float(coords[2]))
                edgy.append(float(coords[3]))
                control_pts = np.column_stack((x,y))

        self.__init__(name,
                      control_pts,
                      n_control_pts,
                      n_sampling_pts,
                      radius,
                      edgy)

    ### ************************************************
    ### Modify shape given a deformation field
    def modify_shape_from_field(self, deformation, pts_list):

        # Deform shape
        for i in range(len(pts_list)):
            self.control_pts[pts_list[i],0] = deformation[i,0]
            self.control_pts[pts_list[i],1] = deformation[i,1]
            self.edgy[pts_list[i]]          = deformation[i,2]

### End of class Shape
### ************************************************

### ************************************************
### Compute distance between two points
def compute_distance(p1, p2):

    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

### ************************************************
### Generate cylinder points
def generate_cylinder_pts(n_pts):
    if (n_pts < 4):
        print('Not enough points to generate cylinder')
        exit()

    pts = np.zeros([n_pts, 2])
    ang = 2.0*math.pi/n_pts
    for i in range(0,n_pts):
        pts[i,:] = [0.5*math.cos(float(i)*ang),
                    0.5*math.sin(float(i)*ang)]

    return pts

### ************************************************
### Generate prism1 points正
def generate_prism1_pts(n_pts):
  if (n_pts != 3):
      print('You should have n_pts = 3 for a triangular prism1')
      exit()

  pts = np.zeros([n_pts, 2])
  pts[0,:] = [0.0, 0.0]
  pts[1,:] = [1.0, 0.0]
  pts[2,:] = [0.5, 1.0]

  pts[:,:] *= 0.5
  return pts

### ************************************************
### Generate prism2 points反
def generate_prism2_pts(n_pts):
  if (n_pts != 3):
      print('You should have n_pts = 3 for a triangular prism2')
      exit()

  pts = np.zeros([n_pts, 2])
  pts[0,:] = [0.0, 0.0]
  pts[1,:] = [1.0, 0.0]
  pts[2,:] = [0.5, -1.0]

  pts[:,:] *= 0.5
  return pts

### ************************************************
### Generate square points
def generate_square_pts(n_pts):
    if (n_pts != 4):
        print('You should have n_pts = 4 for square')
        exit()

    pts       = np.zeros([n_pts, 2])
    pts[0,:]  = [ 1.0, 1.0]
    pts[1,:]  = [-1.0, 1.0]
    pts[2,:]  = [-1.0,-1.0]
    pts[3,:]  = [ 1.0,-1.0]

    pts[:,:] *= 0.5

    return pts

### ************************************************
### Generate ellipse points
def generate_ellipse_pts(n_pts, a=1.0, b=0.5):
    pts = np.zeros([n_pts, 2])
    ang = 2.0 * math.pi / n_pts
    for i in range(n_pts):
        theta = i * ang
        pts[i, 0] = a * math.cos(theta)
        pts[i, 1] = b * math.sin(theta)
    return pts


### ************************************************
### Generate 5-point star (pentagram) points
def generate_star_pts(n_pts):
    if n_pts != 10:
        raise ValueError("Five-point star needs exactly 10 points (outer + inner)")

    pts = np.zeros((n_pts, 2))
    outer_radius = 1.0
    inner_radius = 0.4

    for i in range(n_pts):
        angle = i * np.pi / 5  # 36 degrees
        radius = outer_radius if i % 2 == 0 else inner_radius
        pts[i, 0] = radius * np.cos(angle)
        pts[i, 1] = radius * np.sin(angle)

    return pts * 0.5


### ************************************************
### Generate heart shape points
def generate_heart_pts(n_pts):
    pts = np.zeros([n_pts, 2])
    theta_vals = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    for i, theta in enumerate(theta_vals):
        x = 16 * np.sin(theta) ** 3
        y = 13 * np.cos(theta) - 5 * np.cos(2 * theta) - 2 * np.cos(3 * theta) - np.cos(4 * theta)
        pts[i, 0] = x
        pts[i, 1] = y
    pts *= 0.2  # scale down to similar size as other obstacles
    return pts

### ************************************************
### Generate hexagon points
def generate_hexagon_pts(n_pts):
    if n_pts != 6:
        print("Hexagon shape requires n_pts = 6")
        exit()

    pts = np.zeros([n_pts, 2])
    for i in range(n_pts):
        angle = 2 * math.pi * i / n_pts
        pts[i, 0] = math.cos(angle)
        pts[i, 1] = math.sin(angle)
    return pts * 0.5



### if you want to define more obstacles, start here.

### ************************************************
### Remove duplicate points in input coordinates array
### WARNING : this routine is highly sub-optimal
def remove_duplicate_pts(pts):
    to_remove = []

    for i in range(len(pts)):
        for j in range(len(pts)):
            # Check that i and j are not identical
            if (i == j):
                continue

            # Check that i and j are not removed points
            if (i in to_remove) or (j in to_remove):
                continue

            # Compute distance between points
            pi = pts[i,:]
            pj = pts[j,:]
            dist = compute_distance(pi,pj)

            # Tag the point to be removed
            if (dist < 1.0e-8):
                to_remove.append(j)

    # Sort elements to remove in reverse order
    to_remove.sort(reverse=True)

    # Remove elements from pts
    for pt in to_remove:
        pts = np.delete(pts, pt, 0)

    return pts

### ************************************************
### Counter Clock-Wise sort
###  - Take a cloud of points and compute its geometric center
###  - Translate points to have their geometric center at origin
###  - Compute the angle from origin for each point
###  - Sort angles by ascending order
def ccw_sort(pts, rad, edg):
    geometric_center = np.mean(pts,axis=0)
    translated_pts   = pts - geometric_center
    angles           = np.arctan2(translated_pts[:,1], translated_pts[:,0])
    x                = angles.argsort()
    pts2             = np.array(pts)
    rad2             = np.array(rad)
    edg2             = np.array(edg)

    return pts2[x,:], rad2[x], edg2[x]

### ************************************************
### Compute Bernstein polynomial value
def compute_bernstein(n,k,t):
    k_choose_n = scipy.special.binom(n,k)

    return k_choose_n * (t**k) * ((1.0-t)**(n-k))

### ************************************************
### Sample Bezier curves given set of control points
### and the number of sampling points
### Bezier curves are parameterized with t in [0,1]
### and are defined with n control points P_i :
### B(t) = sum_{i=0,n} B_i^n(t) * P_i
def sample_bezier_curve(control_pts, n_sampling_pts):
    n_control_pts = len(control_pts)
    t             = np.linspace(0, 1, n_sampling_pts)
    curve         = np.zeros((n_sampling_pts, 2))

    for i in range(n_control_pts):
        curve += np.outer(compute_bernstein(n_control_pts-1, i, t),
                          control_pts[i])

    return curve

### ************************************************
### Generate Bezier curve between two pts
def generate_bezier_curve(p1,       p2,
                          delta1,   delta2,
                          delta_b1, delta_b2,
                          radius1,  radius2,
                          edgy1,    edgy2,
                          n_sampling_pts):

    # Lambda function to wrap angles
    #wrap = lambda angle: (angle >= 0.0)*angle + (angle < 0.0)*(angle+2*np.pi)

    # Sample the curve if necessary
    if (n_sampling_pts != 0):
        # Create array of control pts for cubic Bezier curve
        # First and last points are given, while the two intermediate
        # points are computed from edge points, angles and radius
        control_pts      = np.zeros((4,2))
        control_pts[0,:] = p1[:]
        control_pts[3,:] = p2[:]

        # Compute baseline intermediate control pts ctrl_p1 and ctrl_p2
        ctrl_p1_base = radius1*delta1
        ctrl_p2_base =-radius2*delta2

        ctrl_p1_edgy = radius1*delta_b1
        ctrl_p2_edgy = radius2*delta_b2

        control_pts[1,:] = p1 + edgy1*ctrl_p1_base + (1.0-edgy1)*ctrl_p1_edgy
        control_pts[2,:] = p2 + edgy2*ctrl_p2_base + (1.0-edgy2)*ctrl_p2_edgy

        # Compute points on the Bezier curve
        curve = sample_bezier_curve(control_pts, n_sampling_pts)

    # Else return just a straight line
    else:
        curve = p1
        curve = np.vstack([curve,p2])

    return curve

### ************************************************
### Crop white background from image
def trim_white(filename):
    im   = PIL.Image.open(filename)
    bg   = PIL.Image.new(im.mode, im.size, (255,255,255))
    diff = PIL.ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    cp   = im.crop(bbox)
    cp.save(filename)


### ************************************************
### Generate shape
def generate_shape(n_pts,
                   position,
                   shape_type,
                   shape_size,
                   shape_name,
                   n_sampling_pts,
                   base_output_dir):
    # Check input
    if (shape_type not in ['cylinder', 'square', 'prism1', 'prism2',
                           'random', 'ellipse', 'star', 'heart', 'hexagon']):
        print('Error in shape_type')
        print('Authorized values are "cylinder", "square", "prism1", "prism2", "random", "ellipse", "star", "heart", "hexagon"')
        exit()

    shape_folder = base_output_dir

    # Select shape type
    if (shape_type == 'cylinder'):
        radius   = 0.5 * np.ones((n_pts))
        edgy     = 1.0 * np.ones((n_pts))
        ctrl_pts = generate_cylinder_pts(n_pts)

    elif (shape_type == 'square'):
        radius   = np.zeros((n_pts))
        edgy     = np.ones((n_pts))
        ctrl_pts = generate_square_pts(n_pts)

    elif (shape_type == 'prism1'):
        radius   = np.zeros((n_pts))
        edgy     = np.ones((n_pts))
        ctrl_pts = generate_prism1_pts(n_pts)

    elif (shape_type == 'prism2'):
        radius   = np.zeros((n_pts))
        edgy     = np.ones((n_pts))
        ctrl_pts = generate_prism2_pts(n_pts)

    elif (shape_type == 'random'):
        radius   = np.random.uniform(low=0.8, high=1.0, size=n_pts)
        edgy     = np.random.uniform(low=0.45, high=0.55, size=n_pts)
        ctrl_pts = np.random.rand(n_pts, 2)

    elif (shape_type == 'ellipse'):
        radius   = np.ones((n_pts))
        edgy     = np.ones((n_pts))
        ctrl_pts = generate_ellipse_pts(n_pts)

    elif (shape_type == 'star'):
        if n_pts != 10:
            print("Star shape requires n_pts = 10")
            exit()
        radius   = np.ones((n_pts))
        edgy     = np.ones((n_pts))
        ctrl_pts = generate_star_pts(n_pts)

    elif (shape_type == 'heart'):
        radius   = np.ones((n_pts))
        edgy     = np.ones((n_pts))
        ctrl_pts = generate_heart_pts(n_pts)

    elif (shape_type == 'hexagon'):
        if n_pts != 6:
            print("Hexagon shape requires n_pts = 6")
            exit()
        radius   = np.ones((n_pts))
        edgy     = np.ones((n_pts))
        ctrl_pts = generate_hexagon_pts(n_pts)

    # Uniform scale
    ctrl_pts *= shape_size

    # Apply random rotation
    theta = random.uniform(0, 2 * math.pi)
    rotation_matrix = np.array([[math.cos(theta), -math.sin(theta)],
                                [math.sin(theta),  math.cos(theta)]])
    ctrl_pts = ctrl_pts @ rotation_matrix.T

    # Initialize and build shape
    s = shape(shape_name,
              position,
              ctrl_pts,
              n_pts,
              n_sampling_pts,
              radius,
              edgy,
              shape_folder,
              rotation_angle=theta)

    s.build()
    # s.generate_image(xmin=-shape_size,
    #                  xmax=shape_size,
    #                  ymin=-shape_size,
    #                  ymax=shape_size)
    s.write_csv()

    return s
