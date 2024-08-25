# Import packages
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from timeit import default_timer as timer
from typing import Dict, List
from sklearn.model_selection import train_test_split
import platform
from scipy.stats import multivariate_normal, lognorm, expon, powerlaw
from sklearn.utils import shuffle
import os 

cpu_info = platform.processor()
print(f"CPU Model: {cpu_info}")

	
def generate_ratios(mean, cov, distribution='gaussian'):
    if distribution == 'lognormal':
        # Use the mean and standard deviation to create log-normal distribution
        mean_a_c, mean_b_c = mean
        var_a_c, var_b_c = np.diag(cov)
        s_a_c = np.sqrt(var_a_c)
        s_b_c = np.sqrt(var_b_c)
        
        # Generate a/c and b/c from log-normal distributions
        a_c = lognorm.rvs(s=s_a_c, scale=np.exp(mean_a_c))
        b_c = lognorm.rvs(s=s_b_c, scale=np.exp(mean_b_c))
    
    elif distribution == 'exponential':
        # Use the mean as the scale parameter for exponential distribution
        a_c = expon.rvs(scale=mean[0])
        b_c = expon.rvs(scale=mean[1])
    
    elif distribution == 'powerlaw':
        # Generate a/c and b/c from a power law distribution
        # The parameter `a` controls the shape of the distribution
        a_c = powerlaw.rvs(a=mean[0], scale=1)
        b_c = powerlaw.rvs(a=mean[1], scale=1)
    
    else:  # Gaussian by default
        # Generate ratios from a 2D Gaussian distribution
        a_c, b_c = multivariate_normal.rvs(mean, cov)
    
    return a_c, b_c

""" Functions that are used to generate filled sphere and
elipsoid"""

# Generate points on surface of sphere
def sphere(radius, num_points, plot=False):
    u = np.random.rand(num_points) 
    v = np.random.rand(num_points) 
    r = np.cbrt(np.random.rand(num_points) * radius)
     
    theta = u * 2 * np.pi
    phi = v * np.pi

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    if plot:
        # Plot Structure
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(projection='3d')
        plt.gca().set_aspect('auto', adjustable='box')
        ax.scatter(x,y,z, marker='.')
        ax.set_aspect('equal', 'box') #auto adjust limits
        #ax.axis('equal')
        ax.set_title('Structure of Circle', fontsize=10)
        plt.show()

    # Return points as list of tuples
    points = np.column_stack((x.flatten(), y.flatten(), z.flatten()))

    return points

def ellipsoid(radius, num_points, mean, cov, distribution='gaussian', plot=False):
    # Generate random angles and radius for spherical coordinates
    u = np.random.rand(num_points) 
    v = np.random.rand(num_points) 
    r = np.cbrt(np.random.rand(num_points))
     
    theta = u * 2 * np.pi
    phi = v * np.pi

    # Generate a/c and b/c ratios
    a_c, b_c = generate_ratios(mean, cov, distribution)

    # Generate ellipsoid coordinates with the scaling factors
    x = a_c * radius * r * np.sin(phi) * np.cos(theta)
    y = b_c * radius * r * np.sin(phi) * np.sin(theta)
    z = radius * r * np.cos(phi)  # Here c is set to radius

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, marker='.')
        ax.set_title(f'Ellipsoid with a/c={a_c:.2f}, b/c={b_c:.2f}')
        plt.show()
    
    # Return points as list of tuples
    points = np.column_stack((x.flatten(), y.flatten(), z.flatten()))

    return points

def points_projection(structure_coords, num_points):
    """ 
    Functions for projection
    """
    # Assign structure coords into z
    z = structure_coords
    
    # normal vectors generation
    normal = sphere(1, num_points)

    all_projected_points = []
    for n in normal:
        #Find two orthogonal vectors u and v (both orthogonal to n)
        #Calc value for t (random vector), ensuring not a scaled version of n
        if n[0] != 0:
            t = np.array([-(n[1]+n[2]) / n[0], 1, 1])
        elif n[1] != 0:
            t = np.array([-(n[0]+n[2]) / n[1], 1, 1])
        else:
            t = np.array([-(n[0]+n[1]) / n[2], 1, 1])
        
        u = np.cross(t,n)
        v = np.cross(n,u)
        
        # Normalize u and v (vector length become 1 unit long)
        u = u / np.linalg.norm(u)
        v = v / np.linalg.norm(v)
        
        vec_mat = np.array([u,v])
        
        #Project structure points onto plane
        #Individual component of normal
        a = n[0]
        b = n[1]
        c = n[2]
        #d = 0 #component of equation of planes

        projected_points = []
        for point in z:
            z1, z2, z3 = point
            
            k = (0 - a*z1 - b*z2 - c*z3) / (a**2 + b**2 + c**2) 
            
            p1 = z1 + k*a
            p2 = z2 + k*b
            p3 = z3 + k*c
            
            p = np.array([p1,p2,p3])

            #Convert 3D points to 2D
            p_trans = p.transpose()
            proj_2d = np.dot(vec_mat,p_trans)
            projected_points.append(proj_2d)
            
        all_projected_points.append(projected_points)

    return np.array(all_projected_points)


def cluster_per_cell(projected_points, image_size, grid_size):
    '''
    Functiom that transforms projections into grid and no of points
    '''
    all_projections = np.array(projected_points)
    image_size = image_size
    grid_x = grid_size[0]
    grid_y = grid_size[1]
    
    #Calc size of grid cell
    cell_x = image_size[0] / grid_x
    cell_y = image_size[1] / grid_y

    all_grid = []
    for projection in all_projections:
        grid = np.zeros((grid_x,grid_y), dtype=int)
        
        #Normalise 2D coords for better scalling (between 0-1)
        min_val = np.min(projection, axis=0)
        max_val = np.max(projection, axis=0)
        
        #Feature scaling 
        points_norm = (projection - min_val) / (max_val - min_val) 
        
        scaled_points = (points_norm * (np.array(image_size) - 1)).astype(int)
        
        for points in scaled_points:
            x,y = points
            gridx_index = int(x // cell_x) #floor division followed by conversion to integer
            gridy_index = int(y // cell_y)
            grid[gridy_index, gridx_index] += 1
            
        all_grid.append(grid)
        
    # transform into bw image 
    all_images = []
    for grid_img in all_grid:
        min = np.min(grid_img)
        max = np.max(grid_img)
        points_norm = (grid_img - min) / (max - min) 
        all_images.append(points_norm)

    return  all_images


def image_projection(coords, size):
    '''
    # Transform projected points into image with 1s and 0s
    '''
    all_projects = np.array(coords)
    image_size = size

    all_images = []
    for projects in all_projects:
    
        #Normalise 2D coords for better scalling (between 0-1)
        min_val = np.min(projects, axis=0)
        max_val = np.max(projects, axis=0)
        
        #Feature scaling 
        points_norm = (projects - min_val) / (max_val - min_val) 
        
        # Scale points to image size
        points_scaled = (points_norm * (np.array(image_size) -1 )).astype(int)
        
        # Create an empty image
        image = np.zeros(image_size)
        
        # Populate the image with points
        for point in points_scaled:
            x, y = point
            image[y,x] = 1  # Note: (y, x) because image coordinates are row-major
        
        all_images.append(image)
    
    return all_images


#Function that makes labels
def label_making(label_num, lst):
    label = [label_num] * len(lst)
    return label

def rotation(structure):
    point_cloud = structure
    
    # Convert to homogeneous coordinates
    point_cloud_homogeneous = []
    for point in structure:
        point_homogeneous = np.append(point,1)
        point_cloud_homogeneous.append(point_homogeneous)
    
    x, y, z= np.random.uniform(low = 0, high = 2 * np.pi, size=3)
    
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    
    rotate_x = np.array([
        [1, 0, 0, 0],
        [0, cx, -sx, 0],
        [0, sx, cx, 0],
        [0, 0, 0, 1],
        ])

    rotate_y = np.array([
        [cy, 0, sy, 0],
        [0, 1, 0, 0],
        [-sy, 0, cy, 0],
        [0, 0, 0, 1],
        ])

    rotate_z = np.array([
        [cy, -sy, 0, 0],
        [-sy, cy, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        ])

    # Rotate in a 3 axis
    rotated_points = np.matmul(
        point_cloud_homogeneous,
        rotate_x)
    
    rotated_points = np.matmul(
        rotated_points,
        rotate_y)
    
    rotated_points = np.matmul(
        rotated_points,
        rotate_z)
    
    # Convert to cartesian coordinates
    rotated_points_xyz = []
    for point in rotated_points:
        point = np.array(point[:-1])
        rotated_points_xyz.append(point)

    return np.array(rotated_points_xyz)
    
def system_maker(no_of_systems ,max_spheres, max_sphere_size, no_of_points, no_of_projections, image_res, distance):
    bw_img_all = []
    grid_img_all = []
    for _ in range(no_of_systems):
        systems = []
        no_of_spheres = np.random.randint(5,max_spheres) # min 5 sphere per system
        distance_s = np.random.randint(2, distance)
		
        for _ in range(no_of_spheres):
            a = sphere(max_sphere_size, no_of_points) # create sphere
            a = a + (np.random.rand(1,3) * np.random.randint(1,distance_s)) # translate sphere around
            systems.append(a) # add spheres
        
        systems = np.array(systems) #transform into numpy array
        systems = systems.reshape(-1 , systems.shape[-1]) # reshape so that spheres coordinates in each systems combines

        # Project 3D system ontto 2D plane
        proj_2D = points_projection(systems,no_of_projections)
        
        # Transform 2D points into 1s ad 0s image & 0-1 range image
        image_bw = image_projection(proj_2D, image_res)
        #image_contrast = cluster_per_cell(proj_2D, (720, 720), image_res)

        bw_img_all.append(image_bw)
        #grid_img_all.append(image_contrast)
        
        del systems, proj_2D

    bw_img_all = np.array(bw_img_all)
    #grid_img_all = np.array(grid_img_all)
    bw_img_all = bw_img_all.reshape(-1,bw_img_all.shape[-2],bw_img_all.shape[-1] )
    #grid_img_all = grid_img_all.reshape(-1,grid_img_all.shape[-2],grid_img_all.shape[-1] )

    return bw_img_all

def system_maker2(no_of_systems ,max_ellipsoids, max_ellipsoid_size, no_of_points, no_of_projections, image_res, distance, distribution_type='gaussian'):
    bw_img_all = []
    grid_img_all = []
    for _ in range(no_of_systems):
        systems = []
        no_of_ellipsoids = np.random.randint(5,max_ellipsoids) # min 5 sphere per system
        distance1 = np.random.randint(2, distance)
        
		
        for _ in range(no_of_ellipsoids):
            a = ellipsoid(radius=max_ellipsoid_size, num_points=no_of_points, mean=mean, cov=cov, distribution=distribution_type)
            a = rotation(a)
            a = a + (np.random.rand(1,3) * np.random.randint(1, distance1)) # translate sphere around
            systems.append(a) # add spheres
        
        systems = np.array(systems) #transform into numpy array
        systems = systems.reshape(-1 , systems.shape[-1]) # reshape so that spheres coordinates in each systems combines

        # Project 3D system ontto 2D plane
        proj_2D = points_projection(systems,no_of_projections)
        
        # Transform 2D points into 1s ad 0s image & 0-1 range image
        image_bw = image_projection(proj_2D, image_res)
        #image_contrast = cluster_per_cell(proj_2D, (720, 720), image_res)

        bw_img_all.append(image_bw)
        #grid_img_all.append(image_contrast)
        del systems, proj_2D

    bw_img_all = np.array(bw_img_all)
    #grid_img_all = np.array(grid_img_all)
    bw_img_all = bw_img_all.reshape(-1,bw_img_all.shape[-2],bw_img_all.shape[-1] )
    #grid_img_all = grid_img_all.reshape(-1,grid_img_all.shape[-2],grid_img_all.shape[-1] )

    return bw_img_all	
## Test script
cov_ab = 0.01  # Covariance between a/c and b/c
variance_a = 0.01  # Variance for a/c
variance_b = 0.1  # Variance for b/c

mean_a = mean_b = 0
mean = [mean_a, mean_b] # a/c, b/c
cov = [[variance_a, cov_ab], 
       [cov_ab, variance_b]]

d = 300

start = timer()
'''system_maker(no_of_systems ,max_spheres, max_sphere_size, no_of_points, no_of_projections, image_res)'''

# Generate training images
sphere_img_train = system_maker(100000, 300, 1, 5, 2, (64, 64),d)
ellips_img_train = system_maker2(100000, 300, 1, 5, 2, (64, 64),d, 'lognormal')

# Generate testing images with different parameters
sphere_img_test = system_maker(25000, 300, 1, 5, 2, (64, 64),d)
ellips_img_test = system_maker2(25000, 300, 1, 5, 2,(64, 64),d, 'lognormal')

end = timer()

print(f'\nvariance a:{variance_a}, variance b:{variance_b}, mean: {mean}, covarient: {cov_ab}')
print(f"Total generation time: {end-start:.3f} seconds")

# Concatenate the training images and labels
images_train = np.concatenate((sphere_img_train, ellips_img_train), axis=0)
labels_train = label_making(0, sphere_img_train) + label_making(1, ellips_img_train)

# Concatenate the testing images and labels
images_test = np.concatenate((sphere_img_test, ellips_img_test), axis=0)
labels_test = label_making(0, sphere_img_test) + label_making(1, ellips_img_test)

# Convert labels to numpy arrays
labels_array_train = np.array(labels_train)
labels_array_test = np.array(labels_test)


# Shuffle the training data
X_train, y_train = shuffle(images_train, labels_array_train, random_state=42)
X_val, y_val = shuffle(images_test, labels_array_test, random_state=42)
np.savez('dataset',X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}\n")
