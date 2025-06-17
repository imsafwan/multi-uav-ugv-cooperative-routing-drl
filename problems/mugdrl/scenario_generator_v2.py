'''
random scenario generators 
It generates seperate UGV points, sperate UAV points 
Any UAV point is not away > 6.5 km than any UGV point
Any UAV point is not < 2 km than its closest UGV point 

'''


# Libraries #

import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.spatial import distance
import os

#np.random.seed(0)
#random.seed(0)

# ---- initialization ------- #

output_folder = 'generated_scenarios'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


all_ugv_points = all_ugv_points = [[ 6. ,   8.  ],
 [ 5.5 ,  8.  ],
 [ 5.  ,  8.  ],
 [ 4.5  , 8.  ],
 [ 4.  ,  8.  ],
 [ 3.5 ,  8.  ],
 [ 3.   , 8.  ],
 [ 6.36 , 8.36],
 [ 6.73 , 8.73],
 [ 7.09 , 9.09],
 [ 7.45 , 9.45],
 [ 7.82 , 9.82],
 [ 8.18 ,10.18],
 [ 8.55 ,10.55],
 [ 8.91 ,10.91],
 [ 9.27 ,11.27],
 [ 9.64 ,11.64],
 [10.  , 12.  ],
 [ 6.4 ,  7.7 ],
 [ 6.8  , 7.4 ],
 [ 7.2  , 7.1 ],
 [ 7.6  , 6.8 ],
 [ 8.   , 6.5 ],
 [ 8.4   ,6.2 ],
 [ 8.8  , 5.9 ],
 [ 9.2  , 5.6 ],
 [ 9.6 ,  5.3 ],
 [10.  ,  5.  ]]

def sample_in_circle(center, radius, num_samples):
    angles = np.random.uniform(0, 2 * np.pi, num_samples)
    radii = np.sqrt(np.random.uniform(0, radius**2, num_samples))
    x_vals = center[0] + radii * np.cos(angles)
    y_vals = center[1] + radii * np.sin(angles)
    return np.column_stack((x_vals, y_vals))

    
def generate_uav_points(ugv_points, ua_p_d_f, safe_radius):
    
    all_potential_points = np.vstack([sample_in_circle(ugv, safe_radius, ua_p_d_f ) for ugv in ugv_points]) # Generate potential points for each UGV point and concatenate
    ugv_points_set = set(map(tuple, ugv_points))
    all_potential_points = np.array([point for point in all_potential_points if tuple(point) not in ugv_points_set])
    
    # Randomly select the required number of UAV points
    if len(all_potential_points) > ua_p_d_f:
        uav_points = all_potential_points[np.random.choice(len(all_potential_points), ua_p_d_f, replace=False)]
    else:
        uav_points = all_potential_points
        kkk

    return uav_points





def scenario_gen(uav_graph_size = 15, ugv_graph_size = 5):
    
    ua_p_d_f = uav_graph_size # number of uav points              
    ug_p_d_f = ugv_graph_size # number of ugv points
    safe_radius = 7.0 # km  

    mandatory_points = [[3 , 8], [10 , 5], [10, 12] ]  # mandatory points: terminal points
    remaining_points = [point for point in all_ugv_points if point not in mandatory_points]
    sampled_points = random.sample(remaining_points, ug_p_d_f - len(mandatory_points))
    ugv_points = mandatory_points + sampled_points
    ugv_points_miles = np.round(np.array(ugv_points) * 0.62137,2)
    
    assert len(ugv_points_miles) == len(set(map(tuple, ugv_points_miles))), "Duplicates found in UGV points"

    uav_points_miles = None
    
    while True:
        
        uav_points = generate_uav_points(ugv_points, uav_graph_size, safe_radius)
        uav_points_miles = np.round(uav_points * 0.62137, 2)
        
        # Convert to set for comparison
        uav_set = set(map(tuple, uav_points_miles))
        ugv_set = set(map(tuple, ugv_points_miles))

        # Check for duplicates between uav and ugv
        if uav_set.isdisjoint(ugv_set):
            break

    # Assert that there are no duplicates between uav and ugv
    assert uav_set.isdisjoint(ugv_set), "Duplicates found between UAV and UGV points"

    return uav_points_miles, ugv_points_miles
    
    
    


