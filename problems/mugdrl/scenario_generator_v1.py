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



# ---- initialization ------- #

output_folder = 'generated_scenarios'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)




def scenario_gen(uav_graph_size = 20):
    
    ua_p_d_f = uav_graph_size # number of uav points              
    ug_p_d_f = 2 # controls number of ugv points
        
    p1 = (6,8)   #(random.randint(6,8),random.randint(8,10)) # Nodal point
    p2 = (3, 8)  #(random.randint(2,6),random.randint(8,12))
    p3 = (8, 13) #(random.randint(8,12),random.randint(10,14))
    p4 = (10, 5) #(random.randint(14,16),random.randint(6,8))

    points = [p1, p2, p3, p4]


    ugv_points = []
    
    for i in range(1, len(points)):
        coeff = np.polyfit([p1[0], points[i][0]], [p1[1], points[i][1]], 1)
        f = np.poly1d(coeff)
        # Calculate Euclidean distance
        dist = distance.euclidean(p1, points[i])
        # Calculate number of steps based on desired step size of 0.25
        num_steps = int(dist / ug_p_d_f) + 1
        x = np.linspace(p1[0], points[i][0], num_steps )
        y = f(x)
        ugv_points += list(zip(x, y))
        #plt.scatter(x, y, c ='r')
        


    # Now we generate UAV points
    uav_points = []
    while len(uav_points) < ua_p_d_f:
        # Select a random UGV point to generate nearby UAV points
        base_ugv_point = random.choice(ugv_points)

        # Generate a potential UAV point near the selected UGV point
        potential_uav_point = (
            random.uniform(base_ugv_point[0] - 5.5, base_ugv_point[0] + 5.5),
            random.uniform(base_ugv_point[1] - 5.5, base_ugv_point[1] + 5.5)
        )

        # Check distance criteria
        if any(distance.euclidean(potential_uav_point, ugv_point) <= 5.5 for ugv_point in ugv_points) and \
           all(distance.euclidean(potential_uav_point, ugv_point) >= 2 for ugv_point in ugv_points):
            uav_points.append(potential_uav_point)
            
            
            
    '''uav_points = []
    
    while len(uav_points) < ua_p_d_f :
        potential_uav_point = (random.uniform(1, 18), random.uniform(1, 18))
        if any(distance.euclidean(potential_uav_point, ugv_point) <= 5.5 for ugv_point in ugv_points):
            if all(distance.euclidean(potential_uav_point, ugv_point) >= 2 for ugv_point in ugv_points):
               uav_points.append(potential_uav_point)'''

    # Plot UAV points
    uav_points = np.array(uav_points)
    #plt.scatter(uav_points[:, 0], uav_points[:, 1], color='blue')


    uav_points_miles = np.round(uav_points * 0.62137,2)
    ugv_points_miles = np.round(np.array(ugv_points) * 0.62137,2)


    # Show the plot
    #plt.grid(True)
    #plt.show()
    
    return uav_points_miles, ugv_points_miles




'''while True: 
    
    
    uav_points_miles, ugv_points_miles = scenario_gen()
    print(type(uav_points_miles), uav_points_miles)
    kk
    
    command = input('Enter command (press "q" to save and exit): ')
    if command == 'q':
        np.savetxt(os.path.join(output_folder, "uav_data_points_sc1.csv") , uav_points_miles, delimiter=",", header="UAV_X,UAV_Y", comments='')
        np.savetxt(os.path.join(output_folder, "ugv_data_points_sc1.csv"), ugv_points_miles, delimiter=",", header="UGV_X,UGV_Y", comments='')
        break  
    elif command == 'k':
        break
    else:
        continue'''