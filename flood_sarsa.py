import numpy as np


def main():

    state_vector = np.zeros(6)  # vector is <water level, camera, level_analytics, camera_analytics, day/night, rain level> 
    
    base_vector_1 = state_vector
    base_vector_1[0] = base_vector_1[2] = 1  # turn on level sensor and level analytics during day
    base_vector_2 = state_vector
    base_vector_2[0] = base_vector_2[2] = base_vector_2[4] = 1 #turn on level sensor and level analytics during night. 


    

if __name__ == "__main__":
    main()


