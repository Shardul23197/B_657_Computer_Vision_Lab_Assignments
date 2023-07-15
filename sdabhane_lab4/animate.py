import numpy as np
from math import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# this function gets called every time a new frame should be generated.
# 
def animate_above(frame_number): 
    global tx, ty, tz, compass, tilt, twist
    #increment ty by 20 for every frame
    ty+=20

    #decrement tz by 20 for every frame
    tz-=20

    #focal point matrix
    focal_point_matrix=np.array([[0.002,0,0],[0,0.002,0],[0,0,1]])

    #tilt matrix
    tilt_matrix=np.array([[1,0,0],[0,cos(tilt),-sin(tilt)],[0,sin(tilt),cos(tilt)]])

    #twist matrix
    twist_matrix=np.array([[cos(twist),0,-sin(twist)],[0,1,0],[sin(twist),0,-cos(twist)]])

    #yaw matrix
    yaw_matrix=np.array([[cos(compass),-sin(compass),0],[sin(compass),cos(compass),0],[0,0,1]])

    #camera matrix
    camera_matrix=np.array([[1,0,0,tx],[0,1,0,ty],[0,0,1,tz]])

    #Multiplication done using library function. Source of function: https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
    #first multiplication for the final projection matrix
    mat_mul_1=np.matmul(focal_point_matrix, tilt_matrix)

    #second multiplication for the final projection matrix
    mat_mul_2=np.matmul(mat_mul_1,twist_matrix)

    #third multiplication for the final projection matrix
    mat_mul_3=np.matmul(mat_mul_2,yaw_matrix)

    #final projection matrix generated
    final_h=np.matmul(mat_mul_3,camera_matrix)
    
    pr=[]
    pc=[]
    for p in pts3:
        
        #Convert 3d to 4d points
        p_here=[[p[0]],[p[1]],[p[2]],[1]]

        #Multiply the new point matrix with projection matrix
        p_new=np.matmul(final_h,p_here)

        #Append x coordinate of final coordinate generated to pr
        pr += [p_new[0]/p_new[2]]

        #Append y coordinate of final coordinate generated to pc
        pc += [(p_new[1])/p_new[2]]
        
    plt.cla()
    plt.gca().set_xlim([-0.002,0.002])
    plt.gca().set_ylim([-0.002,0.002])

    #plot the points
    line, = plt.plot(pr, pc, 'k',  linestyle="", marker=".", markersize=2)
    return line,

# load in 3d point cloud
with open("airport.pts", "r") as f:
    pts3 = [ [ float(x) for x in l.split(" ") ] for l in f.readlines() ]

# initialize plane pose (translation and rotation)
(tx, ty, tz) = (0, 0, -10)
(compass, tilt, twist) = (0, -pi/2, pi)

# create animation!
fig, ax  = plt.subplots()
frame_count = 50
ani = animation.FuncAnimation(fig, animate_above, frames=range(0,frame_count))

# uncomment if you want to save your animation as a movie. :)
ani.save("final_movie.mp4")

plt.show()


