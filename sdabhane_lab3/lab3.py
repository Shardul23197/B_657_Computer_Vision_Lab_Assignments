#import libraries
from PIL import Image,ImageDraw
import numpy as np
from scipy import signal

#function to normalize numpy.int32 values.Function taken from lab3 pdf
def normalize(input_im) :
    base = input_im.min()
    roof = input_im.max()
    diff = roof-base
    scale= diff/255
    input_im = input_im-base
    output = input_im/scale
    return np.uint8(output)



#Open the image 
im = Image.open("input.png")

#Step 1: Convert given image to grayscale
im1=im.convert('L')

#Save grayscale image
im1.save("grayscaleofinput.png")

#Save image as an array of data type int32
im2=np.array(im1).astype('int32')

#Defining the kernels to compute partial derivatives
derivative_kernel = np.array([-1,1])

#x derivative kernel
x_derivative_kernel = derivative_kernel.reshape(1, 2)

#y derivative kernel
y_derivative_kernel = derivative_kernel.reshape(2, 1)

#Step 2: Convolve the grayscale image in X and Y directions
#signal.convolve2d function from: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html
I_x=signal.convolve2d(im1, x_derivative_kernel,mode='same')
I_y=signal.convolve2d(im2, y_derivative_kernel,mode='same')

#Save the images from step 2
I_x_image=Image.fromarray(normalize(I_x))
I_y_image=Image.fromarray(normalize(I_y))
I_x_image.save("Ix_result.png")
I_y_image.save("Iy_result.png")


#Step 3: Let Ix^2=Ix*Ix
I_x_2=I_x*I_x

# Step 4:  Let IxIy=Ix*Iy
I_x_I_y=I_x*I_y

# Step 5: Let Iy^2=Iy*Iy
I_y_2=I_y*I_y


#Save images from Steps 3,4,5
I_x_2_image=Image.fromarray(normalize(I_x_2))
I_y_2_image=Image.fromarray(normalize(I_y_2))
I_x_I_y_image=Image.fromarray(normalize(I_x_I_y))
I_x_2_image.save("Ix2_result.png")
I_y_2_image.save("Iy2_result.png")
I_x_I_y_image.save("IxIy_result.png")


#Define box filter for step 6
box_filter=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

#Step 6: Calculate AW , BW , CW by convolving Ix^2,IxIy,Iy^2 with box filter
#signal.convolve2d function from: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html
A_w=signal.convolve2d(I_x_2, box_filter,mode='same')
B_w=signal.convolve2d(I_x_I_y, box_filter,mode='same')
C_w=signal.convolve2d(I_y_2, box_filter,mode='same')

#Save results of step 6
A_w_image=Image.fromarray(normalize(A_w))
B_w_image=Image.fromarray(normalize(B_w))
C_w_image=Image.fromarray(normalize(C_w))
A_w_image.save("aw_result.png")
B_w_image.save("bw_result.png")
C_w_image.save("cw_result.png")

#Step 7: Compute Eigenvalues of the matrix at each pixel and only keep those where min(λ1, λ2) > T where T is some threshold.
#Step 7 steps from lab3 pdf
#After trying different threshold values, settled for 26000 as threshold value.
draw = ImageDraw.Draw(im)
for x in range(im1.size[0]):
    for y in range(im1.size[1]):
        e_values , e_vectors = np.linalg.eig(np.array([[A_w[y][x],B_w[y][x]],[B_w[y][x],C_w[y][x]]]))
        if min(e_values) > 26000:
            #Step 8: Mark the points which are above threshold and save final image
            draw.line(((x-1,y),(x+1,y)),fill=(255,0,0))
            draw.line(((x,y-1),(x,y+1)),fill=(255,0,0))

#Saving image from step 8
im.save("FinalResult.png")



