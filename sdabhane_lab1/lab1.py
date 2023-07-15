#Import the Image and ImageFilter classes from PIL (Pillow)
from PIL import Image
from PIL import ImageFilter
import sys
import random
import numpy as np
import PIL

#I've used the gray image for both examples to do the convolutions on. 

#Source of image used for other image:
#https://www.starsandstripesfc.com/2021/2/8/22267768/black-history-month-marcus-rashford-united-kingdom-england-school-meal-voucher-program-children
#Photo by Michael Regan/Getty Images
 

#Function from https://stackoverflow.com/questions/2448015/2d-convolution-using-python-and-numpy
#Line 23,24 from https://github.com/ashushekar/image-convolution-from-scratch

def convolution_function(image, kernel):
    
    m, n = kernel.shape
    kernel=np.flipud(np.fliplr(kernel))
    
    y1, x1 = image.shape
    #y1 = y1 - m + 1
    #x1 = x1 - m + 1
        
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    image_padded[1:-1, 1:-1] = image
    if kernel.shape==(5,5):
        y1 = y1 - m + 1
        x1 = x1 - m + 1
        new_image = np.zeros((y1,x1))
        for i in range(y1):
            for j in range(x1):
                new_image[i][j] = np.sum(image_padded[i:i+m, j:j+m]*kernel)
        return new_image
    else:
        new_image = np.zeros((y1,x1))
        for i in range(y1):
            for j in range(x1):
                new_image[i][j] = np.sum(image_padded[i:i+m, j:j+m]*kernel)
        return new_image


    
if __name__ == '__main__':
    # Load an image 
    im = Image.open(sys.argv[1])

    # Check its width, height, and number of color channels
    print("Image is %s pixels wide." % im.width)
    print("Image is %s pixels high." % im.height)
    print("Image mode is %s." % im.mode)

    # Pixels are accessed via an (X,Y) tuple.
    # The coordinate system starts at (0,0) in the upper left-hand corner,
    # and increases moving right (first coordinate) and down (second coordinate).
    # So it's a (col, row) indexing system, not (row, col) like we're used to
    # when dealing with matrices or 2d arrays.
    print("Pixel value at (10,10) is %s" % str(im.getpixel((10,10))))
    
    # Pixels can be modified by specifying the coordinate and RGB value
    # (255, 0, 0) is a pure red pixel.
    im.putpixel((10,10), (255, 0, 0))
    print("New pixel value is %s" % str(im.getpixel((10,10))))

    # Let's create a grayscale version of the image:
    # the "L" means there's only a single channel, "Lightness"
    gray_im = im.convert("L")
    
    # Create a new blank color image the same size as the input
    color_im = Image.new("RGB", (im.width, im.height), color=0)
    gray_im.save("gray.png")
    
    # Highlights any very dark areas with yellow.
    for x in range(im.width):
        for y in range(im.height):
            p = gray_im.getpixel((x,y))
            if p < 5:
                (R,G,B) = (255,255,0)
                color_im.putpixel((x,y), (R,G,B))
            else:
                color_im.putpixel((x,y), (p,p,p))

    # Show the image. We're commenting this out because it won't work on the Linux
    # server (unless you set up an X Window server or remote desktop) and may not
    # work by default on your local machine. But you may want to try uncommenting it,
    # as seeing results in real-time can be very useful for debugging!
    # color_im.show()

    # Save the image
    color_im.save("output.png")

    # This uses Pillow's code to create a 5x5 mean filter and apply it to
    # our image. In the lab, you'll need to write your own convolution code (using
    # "for" loops, but you can use Pillow's code to check that your answer is correct.
    # Since the input is a color image, Pillow applies the filter to each
    # of the three color planes (R, G, and B) independently.
    box = [1]*25
    result = color_im.filter(ImageFilter.Kernel((5,5),box,sum(box)))
    # result.show()
    result.save("output2.png")

    #identity kernel
    identity_kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    image_here1=np.array(gray_im)
    #identity_filter_here = convolution_function(image_here1, identity_kernel)
    identity_filter_here=convolution_function(image_here1,identity_kernel)
    x=Image.fromarray(identity_filter_here)
    x = x.convert('RGB')
    x.save("a.png")
    print("Save")

    #test image for identity filter
    # x_result=gray_im.filter(ImageFilter.Kernel((3, 3),(0, 0, 0, 0, 1, -0, 0, 0, 0)))
    # x_result.save("Test_Identity_Filter.png")
    # print("Save")
    #x_result=gray_im.filter(ImageFilter.Kernel((3, 3),(0, 0, 0, 0, 1, -0, 0, 0, 0), 1, 0))
    #x_result.save("LinearfilterResult2.png")

    #Box filter
    box_blur_kernel = np.array([[0.1111111111111111, 0.1111111111111111, 0.1111111111111111], [0.1111111111111111, 0.1111111111111111, 0.1111111111111111], [0.1111111111111111, 0.1111111111111111, 0.1111111111111111]])
    #box_filter=convolution_function(image_here1, box_blur_kernel)
    box_filter=convolution_function(image_here1, box_blur_kernel)
    x1=Image.fromarray(box_filter)
    x1 = x1.convert('RGB')
    x1.save("b.png")
    print("Save")

    #test image for box filter
    # box_result=gray_im.filter(ImageFilter.Kernel((3, 3),(0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111)))
    # box_result.save("Test_Box_Filter.png")
    # print("Save")

    #horizontal kernel
    horizontal_kernel = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
    #horinzontal_filter = convolution_function(image_here1, horizontal_kernel)
    horinzontal_filter = convolution_function(image_here1, horizontal_kernel)
    x2=Image.fromarray(horinzontal_filter)
    x2 = x2.convert('RGB')
    x2.save("c.png")
    print("Save")

    #test image for horizontal filter
    # horizontal_test=gray_im.filter(ImageFilter.Kernel((3, 3),(0, 0, 0,-1, 0, 1,0, 0, 0)))
    # horizontal_test.save("Test_HorizontalDerivative.png")
    # print("Save")

    #approximate gaussian
    gaussian_kernel=np.array([[0.003,0.013,0.022,0.013,0.003],[0.013, 0.059, 0.097, 0.059, 0.013],[0.022,0.097,0.159,0.097,0.022],[0.013,0.059,0.097,0.059,0.013],[0.003,0.013,0.022,0.013,0.003]])
    gaussian_filter = convolution_function(image_here1, gaussian_kernel)
    x3=Image.fromarray(gaussian_filter)
    x3 = x3.convert('RGB')
    x3.save("d.png")
    print("Save")

    #test image for approximate gaussian filter
    # x3_result=gray_im.filter(ImageFilter.Kernel((5, 5),(0.003,0.013,0.022,0.013,0.003,0.013, 0.059, 0.097, 0.059, 0.013,0.022,0.097,0.159,0.097,0.022,0.013,0.059,0.097,0.059,0.013,0.003,0.013,0.022,0.013,0.003)))
    # x3_result.save("Test_Gaussian_Filter.png")
    # print("Save")

   
    #calculate ((1+alpha)(a)-d)). Calcuated alpha=2 for sharper image. 
    identity_kernel_1=np.multiply(identity_kernel,2)
    #identity_kernel_1=(1+0.25)*identity_kernel
    #identity_kernel_1
    identity_kernel_2 = np.zeros((identity_kernel_1.shape[0] + 2, identity_kernel_1.shape[1] + 2))
    identity_kernel_2[1:-1, 1:-1] = identity_kernel_1
    sharpening_kernel=np.subtract(identity_kernel_2,gaussian_kernel)
    sharpening_filter = convolution_function(image_here1, sharpening_kernel)
    x4=Image.fromarray(sharpening_filter)
    x4 = x4.convert('RGB')
    x4.save("e.png")
    print("Save")

    #test image for sharpening filter filter
    # sharpened2 = gray_im.filter(ImageFilter.SHARPEN)
    # sharpened2.save("Test_Sharpening_Filter.png")
    # print("Save")
    #print(type(sharpened2))
    #print(identity_kernel_2.shape)
    #print(identity_kernel_2)

    #filter f
    final_filter=convolution_function(gaussian_kernel,horizontal_kernel)
    #print(final_filter.shape)
    the_f_filter=convolution_function(image_here1,final_filter)
    x5=Image.fromarray(the_f_filter)
    x5 = x5.convert('RGB')
    x5.save("f.png")
    print("Save")
    #print(final_filter)

    # the_result_here_final_filter_test=gray_im.filter(ImageFilter.Kernel((3, 3),final_filter))
    # the_result_here_final_filter_test.save("Test_Derivative_of_Gaussian.png")
    # print("Save")