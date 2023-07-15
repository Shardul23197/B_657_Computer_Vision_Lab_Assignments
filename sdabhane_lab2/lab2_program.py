#Import relevant class from PIL (Pillow)
from PIL import Image , ImageOps , ImageDraw
import numpy as np
from scipy import fftpack


#Source of image used for other image:
#https://www.mirror.co.uk/sport/football/news/man-utds-scott-mctominay-hits-24164232
im1 = Image.open("input_A.jpg")

#Source of image used for other image:
#https://www.starsandstripesfc.com/2021/2/8/22267768/black-history-month-marcus-rashford-united-kingdom-england-school-meal-voucher-program-children
#Photo by Michael Regan/Getty Images
im2=Image.open("input_B.jpg")

#Resize image for easier combining of images
im2=im2.resize((im1.size[0],im1.size[1]))

#Code from lines 21-44 from Lab 2 PDF
#Convert image modes
im1 = ImageOps.grayscale(im1)
x1,y1 = im1.size

#Convert image modes
im2 = ImageOps.grayscale(im2)
x2,y2 = im2.size

# we will create a circle with a diameter of 30 pixels
eX1,eY1=30,30

# we will create a circle with a diameter of 30 pixels
eX2,eY2=30,30


#create bounding box info for drawing ellipse
bbox1=(x1/2-eX1/2,y1/2-eY1/2,x1/2+eX1/2,y1/2+eY1/2)

#create bounding box info for drawing ellipse
bbox2=(x2/2-eX2/2,y2/2-eY2/2,x2/2+eX2/2,y2/2+eY2/2)

# create low filter image
low_pass = Image.new("L" ,(im1.width , im1.height),color =0)
draw1 = ImageDraw.Draw(low_pass)
draw1.ellipse(bbox1,fill =255)

# create hight filter image
high_pass = Image.new("L" ,(im2.width , im2.height),color =255)
draw2 = ImageDraw.Draw(high_pass)
draw2.ellipse(bbox2,fill =0)

 
# turn filter image into numpy array
low_pass = np.array(low_pass)

# turn filter image into numpy array
high_pass = np.array(high_pass)


#fft of image 1
fft1 = fftpack.fftshift(fftpack.fft2(im1))
image_1_fft_array=(np.log(abs(fft1))* 255 /np.amax(np.log(abs(fft1)))).astype(np.uint8)
image_1_fft=Image.fromarray(image_1_fft_array)
image_1_fft.convert('L').save("image_1_fft.png")

#apply low pass filter. Make pixels intensity everywhere except the center white circle to 0.
for i in range(0,(fft1.shape[0])):
    for j in range(0,(fft1.shape[1])):
        if i in range(fft1.shape[0]//2-eX1//2,fft1.shape[0]//2+eX1//2) and j in range(fft1.shape[1]//2-eY1//2,fft1.shape[1]//2+eY1//2):        
            continue
        else:
            fft1[i][j]=0

#Get fft inverse for image 1
image_1_ifft2 = abs(fftpack.ifft2(fftpack.ifftshift(fft1)))


#fft of image 2
fft2 = fftpack.fftshift(fftpack.fft2(im2))
image_2_fft_array=(np.log(abs(fft2))* 255 /np.amax(np.log(abs(fft2)))).astype(np.uint8)
image_2_fft=Image.fromarray(image_2_fft_array)
image_2_fft.convert('L').save("image_2_fft.png")

#apply high pass filter. Make pixels intensity at the center black circle to 0.
for i in range((fft2.shape[0]//2),(fft2.shape[0]//2)+eX2//2):
    for j in range((fft2.shape[1]//2),(fft2.shape[1]//2)+eX2):
        fft2[i][j]=0

for i in range((fft2.shape[0]//2),(fft2.shape[0]//2)+eX2//2):
    for j in range((fft2.shape[1]//2),(fft2.shape[1]//2)-eX2//2,-1):
        fft2[i][j]=0

for i in range((fft2.shape[0]//2),(fft2.shape[0]//2)-eX2//2,-1):
    for j in range((fft2.shape[1]//2),(fft2.shape[1]//2)+eX2//2):
        fft2[i][j]=0

for i in range((fft2.shape[0]//2),(fft2.shape[0]//2)-eX2//2,-1):
    for j in range((fft2.shape[1]//2),(fft2.shape[1]//2)-eX2//2,-1):
        fft2[i][j]=0


##Get fft inverse for image 2
image_2_ifft2 = abs(fftpack.ifft2(fftpack.ifftshift(fft2)))


#combine result of high pass and low pass
final_image_array=image_1_ifft2+image_2_ifft2
final_image=Image.fromarray(final_image_array)
final_image.convert('RGB').save("Hybrid_Image.png")