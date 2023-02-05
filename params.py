import numpy as np

# Screen parameters
screen_width = 1200 # in pixels
screen_height = 900 # in pixels
screen_bg = (128, 128, 128)
flip_time = 100 # In milliseconds

# Image parameters
img_width = 4
img_height = 4

# Board size
x_cells = int(screen_width/img_width) # Number of images that can be fit in a row
y_cells = int(screen_height/img_height) # Number of images that can be fit in a column
print(f'Board size = {x_cells} * {y_cells}')
total_cells = x_cells * y_cells

# Game params
start_smilies = np.random.randint(int(total_cells*0.4), int(total_cells*0.6))
step_counter = 0
kernel = np.ones((3, 3))
kernel[1, 1] = 0 # Filter for image convolution
