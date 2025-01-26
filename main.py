from tkinter import filedialog, Menu, Label, Scale, HORIZONTAL, Toplevel, Entry, Button 
from PIL import Image, ImageTk 
import tkinter as tk 
import numpy as np # https://numpy.org/devdocs/reference/arrays.html very helpful for using NumPy for  this assignment and keeping track of 
import math

loaded_image = None
image_history = []
current_index = -1
current_angle = 0

# Function to open and display the BMP image useful for the PIL library: https://www.geeksforgeeks.org/python-pil-image-open-method/ and https://www.geeksforgeeks.org/loading-images-in-tkinter-using-pil/
def open_bmp(label):
    global loaded_image
    filepath = filedialog.askopenfilename(filetypes=[("BMP files", "*.bmp")])
    if filepath:
        loaded_image = Image.open(filepath)
        display_image(loaded_image, "Original Image")
        return loaded_image
    return None

def display_image(modified, title="Mini Photoshop"):
    img_tk = ImageTk.PhotoImage(modified)
    label.config(image=img_tk)
    label.image = img_tk
    root.title(title)

# Function to exit the application
def exit_app():
    root.quit() 

#Function to grayscale image
def grayscale(image):
    # Y' = 0.299 R + 0.5870 G + 0.1140 B 
    if image is None:
        print("No image loaded.")
        return None

    # Create a grayscale image using the  nums from lecture
    gray_image = np.zeros((image.height, image.width, 3), dtype=np.uint8)
    try:
        R = np.array(image)[:, :, 0] * 0.299
        G = np.array(image)[:, :, 1] * 0.587
        B = np.array(image)[:, :, 2] * 0.114
        gray_value = R + G + B
        for i in range(3):
            gray_image[:, :, i] = gray_value
    except IndexError as e:
        print("Error during grayscale conversion:", e)
        return None

    add_to_history(Image.fromarray(gray_image))
    return Image.fromarray(gray_image.astype(np.uint8))

# Function to display images side by side
def display_side_by_side(original, modified, title="Mini Photoshop"):
    if modified.size != original.size:
        modified = modified.resize(original.size)

    # Create a new image to hold both side by side
    combined = Image.new("RGB", (original.width * 2, original.height))
    combined.paste(original, (0, 0))
    combined.paste(modified, (original.width, 0))
    display_image(combined, title)


#Function to perform Ordered Dithering (onto grayscale image)
def ordered_dithering(image_pixel):
    # ensure theres is an image loaded and that its grayscale for this task (per instructions)
    if image_pixel is None:
        print("No image loaded.")
        return None
    grayscale_image = grayscale(image_pixel)
    grayscale_np = np.array(grayscale_image)[:, :, 0]

    # use the 4x4 Bayer matrix for dithering, https://en.wikipedia.org/wiki/Ordered_dithering
    dither_matrix = np.array([
    [0,  8,  2, 10],
    [12, 4, 14,  6],
    [3, 11,  1,  9],
    [15, 7, 13,  5]
    ]) * (255 / 15)
    
    n = dither_matrix.shape[0]
    
    # Apply ordered dithering algorithm, https://numpy.org/devdocs/reference/generated/numpy.ndarray.shape.html
    dithered_image = np.zeros_like(grayscale_np, dtype=np.uint8)
    for y in range(grayscale_np.shape[0]):
        for x in range(grayscale_np.shape[1]):
            i, j = x % n, y % n
            if grayscale_np[y, x] > dither_matrix[i, j]:
                dithered_image[y, x] = 255
            else:
                dithered_image[y, x] = 0

    dithered_img_pil = Image.fromarray(dithered_image, "L").convert("RGB")
    add_to_history(dithered_img_pil)
    return dithered_img_pil 

# Function to do auto leveling of the image, https://numpy.org/devdocs/reference/generated/numpy.ndarray.astype.html
def auto_level(image):
    np_image = np.array(image)
    r, g, b = np_image[:, :, 0], np_image[:, :, 1], np_image[:, :, 2]
    
    def auto_level_channel(channel):
        min_val, max_val = channel.min(), channel.max()
        # error when dividing by zero so check if max_val is greater than min_val
        if max_val > min_val:  
            scale = 255.0 / (max_val - min_val)
            channel = ((channel - min_val) * scale).clip(0, 255)
        return channel.astype(np.uint8)

    # Apply auto-leveling to each colour channel
    r = auto_level_channel(r)
    g = auto_level_channel(g)
    b = auto_level_channel(b)

    #end of function saving and displaying the image
    auto_leveled_image = np.dstack((r, g, b))
    auto_leveled_img_pil = Image.fromarray(auto_leveled_image)
    display_side_by_side(image, auto_leveled_img_pil, "Auto Level")
    add_to_history(auto_leveled_img_pil)
    return auto_leveled_img_pil

# Optional Operations tasks:
#Functions for undoing
def add_to_history(image):
    global image_history, current_index
    # remove any "future" history if we're in the middle of the stack
    if current_index < len(image_history) - 1:
        image_history = image_history[:current_index + 1]

    # update by adding new image and updating index
    image_history.append(image)
    current_index += 1

def undo():
    global current_index
    if current_index > 0:
        current_index -= 1
        display_image(image_history[current_index], "Undo")

# Function for redoing changes
def redo():
    global current_index
    if current_index < len(image_history) - 1:
        current_index += 1
        display_image(image_history[current_index], "Redo")

#Function for resizing images
def resize(image):
    def apply_resize():
        try:
            # Get user input for dimensions
            width = int(horizontal_entry.get())
            height = int(vertical_entry.get())
            
            # Create a new blank image with the desired size and load the pixels
            original_width, original_height = image.size
            resized_image = Image.new("RGB", (width, height))
            pixels = resized_image.load()
            
            # resize using nearest-neighbor interpolation, map each (i, j) in the resized image to the nearest pixle in the original image
            for i in range(width):
                for j in range(height):
                    orig_x = int(i * original_width / width)
                    orig_y = int(j * original_height / height)
                    pixels[i, j] = image.getpixel((orig_x, orig_y))
            
            display_image(resized_image, f"Resized to {width}x{height} pixels")
            add_to_history(resized_image)
            resize_window.destroy()

        except ValueError:
            print("Please enter valid integer values for width and height.")

    def cancel_resize():
        resize_window.destroy()

    # Create the resize dialog window for user toi input the new dimensions (added the extra atep of being by the top so user can see the window)
    resize_window = Toplevel(root)
    resize_window.title("Resize")
    resize_window.geometry("300x150")

    # Horizontal label and entry
    horizontal_label = Label(resize_window, text="Horizontal (pixels):")
    horizontal_label.grid(row=0, column=0, padx=10, pady=5)
    horizontal_entry = Entry(resize_window)
    horizontal_entry.grid(row=0, column=1, padx=10, pady=5)

    # Vertical label and entry
    vertical_label = Label(resize_window, text="Vertical (pixels):")
    vertical_label.grid(row=1, column=0, padx=10, pady=5)
    vertical_entry = Entry(resize_window)
    vertical_entry.grid(row=1, column=1, padx=10, pady=5)

    # OK and Cancel buttons
    ok_button = Button(resize_window, text="OK", command=apply_resize)
    ok_button.grid(row=2, column=0, padx=10, pady=10)
    cancel_button = Button(resize_window, text="Cancel", command=cancel_resize)
    cancel_button.grid(row=2, column=1, padx=10, pady=10)

#Function for rotating image by 45 degrees
def rotate(image):
    global loaded_image, current_angle
    if loaded_image is None:
        print("No image loaded.")
        return

    # Increment the angle by 45 degrees
    current_angle = (current_angle + 45) % 360
    angle_rad = math.radians(current_angle)

    # change the loaded image to a numpy array for pixel manipulation
    original = np.array(loaded_image)
    original_height, original_width, channels = original.shape

    # Calculate the new canvas size to fit the rotated image
    new_size = int(math.sqrt(original_width**2 + original_height**2))
    rotated_image = np.zeros((new_size, new_size, channels), dtype=np.uint8)

    # Calculate the center of the original and the new image
    original_center_x, original_center_y = original_width // 2, original_height // 2
    new_center_x, new_center_y = new_size // 2, new_size // 2

    # Rotate each pixel manually
    for y in range(original_height):
        for x in range(original_width):
            # Calculate the new coords using rotation matrix
            new_x = int((x - original_center_x) * math.cos(angle_rad) - (y - original_center_y) * math.sin(angle_rad) + new_center_x)
            new_y = int((x - original_center_x) * math.sin(angle_rad) + (y - original_center_y) * math.cos(angle_rad) + new_center_y)

            # Check if the new coords are within the bounds of the new image
            if 0 <= new_x < new_size and 0 <= new_y < new_size:
                rotated_image[new_y, new_x] = original[y, x]

    rotated_img_pil = Image.fromarray(rotated_image)
    add_to_history(rotated_img_pil)
    display_image(rotated_img_pil, f"Rotated by {current_angle} degrees")

# Function for flipping image (horizontally)
def y_flip_img(image):
    global loaded_image
    if loaded_image is None:
        print("No image loaded.")
        return

    original = np.array(loaded_image)
    flipped_image = np.zeros(original.shape, dtype=np.uint8)
    
    # Flip along the y-axis (horizontal flip)
    for y in range(original.shape[0]):
        for x in range(original.shape[1]):
            flipped_image[y, original.shape[1] - x - 1] = original[y, x]

    loaded_image = Image.fromarray(flipped_image)
    add_to_history(loaded_image)
    display_image(loaded_image, "Flipped Horizontally")

# Function for flipping image vertically (along the x-axis)
def x_flip_img(image):
    global loaded_image
    if loaded_image is None:
        print("No image loaded.")
        return

    original = np.array(loaded_image)
    flipped_image = np.zeros(original.shape, dtype=np.uint8)
    
    # Flip along the x-axis (vertical flip)
    for y in range(original.shape[0]):
        for x in range(original.shape[1]):
            flipped_image[original.shape[0] - y - 1, x] = original[y, x]

    # Update loaded_image and display the flipped image
    loaded_image = Image.fromarray(flipped_image)
    add_to_history(loaded_image)
    display_image(loaded_image, "Flipped Vertically")
    

#Funtion for adjusting contrast of image (Gamma)
def contrast(image):
    def update_contrast(val):
        # Contrast value from slider
        factor = float(val)
        midpoint = 128 
        
        # Convert image to a NumPy array and normalize it to [0, 1]
        original = np.array(image).astype(float) / 255
        contrasted_image = ((original - 0.5) * factor + 0.5) * 255
        contrasted_image = np.clip(contrasted_image, 0, 255).astype(np.uint8)
        
        # Display the adjusted image
        display_image(Image.fromarray(contrasted_image), "Contrast Adjustment")
        add_to_history(Image.fromarray(contrasted_image))

    def remove_slider(event=None):
        slider.pack_forget()

    # Display the original image and add a slider for contrast adjustment
    display_image(image, "Contrast Adjustment")
    slider = tk.Scale(root, from_=0.5, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, command=update_contrast, label="Adjust Contrast")
    slider.set(1.0)
    slider.pack()

    # Bind Enter key to remove the slider
    root.bind('<Return>', remove_slider)

# Function for saturation adjustment using a slider like photoshop
def adjust_saturation(image):
    def update_saturation(val):
        factor = float(val) / 100
        original = np.array(image).astype(float)

        grayscale = np.dot(original[..., :3], [0.299, 0.587, 0.114])
        grayscale = np.stack([grayscale] * 3, axis=-1)
        
        # Adjust the saturation of the image
        saturated_image = original + (original - grayscale) * factor
        saturated_image = np.clip(saturated_image, 0, 255).astype(np.uint8)
        
        display_image(Image.fromarray(saturated_image), "Saturation Adjustment")
        add_to_history(Image.fromarray(saturated_image))

    def remove_slider(event=None):
        slider.pack_forget()

    # Display the original image and add a slider for saturation adjustment
    display_image(image, "Saturation Adjustment")
    slider = tk.Scale(root, from_=-100, to=100, orient=tk.HORIZONTAL, command=update_saturation, label="Adjust Saturation")
    slider.set(0)
    slider.pack()

    # Bind Enter key to remove the slider (like actual photoshop)
    root.bind('<Return>', remove_slider)

# Function to save the current image from history (FOR REPORT! Uses the PIL library)
def save_current_image():
    if current_index < 0 or current_index >= len(image_history):
        print("No image available to save.")
        return

    # Get the current image from history
    current_image = image_history[current_index]

    # Open a file dialog for the user to choose the save location and file format
    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("BMP files", "*.bmp"), ("All files", "*.*")]
    )

    if file_path:
        # Save the image in the selected format
        current_image.save(file_path)
        print(f"Image saved to {file_path}")


# Main GUI setup
root = tk.Tk()  # Create the main window
root.title('Mini Faux-toshop') 
root.geometry('1200x800') 
menu = Menu(root)
root.config(menu=menu) 

#All working!!! core ops done!
# Core Operations menu:
core_menu = Menu(menu, tearoff=0)  
menu.add_cascade(label="Core Operations", menu=core_menu)  
core_menu.add_command(label="Open File", command=lambda: open_bmp(label)) 
core_menu.add_command(label="Grayscale", command=lambda: display_side_by_side(loaded_image, grayscale(loaded_image), "Grayscale"))
core_menu.add_command(label="Ordered Dithering", command=lambda: display_side_by_side(grayscale(loaded_image), ordered_dithering(loaded_image), "Ordered Dithering")) #dither option
core_menu.add_command(label="Auto Level", command=lambda: auto_level(loaded_image)) 
core_menu.add_command(label="Exit", command=exit_app) 

# Optional Operations menu:
optional_menu = Menu(menu,tearoff=0)
menu.add_cascade(label="Optional Operations", menu=optional_menu)
optional_menu.add_command(label="Undo", command=undo)
optional_menu.add_command(label="Redo", command=redo)
optional_menu.add_command(label="Resize", command=lambda:resize(loaded_image))
optional_menu.add_command(label="Rotate 45 degrees", command=lambda:rotate(loaded_image))
optional_menu.add_command(label="Flip Vertically", command=lambda:x_flip_img(loaded_image))
optional_menu.add_command(label="Flip Horizontally", command=lambda:y_flip_img(loaded_image))
optional_menu.add_command(label="Saturation", command=lambda:adjust_saturation(loaded_image))
optional_menu.add_command(label="Contrast", command=lambda:contrast(loaded_image))

#for saving the image for report
report_menu = Menu(menu, tearoff=0)
menu.add_cascade(label="Report Operations", menu=report_menu)
report_menu.add_command(label="Save Current Image", command=save_current_image)

label = Label(root) 
label.pack()
root.mainloop()  
