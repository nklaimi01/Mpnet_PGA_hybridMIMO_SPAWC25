import matplotlib.colors as mcolors
import numpy as np
 
def generate_cmap(nb_plots:int):
    # #Generate colors
    # green_hex = '#2ca02c'
    # yellow_hex = '#ffd500'
    # red_hex = '#d62728'
 
    # #Convert hexadecimal to RGB values (0-1 range)
    # green = mcolors.hex2color(green_hex)
    # yellow = mcolors.hex2color(yellow_hex)
    # red = mcolors.hex2color(red_hex)
 
    # colors = [green, yellow, red]
 
    # Define a broader range of color-blind friendly colors including green
    deep_blue_hex = '#2c7bb6'      # Deep Blue
    turquoise_hex = '#00a6ca'      # Turquoise
    green_hex = '#2ca02c'          # Green
    yellow_hex = '#ffd500'         # Yellow
    orange_hex = '#fdae61'         # Soft Orange
    dark_purple_hex = '#d7191c'    # Dark Purple
 
    # Convert hexadecimal to RGB values (0-1 range)
    deep_blue = mcolors.hex2color(deep_blue_hex)
    turquoise = mcolors.hex2color(turquoise_hex)
    green = mcolors.hex2color(green_hex)
    yellow = mcolors.hex2color(yellow_hex)
    orange = mcolors.hex2color(orange_hex)
    dark_purple = mcolors.hex2color(dark_purple_hex)
 
    # Generate the color list for the colormap
    colors = [deep_blue, turquoise, green, yellow, orange, dark_purple]
 
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cm', colors, N=256)
    cmap_sampled = cmap(np.linspace(0,1,nb_plots))
 
    return cmap_sampled