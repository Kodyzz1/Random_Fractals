import numpy as np

# Replace 'zoom_path.npz' with the actual name of your path file if it's different
path_file = 'zoom_path.npz'

try:
    path_data = np.load(path_file)
    path = path_data['path']

    print(f"Shape of the path array: {path.shape}")
    print("\nFirst 5 entries in the path:")
    for i in range(min(5, len(path))):
        print(f"Frame {i}: Center X = {path[i][0]}, Center Y = {path[i][1]}, Zoom = {path[i][2]}")

except FileNotFoundError:
    print(f"Error: The file '{path_file}' was not found. Make sure the filename is correct and the file is in the same directory as you run this script.")
except Exception as e:
    print(f"An error occurred: {e}")