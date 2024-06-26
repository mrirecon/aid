#!/usr/bin/env python3
# this simple array-viewer is generated by chatgpt and works after slight modifications

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import logging
import multiprocessing

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def read_array_file(name):
    """
    Read a .cfl or .npy file and return the data as a NumPy array.
    """
    try:
        if name.endswith('.npy'):
            array = np.load(name)
            dims = list(array.shape)

        elif name.endswith(''):
            # get dims from .hdr
            with open(name + ".hdr", "r") as h:
                h.readline()  # skip
                l = h.readline()
            dims = [int(i) for i in l.split()]

            # remove singleton dimensions from the end
            n = np.prod(dims)
            dims_prod = np.cumprod(dims)
            dims = dims[:np.searchsorted(dims_prod, n) + 1]

            # load data and reshape into dims
            with open(name + ".cfl", "r") as d:
                a = np.fromfile(d, dtype=np.complex64, count=n)
            array = a.reshape(dims, order='F')
        else:
            raise ValueError("Unsupported file format. Only .cfl and .npy files are supported.")
        # Append dimensions to ensure the array has 12 dimensions
        while len(dims) < 12:
            dims.append(1)
        return array.reshape(dims)
        
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")

class ArrayViewer:
    def __init__(self, root, array):
        self.root = root
        self.array = array
        self.array_shape = array.shape
        self.current_indices = [0] * 12  # Initialize indices for the 12 dimensions
        self.enabled_dims = [False] * 12  # Initialize enabled status for the 12 dimensions
        self.rotation_angle = 0  # Initialize rotation angle
        self.mirror = False  # Initialize mirroring
        self.colormap = 'gray'  # Initialize colormap
        
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.controls_frame = ttk.Frame(main_frame)
        self.controls_frame.pack(side=tk.LEFT)
        
        self.index_vars = []
        self.check_vars = []
        non_singleton_dims = [i for i, dim in enumerate(array.shape) if dim > 1]

        if len(non_singleton_dims) >= 2:
            self.enabled_dims[non_singleton_dims[0]] = True
            self.enabled_dims[non_singleton_dims[1]] = True

        for i in range(12):
            var = tk.IntVar(value=0)
            self.index_vars.append(var)
            
            check_var = tk.BooleanVar(value=self.enabled_dims[i])
            self.check_vars.append(check_var)
            
            combined_frame = ttk.Frame(self.controls_frame)
            combined_frame.grid(row=i, column=0, columnspan=2, pady=5)
            
            check = ttk.Checkbutton(combined_frame, style="Small.TCheckbutton", variable=check_var, command=self.update_view, width=2)
            check.pack(side=tk.LEFT)
            label = ttk.Label(combined_frame, text=f"Dim {i}:")
            label.pack(side=tk.LEFT)
            
            
            spinbox = ttk.Spinbox(self.controls_frame, from_=0, to=array.shape[i] - 1, textvariable=var, command=self.update_view, width=2)
            spinbox.grid(row=i, column=2, columnspan=1)

            label1 = ttk.Label(self.controls_frame, text=f"{self.array_shape[i]}")
            label1.grid(row=i, column=3, columnspan=1, padx=5, pady=5)

        # Add buttons for rotation and mirroring
        rotation_frame = ttk.Frame(self.controls_frame)
        rotation_frame.grid(row=13, column=0, columnspan=4, pady=10)

        rotate_left_button = ttk.Button(rotation_frame, text="Rot L", command=self.rotate_left, width=5)
        rotate_left_button.pack(side=tk.LEFT, padx=2)

        rotate_right_button = ttk.Button(rotation_frame, text="Rot R", command=self.rotate_right, width=5)
        rotate_right_button.pack(side=tk.LEFT, padx=2)

        mirror_button = ttk.Button(rotation_frame, text="Mirror", command=self.mirror_image, width=5)
        mirror_button.pack(side=tk.LEFT, padx=2)

        save_button = ttk.Button(rotation_frame, text="Save", command=self.save_image, width=5)
        save_button.pack(side=tk.LEFT, padx=2)
        
        # Add colormap selection
        colormap_label = ttk.Label(self.controls_frame, text="       Colormap:")
        colormap_label.grid(row=14, column=0, columnspan=1)
        
        self.colormap_var = tk.StringVar(value=self.colormap)
        colormap_combobox = ttk.Combobox(self.controls_frame, textvariable=self.colormap_var, values=plt.colormaps(), width=10)
        colormap_combobox.grid(row=14, column=1, columnspan=1)
        colormap_combobox.bind("<<ComboboxSelected>>", self.update_colormap)
    
        self.figure, self.ax = plt.subplots(1, 1, figsize=(10, 10))
        self.canvas = FigureCanvasTkAgg(self.figure, master=main_frame)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.update_view()

    def get_current_slice(self):
        try:
            slices = []
            enabled_indices = [i for i, enabled in enumerate(self.check_vars) if enabled.get()]
            
            if len(enabled_indices) != 2:
                return None  # Only proceed if exactly two dimensions are enabled
            
            for i in range(12):
                if self.check_vars[i].get():
                    slices.append(slice(None))  # Keep this dimension as a slice
                else:
                    slices.append(self.index_vars[i].get())  # Fix this dimension at its current index
                    
            logging.debug(f"Current slices: {slices}")
            return self.array[tuple(slices)]
        except Exception as e:
            logging.error(f"Error in get_current_slice: {e}")
            return None

    def update_view(self, event=None):
        try:
            self.ax.clear()
            current_slice = self.get_current_slice()
            if current_slice is not None and current_slice.ndim == 2:
                image = np.abs(current_slice)
                
                # Apply rotation
                if self.rotation_angle != 0:
                    image = np.rot90(image, self.rotation_angle // 90)
                
                # Apply mirroring
                if self.mirror:
                    image = np.fliplr(image)
                
                self.ax.imshow(image, cmap=self.colormap)
                self.ax.axis('off')
                self.figure.tight_layout()
            else:
                self.ax.text(0.5, 0.5, 'Select exactly 2 dimensions to display', horizontalalignment='center', verticalalignment='center')
                self.ax.axis('off')
                self.figure.tight_layout()
            self.canvas.draw()
        except Exception as e:
            logging.error(f"Error in update_view: {e}")

    def rotate_right(self):
        self.rotation_angle = (self.rotation_angle - 90) % 360
        self.update_view()

    def rotate_left(self):
        self.rotation_angle = (self.rotation_angle + 90) % 360
        self.update_view()

    def mirror_image(self):
        self.mirror = not self.mirror
        self.update_view()

    def update_colormap(self, event):
        self.colormap = self.colormap_var.get()
        self.update_view()

    def save_image(self):
        try:
            current_slice = self.get_current_slice()
            if current_slice is not None and current_slice.ndim == 2:
                image = np.abs(current_slice)

                # Apply rotation
                if self.rotation_angle != 0:
                    image = np.rot90(image, self.rotation_angle // 90)

                # Apply mirroring
                if self.mirror:
                    image = np.fliplr(image)

                filename = tk.filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
                if filename:
                    plt.imsave(filename, image, cmap=self.colormap)
        except Exception as e:
            logging.error(f"Error in save_image: {e}")

def start_viewer(file_path):
    root = tk.Tk()
    root.title("Array Viewer")
    root.geometry("1000x700")
    
    array = read_array_file(file_path)
    app = ArrayViewer(root, array)
    
    root.protocol("WM_DELETE_WINDOW", root.quit)  # Quit the main loop when the window is closed
    root.mainloop()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python viewer.py path_to_file1 [path_to_file2 ...]")
    else:
        processes = []
        for file_path in sys.argv[1:]:
            process = multiprocessing.Process(target=start_viewer, args=(file_path,))
            process.start()
            processes.append(process)
        
        # Wait for all processes to finish
        for process in processes:
            process.join()
