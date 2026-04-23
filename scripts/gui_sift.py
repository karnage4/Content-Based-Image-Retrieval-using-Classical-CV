import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Add parent dir to path to import local scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.visualize_sift import draw_single_image_keypoints, draw_image_matches

class SiftGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Assignment 4 - SIFT Visualizer (Report Generator)")
        self.root.geometry("500x350")
        
        self.img1_path = None
        self.img2_path = None
        
        # --- UI Elements ---
        tk.Label(root, text="SIFT Feature & Match Visualizer", font=("Arial", 14, "bold")).pack(pady=(10, 20))
        
        # Image 1 Selection
        self.btn_img1 = tk.Button(root, text="Select Image 1 (Required)", width=30, command=self.pick_img1)
        self.btn_img1.pack(pady=5)
        
        self.label_img1 = tk.Label(root, text="No image selected", fg="gray")
        self.label_img1.pack(pady=(0, 15))
        
        # Image 2 Selection
        self.btn_img2 = tk.Button(root, text="Select Image 2 (For Matching)", width=30, command=self.pick_img2)
        self.btn_img2.pack(pady=5)
        
        self.label_img2 = tk.Label(root, text="No image selected", fg="gray")
        self.label_img2.pack(pady=(0, 20))
        
        # Action Buttons
        frame_actions = tk.Frame(root)
        frame_actions.pack(pady=10)
        
        self.btn_show_kp = tk.Button(frame_actions, text="1. Visualize Keypoints (Img 1)", bg="lightblue", 
                                     command=self.show_keypoints)
        self.btn_show_kp.pack(side=tk.LEFT, padx=10)
        
        self.btn_show_match = tk.Button(frame_actions, text="2. Visualize Matches (Img 1 -> Img 2)", bg="lightgreen", 
                                        command=self.show_matches)
        self.btn_show_match.pack(side=tk.LEFT, padx=10)

    def _get_base_dir(self):
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

    def pick_img1(self):
        path = filedialog.askopenfilename(initialdir=self._get_base_dir(), title="Select Image 1",
                                          filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*")))
        if path:
            self.img1_path = path
            self.label_img1.config(text=f"...{path[-40:]}", fg="black")

    def pick_img2(self):
        path = filedialog.askopenfilename(initialdir=self._get_base_dir(), title="Select Image 2",
                                          filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*")))
        if path:
            self.img2_path = path
            self.label_img2.config(text=f"...{path[-40:]}", fg="black")

    def show_keypoints(self):
        if not self.img1_path:
            messagebox.showwarning("Warning", "Please select Image 1 first!")
            return
            
        print("Visualizing keypoints for Image 1...")
        try:
            draw_single_image_keypoints(self.img1_path, out_path=None)
        except Exception as e:
            messagebox.showerror("Error", f"Could not visualize keypoints:\n{str(e)}")

    def show_matches(self):
        if not self.img1_path or not self.img2_path:
            messagebox.showwarning("Warning", "Please select BOTH Image 1 and Image 2 for matching!")
            return
            
        print("Visualizing structural matches...")
        try:
            draw_image_matches(self.img1_path, self.img2_path, out_path=None)
        except Exception as e:
            messagebox.showerror("Error", f"Could not generate matches:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SiftGUI(root)
    root.mainloop()
