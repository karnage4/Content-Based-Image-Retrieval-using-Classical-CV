import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Add parent dir to path to import local scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_index import build_and_save_index
from scripts.retrieve import retrieve

class CBIRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Assignment 4 - CBIR Search Engine")
        self.root.geometry("500x400")
        
        self.query_path = None
        
        # --- UI Elements ---
        
        # 1. Dataset Selection
        tk.Label(root, text="Select Dataset:").pack(pady=(10, 0))
        self.dataset_var = tk.StringVar(value="paris")
        self.dataset_dropdown = ttk.Combobox(root, textvariable=self.dataset_var, state="readonly")
        self.dataset_dropdown['values'] = ("paris", "food-101")
        self.dataset_dropdown.pack(pady=5)
        
        # 2. Feature Selection
        tk.Label(root, text="Select Feature Extraction Method:").pack(pady=(10, 0))
        self.method_var = tk.StringVar(value="lbp")
        self.method_dropdown = ttk.Combobox(root, textvariable=self.method_var, state="readonly")
        self.method_dropdown['values'] = ("color_hist", "lbp", "hog", "color_sift_bovw")
        self.method_dropdown.pack(pady=5)
        
        # 3. Metric Selection
        tk.Label(root, text="Select Distance Metric:").pack(pady=(10, 0))
        self.metric_var = tk.StringVar(value="chi2")
        self.metric_dropdown = ttk.Combobox(root, textvariable=self.metric_var, state="readonly")
        self.metric_dropdown['values'] = ("euclidean", "cosine", "chi2", "manhattan")
        self.metric_dropdown.pack(pady=5)
        
        # 4. ROI Checkbox for Part B
        self.roi_var = tk.BooleanVar(value=False)
        self.roi_checkbox = tk.Checkbutton(root, text="Enable Bounding Box Selection (Part B)", variable=self.roi_var)
        self.roi_checkbox.pack(pady=10)
        
        # 5. File Picker
        self.file_btn = tk.Button(root, text="Select Query Image", command=self.pick_file)
        self.file_btn.pack(pady=10)
        
        self.file_label = tk.Label(root, text="No image selected", fg="gray")
        self.file_label.pack(pady=5)
        
        # 6. Execute Search
        self.search_btn = tk.Button(root, text="Search Database", bg="lightblue", font=("Arial", 12, "bold"), command=self.execute_search)
        self.search_btn.pack(pady=20)

    def pick_file(self):
        initial_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        path = filedialog.askopenfilename(initialdir=initial_dir, title="Select Query Image",
                                          filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*")))
        if path:
            self.query_path = path
            self.file_label.config(text=f"...{path[-35:]}", fg="black")

    def execute_search(self):
        if not self.query_path:
            messagebox.showwarning("Warning", "Please select a query image first!")
            return
            
        dataset = self.dataset_var.get()
        method = self.method_var.get()
        metric = self.metric_var.get()
        use_roi = self.roi_var.get()
        
        print(f"--- Starting Search Configuration ---")
        print(f"Dataset: {dataset} | Method: {method} | Metric: {metric} | ROI: {use_roi}")
        print(f"Query: {self.query_path}")
        
        # Overwrite/Bind index with the specifically requested user metric dynamically!
        # Because fitting a matrix takes less than 1 second, we can afford to dynamically re-bind the metric upon search.
        try:
            print(f"Dynamically loading and binding {metric} metric mathematically...")
            build_and_save_index(dataset, method, metric=metric)
            
            # Fire Retrieval Search (Which leverages Matplotlib to popup the images natively!)
            retrieve(dataset, method, self.query_path, use_roi=use_roi)
            
        except Exception as e:
            messagebox.showerror("Execution Error", f"An error occurred:\n\n{str(e)}\n\nDid you make sure to run extract_features for {method} on {dataset}?")

if __name__ == "__main__":
    root = tk.Tk()
    app = CBIRApp(root)
    root.mainloop()
