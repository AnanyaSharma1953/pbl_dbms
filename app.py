import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import duckdb
import os

class SQLQueryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI QUERY ANALYSER")
        self.root.geometry("800x600")

        # --- Apply a modern theme and a professional Natural Hues color scheme ---
        self.style = ttk.Style(self.root)
        self.style.theme_use('clam')

        # Define the Natural Hues color palette
        BG_COLOR = "#F5F5DC"          # Beige (sandy)
        TEXT_COLOR = "#4A442D"        # Dark Olive/Brown
        BUTTON_BG = "#8FBC8F"         # Dark Sea Green
        BUTTON_FG = "#FFFFFF"         # White
        BUTTON_ACTIVE_BG = "#556B2F"  # Dark Olive Green
        self.LABEL_ERROR_FG = "#A52A2A"    # Brown (earthy red) - Made instance attribute
        self.LABEL_SUCCESS_FG = "#228B22"  # Forest Green - Made instance attribute
        TEXT_AREA_BG = "#FFFFFF"      # White for the text area

        self.root.configure(bg=BG_COLOR)

        # Configure styles for all widgets
        self.style.configure('.', background=BG_COLOR, foreground=TEXT_COLOR, font=('Helvetica', 10))
        self.style.configure('TFrame', background=BG_COLOR)
        self.style.configure('TLabel', background=BG_COLOR)
        self.style.configure('TLabelframe', background=BG_COLOR, borderwidth=1, relief="solid")
        self.style.configure('TLabelframe.Label', background=BG_COLOR, foreground=TEXT_COLOR, font=('Helvetica', 11, 'bold'))
        
        # Style the button with the Natural Hues theme
        self.style.configure('TButton', padding=6, relief="flat", background=BUTTON_BG, foreground=BUTTON_FG, font=('Helvetica', 10, 'bold'))
        self.style.map('TButton', background=[('active', BUTTON_ACTIVE_BG), ('pressed', BUTTON_ACTIVE_BG)])

        self.df = None  # To store the pandas DataFrame
        self.table_name = None # To store the table name from the CSV file

        # --- Main Frame ---
        main_frame = ttk.Frame(root, padding="10 10 10 10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Top Frame for Controls ---
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 5))

        # File Upload Button
        self.load_button = ttk.Button(controls_frame, text="Load CSV File", command=self.load_csv)
        self.load_button.pack(side=tk.LEFT, padx=(0, 10))

        # Status Label for File
        self.file_label = ttk.Label(controls_frame, text="No file loaded", foreground=self.LABEL_ERROR_FG)
        self.file_label.pack(side=tk.LEFT, pady=4)

        # --- Query Input Frame ---
        self.query_frame = ttk.LabelFrame(main_frame, text="Write Your SQL Query", padding="5 5 5 5")
        self.query_frame.pack(fill=tk.X, pady=10)

        self.query_text = tk.Text(self.query_frame, height=8, relief="solid", bd=1, bg=TEXT_AREA_BG, fg=TEXT_COLOR, font=('Consolas', 10))
        self.query_text.pack(fill=tk.X, expand=True)
        self.query_text.insert(tk.END, "SELECT * FROM my_table LIMIT 100;")

        # Run Query Button
        self.run_button = ttk.Button(main_frame, text="Run Query", command=self.run_query)
        self.run_button.pack(fill=tk.X)

        # --- Results Frame ---
        results_frame = ttk.LabelFrame(main_frame, text="Query Result", padding="5 5 5 5")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Style for the Treeview header
        self.style.configure("Treeview.Heading", font=('Helvetica', 10, 'bold'))

        # Treeview for displaying results in a table
        self.tree = ttk.Treeview(results_frame)
        self.tree.pack(fill=tk.BOTH, expand=True)

        # Scrollbars for the Treeview
        vsb = ttk.Scrollbar(self.tree, orient="vertical", command=self.tree.yview)
        vsb.pack(side='right', fill='y')
        hsb = ttk.Scrollbar(self.tree, orient="horizontal", command=self.tree.xview)
        hsb.pack(side='bottom', fill='x')
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)


    def load_csv(self):
        """Opens a file dialog to select a CSV and loads it into a pandas DataFrame."""
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv"), ("All files", "*.*")]
        )
        if not file_path:
            return

        try:
            self.df = pd.read_csv(file_path)
            # Derive table name from filename (e.g., 'students.csv' -> 'students')
            filename = os.path.basename(file_path)
            self.table_name = os.path.splitext(filename)[0]

            # Update UI elements with the new table name
            self.file_label.config(text=f"Loaded: {filename}", foreground=self.LABEL_SUCCESS_FG)
            self.query_frame.config(text=f"2. Write Your SQL Query (use '{self.table_name}' as the table name)")
            
            # Update the example query
            self.query_text.delete("1.0", tk.END)
            self.query_text.insert(tk.END, f"SELECT * FROM {self.table_name} LIMIT 100;")
            
            messagebox.showinfo("Success", f"Successfully loaded {filename}.\n{len(self.df)} rows found.\nTable name is '{self.table_name}'.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")
            self.df = None
            self.table_name = None
            self.file_label.config(text="No file loaded", foreground=self.LABEL_ERROR_FG)

    def run_query(self):
        """Executes the SQL query on the loaded DataFrame using DuckDB."""
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a CSV file first.")
            return

        query = self.query_text.get("1.0", tk.END).strip()
        if not query:
            messagebox.showwarning("Warning", "Query is empty. Please enter a SQL query.")
            return

        # Clear previous results from the treeview
        for i in self.tree.get_children():
            self.tree.delete(i)
        self.tree["columns"] = []

        try:
            # Run the query using DuckDB on the pandas DataFrame
            con = duckdb.connect(database=':memory:', read_only=False)
            # Use the dynamic table name
            con.register(self.table_name, self.df)
            result_df = con.execute(query).fetchdf()

            # Set up the Treeview columns
            self.tree["columns"] = list(result_df.columns)
            self.tree["show"] = "headings"  # Hide the default first column

            # Add column headings
            for col in self.tree["columns"]:
                self.tree.heading(col, text=col)
                self.tree.column(col, width=100) # Set a default column width

            # Add data rows
            for index, row in result_df.iterrows():
                self.tree.insert("", "end", values=list(row))

        except Exception as e:
            messagebox.showerror("Query Error", f"An error occurred:\n{e}")
        finally:
            if 'con' in locals():
                con.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = SQLQueryApp(root)
    root.mainloop()