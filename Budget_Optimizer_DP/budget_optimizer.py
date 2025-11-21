import json
import os


class Item:
    def __init__(self, name, cost, value):
        self.name = name
        self.cost = cost
        self.value = value


class BudgetOptimizer:
    def __init__(self):
        self.items = []
        self.budget = 0
        self.dp_table = []
        self.selected_items = []

    def add_item(self, name, cost, value):
        self.items.append(Item(name, cost, value))

    def set_budget(self, budget):
        if budget < 0:
            budget = 0
        self.budget = budget

    def compute(self):
        n = len(self.items)
        W = self.budget
        self.dp_table = [[0] * (W + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            cost = self.items[i - 1].cost
            value = self.items[i - 1].value
            for w in range(0, W + 1):
                if cost > w:
                    self.dp_table[i][w] = self.dp_table[i - 1][w]
                else:
                    without_item = self.dp_table[i - 1][w]
                    with_item = self.dp_table[i - 1][w - cost] + value
                    self.dp_table[i][w] = max(without_item, with_item)
        self.selected_items = self._reconstruct_solution()

    def _reconstruct_solution(self):
        result = []
        n = len(self.items)
        W = self.budget
        i = n
        w = W
        while i > 0 and w >= 0:
            if self.dp_table[i][w] != self.dp_table[i - 1][w]:
                item = self.items[i - 1]
                result.append(item)
                w -= item.cost
            i -= 1
        result.reverse()
        return result

    def get_max_value(self):
        if not self.dp_table:
            return 0
        return self.dp_table[len(self.items)][self.budget]

    def get_total_cost_of_selected(self):
        return sum(item.cost for item in self.selected_items)

    def get_dp_table(self):
        return self.dp_table
    
    def load_products_from_file(self, filename="products.json"):
        """Load products from a JSON file."""
        if not os.path.exists(filename):
            return False
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                products = json.load(f)
                for product in products:
                    self.add_item(product['name'], product['cost'], product['value'])
                return True
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error loading products: {e}")
            return False


class TextInterface:
    def __init__(self):
        self.optimizer = BudgetOptimizer()
        # Load default products
        if self.optimizer.load_products_from_file():
            print("Default products loaded from products.json")

    def run(self):
        while True:
            self.print_menu()
            choice = input("Enter choice: ").strip()
            if choice == "1":
                self.handle_add_item()
            elif choice == "2":
                self.handle_set_budget()
            elif choice == "3":
                self.handle_compute()
            elif choice == "4":
                self.handle_display_dp_table()
            elif choice == "0":
                print("Exiting...")
                break
            else:
                print("Invalid choice. Try again.")

    def print_menu(self):
        print()
        print("===== Dynamic Programming Budget Optimizer =====")
        print("Current budget:", self.optimizer.budget)
        print("Items:")
        if not self.optimizer.items:
            print("  (no items)")
        else:
            for idx, item in enumerate(self.optimizer.items, start=1):
                print(f"  {idx}. {item.name} | Cost: {item.cost} | Value: {item.value}")
        print("-----------------------------------------------")
        print("1. Add Item")
        print("2. Set Budget")
        print("3. Compute Optimal Selection")
        print("4. Display DP Table")
        print("0. Exit")

    def handle_add_item(self):
        name = input("Item name: ").strip()
        if not name:
            print("Name cannot be empty.")
            return
        try:
            cost = int(input("Item cost (integer): ").strip())
            value = int(input("Item value (integer): ").strip())
        except ValueError:
            print("Cost and value must be integers.")
            return
        if cost < 0 or value < 0:
            print("Cost and value must be non-negative.")
            return
        self.optimizer.add_item(name, cost, value)
        print("Item added.")

    def handle_set_budget(self):
        try:
            budget = int(input("Set budget (integer): ").strip())
        except ValueError:
            print("Budget must be integer.")
            return
        if budget < 0:
            print("Budget cannot be negative.")
            return
        self.optimizer.set_budget(budget)
        print("Budget set.")

    def handle_compute(self):
        if self.optimizer.budget <= 0:
            print("Set budget first.")
            return
        if not self.optimizer.items:
            print("Add at least one item.")
            return
        self.optimizer.compute()
        max_value = self.optimizer.get_max_value()
        selected = self.optimizer.selected_items
        total_cost = self.optimizer.get_total_cost_of_selected()
        print()
        print("===== Optimal Selection Result =====")
        print("Max total value:", max_value)
        print("Total cost:", total_cost)
        if not selected:
            print("No items selected.")
        else:
            print("Selected items:")
            for item in selected:
                print(f"- {item.name} (Cost: {item.cost}, Value: {item.value})")

    def handle_display_dp_table(self):
        if not self.optimizer.dp_table:
            print("No DP table. Compute optimal selection first.")
            return
        table = self.optimizer.get_dp_table()
        W = self.optimizer.budget
        header = ["i/w"]
        for w in range(W + 1):
            header.append(str(w))
        print()
        print("DP table (rows: items, columns: budget):")
        print(" | ".join(header))
        print("-" * (4 * (W + 2)))
        for i in range(len(table)):
            row_values = [str(i)]
            for w in range(W + 1):
                row_values.append(str(table[i][w]))
            print(" | ".join(row_values))


import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class GUIInterface:
    def __init__(self):
        self.optimizer = BudgetOptimizer()
        # Load default products
        self.optimizer.load_products_from_file()
        
        self.root = tk.Tk()
        self.root.title("Dynamic Programming Budget Optimizer")
        self.root.geometry("1200x800")
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Left panel - Input section
        left_frame = ttk.LabelFrame(main_frame, text="Input Data", padding="10")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Budget input
        budget_frame = ttk.Frame(left_frame)
        budget_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(budget_frame, text="Budget:").grid(row=0, column=0, padx=5)
        self.budget_var = tk.StringVar(value="0")
        budget_entry = ttk.Entry(budget_frame, textvariable=self.budget_var, width=15)
        budget_entry.grid(row=0, column=1, padx=5)
        ttk.Button(budget_frame, text="Set Budget", command=self.set_budget).grid(row=0, column=2, padx=5)
        
        # Items table
        items_label = ttk.Label(left_frame, text="Items:")
        items_label.grid(row=1, column=0, sticky=tk.W, pady=(10, 5))
        
        # Treeview for items
        tree_frame = ttk.Frame(left_frame)
        tree_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        columns = ("Name", "Cost", "Value")
        self.items_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=8)
        self.items_tree.heading("Name", text="Name")
        self.items_tree.heading("Cost", text="Cost")
        self.items_tree.heading("Value", text="Value")
        self.items_tree.column("Name", width=150)
        self.items_tree.column("Cost", width=80)
        self.items_tree.column("Value", width=80)
        
        scrollbar_items = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.items_tree.yview)
        self.items_tree.configure(yscrollcommand=scrollbar_items.set)
        self.items_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar_items.grid(row=0, column=1, sticky=(tk.N, tk.S))
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        
        # Add item section
        add_frame = ttk.LabelFrame(left_frame, text="Add New Item", padding="5")
        add_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(add_frame, text="Name:").grid(row=0, column=0, padx=2, pady=2)
        self.name_var = tk.StringVar()
        ttk.Entry(add_frame, textvariable=self.name_var, width=15).grid(row=0, column=1, padx=2, pady=2)
        
        ttk.Label(add_frame, text="Cost:").grid(row=0, column=2, padx=2, pady=2)
        self.cost_var = tk.StringVar()
        ttk.Entry(add_frame, textvariable=self.cost_var, width=10).grid(row=0, column=3, padx=2, pady=2)
        
        ttk.Label(add_frame, text="Value:").grid(row=0, column=4, padx=2, pady=2)
        self.value_var = tk.StringVar()
        ttk.Entry(add_frame, textvariable=self.value_var, width=10).grid(row=0, column=5, padx=2, pady=2)
        
        ttk.Button(add_frame, text="Add Item", command=self.add_item).grid(row=0, column=6, padx=5, pady=2)
        ttk.Button(add_frame, text="Delete Selected", command=self.delete_item).grid(row=1, column=6, padx=5, pady=2)
        
        # Buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.grid(row=4, column=0, pady=10)
        ttk.Button(button_frame, text="Compute Optimal Selection", command=self.compute_optimal).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Show Graph", command=self.show_graph).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Reset", command=self.reset).grid(row=0, column=2, padx=5)
        
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(2, weight=1)
        
        # Right panel - Output section
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Results section
        results_frame = ttk.LabelFrame(right_frame, text="Results", padding="10")
        results_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=10, width=40, wrap=tk.WORD)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # DP Table section
        dp_frame = ttk.LabelFrame(right_frame, text="DP Table", padding="10")
        dp_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.dp_text = scrolledtext.ScrolledText(dp_frame, height=12, width=40, wrap=tk.NONE, font=("Courier", 9))
        self.dp_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        dp_frame.columnconfigure(0, weight=1)
        dp_frame.rowconfigure(0, weight=1)
        
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)
        
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(0, weight=1)
        
        # Update display
        self.update_items_display()
        
    def set_budget(self):
        try:
            budget = int(self.budget_var.get())
            if budget < 0:
                messagebox.showerror("Error", "Budget cannot be negative.")
                return
            self.optimizer.set_budget(budget)
            messagebox.showinfo("Success", f"Budget set to {budget}")
        except ValueError:
            messagebox.showerror("Error", "Budget must be an integer.")
    
    def add_item(self):
        name = self.name_var.get().strip()
        if not name:
            messagebox.showerror("Error", "Item name cannot be empty.")
            return
        try:
            cost = int(self.cost_var.get())
            value = int(self.value_var.get())
            if cost < 0 or value < 0:
                messagebox.showerror("Error", "Cost and value must be non-negative.")
                return
            self.optimizer.add_item(name, cost, value)
            self.name_var.set("")
            self.cost_var.set("")
            self.value_var.set("")
            self.update_items_display()
        except ValueError:
            messagebox.showerror("Error", "Cost and value must be integers.")
    
    def delete_item(self):
        selected = self.items_tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select an item to delete.")
            return
        for item_id in selected:
            item = self.items_tree.item(item_id)
            name = item['values'][0]
            # Find and remove from optimizer
            for i, opt_item in enumerate(self.optimizer.items):
                if opt_item.name == name:
                    self.optimizer.items.pop(i)
                    break
        self.update_items_display()
    
    def update_items_display(self):
        # Clear tree
        for item in self.items_tree.get_children():
            self.items_tree.delete(item)
        # Add items
        for item in self.optimizer.items:
            self.items_tree.insert("", tk.END, values=(item.name, item.cost, item.value))
    
    def compute_optimal(self):
        if self.optimizer.budget <= 0:
            messagebox.showerror("Error", "Please set a budget first.")
            return
        if not self.optimizer.items:
            messagebox.showerror("Error", "Please add at least one item.")
            return
        
        self.optimizer.compute()
        max_value = self.optimizer.get_max_value()
        selected = self.optimizer.selected_items
        total_cost = self.optimizer.get_total_cost_of_selected()
        
        # Display results
        self.results_text.delete(1.0, tk.END)
        result_str = "===== Optimal Selection Result =====\n\n"
        result_str += f"Maximum Total Value: {max_value}\n"
        result_str += f"Total Cost: {total_cost}\n"
        result_str += f"Budget Used: {total_cost} / {self.optimizer.budget}\n\n"
        
        if not selected:
            result_str += "No items selected.\n"
        else:
            result_str += "Selected Items:\n"
            for item in selected:
                result_str += f"  • {item.name} (Cost: {item.cost}, Value: {item.value})\n"
        
        self.results_text.insert(1.0, result_str)
        
        # Display DP table
        self.display_dp_table()
    
    def display_dp_table(self):
        if not self.optimizer.dp_table:
            self.dp_text.delete(1.0, tk.END)
            self.dp_text.insert(1.0, "No DP table available. Compute optimal selection first.")
            return
        
        table = self.optimizer.get_dp_table()
        W = self.optimizer.budget
        
        # Build table string
        table_str = "DP Table (rows: items, columns: budget):\n\n"
        
        # Header
        header = "i\\w |"
        for w in range(min(W + 1, 20)):  # Limit display to 20 columns for readability
            header += f" {w:>4} |"
        if W + 1 > 20:
            header += " ..."
        table_str += header + "\n"
        table_str += "-" * len(header) + "\n"
        
        # Rows
        for i in range(len(table)):
            row_str = f"{i:>3} |"
            for w in range(min(W + 1, 20)):
                row_str += f" {table[i][w]:>4} |"
            if W + 1 > 20:
                row_str += " ..."
            table_str += row_str + "\n"
        
        if W + 1 > 20:
            table_str += f"\n(Table truncated. Full table has {W + 1} columns.)\n"
        
        self.dp_text.delete(1.0, tk.END)
        self.dp_text.insert(1.0, table_str)
    
    def show_graph(self):
        if not MATPLOTLIB_AVAILABLE:
            messagebox.showerror("Error", "Matplotlib is not installed. Please install it using: pip install matplotlib")
            return
        
        if not self.optimizer.items:
            messagebox.showwarning("Warning", "Please add items first.")
            return
        
        # Create a new window for the graph
        graph_window = tk.Toplevel(self.root)
        graph_window.title("Value vs Cost Trade-off")
        graph_window.geometry("600x500")
        
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Plot all items
        costs = [item.cost for item in self.optimizer.items]
        values = [item.value for item in self.optimizer.items]
        names = [item.name for item in self.optimizer.items]
        
        # Plot all items
        ax.scatter(costs, values, color='blue', s=100, alpha=0.6, label='All Items')
        
        # Highlight selected items if computation was done
        if self.optimizer.selected_items:
            selected_costs = [item.cost for item in self.optimizer.selected_items]
            selected_values = [item.value for item in self.optimizer.selected_items]
            selected_names = [item.name for item in self.optimizer.selected_items]
            ax.scatter(selected_costs, selected_values, color='red', s=150, 
                      marker='*', label='Selected Items', zorder=5)
            
            # Add annotations for selected items
            for i, name in enumerate(selected_names):
                ax.annotate(name, (selected_costs[i], selected_values[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Add annotations for all items
        for i, name in enumerate(names):
            if not self.optimizer.selected_items or self.optimizer.items[i] not in self.optimizer.selected_items:
                ax.annotate(name, (costs[i], values[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=7, alpha=0.7)
        
        ax.set_xlabel('Cost', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_title('Value vs Cost Trade-off', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        canvas = FigureCanvasTkAgg(fig, graph_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add close button
        ttk.Button(graph_window, text="Close", command=graph_window.destroy).pack(pady=5)
    
    def reset(self):
        self.optimizer = BudgetOptimizer()
        self.budget_var.set("0")
        self.name_var.set("")
        self.cost_var.set("")
        self.value_var.set("")
        self.update_items_display()
        self.results_text.delete(1.0, tk.END)
        self.dp_text.delete(1.0, tk.END)
        messagebox.showinfo("Reset", "All data has been reset.")
    
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--gui":
        ui = GUIInterface()
        ui.run()
    else:
        ui = TextInterface()
        ui.run()
