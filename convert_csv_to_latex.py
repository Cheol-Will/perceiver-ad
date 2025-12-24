import pandas as pd
import numpy as np
import sys

def csv_to_latex(csv_file, output_file, caption, label, ours_column):
    """
    Convert CSV file to LaTeX table format matching the provided template.
    
    Parameters:
    -----------
    csv_file : str
        Path to input CSV file
    output_file : str
        Path to output LaTeX txt file
    caption : str
        Table caption
    label : str
        Table label
    ours_column : str
        Name of the column representing "Ours" method
    """
    # Read CSV file
    df = pd.read_csv(csv_file, index_col=0)
    
    # Separate data rows from average rows
    avg_auc_row = None
    avg_rank_row = None
    
    if 'AVG_AUC' in df.index:
        avg_auc_row = df.loc['AVG_AUC']
        df = df.drop('AVG_AUC')
    
    if 'AVG_RANK' in df.index:
        avg_rank_row = df.loc['AVG_RANK']
        df = df.drop('AVG_RANK')
    
    # Get dataset names and method names
    datasets = df.index.tolist()
    methods = df.columns.tolist()
    
    # Define machine learning and deep learning methods
    ml_methods = ['IForest', 'LOF', 'OCSVM', 'ECOD', 'KNN', 'PCA']
    dl_methods = [m for m in methods if m not in ml_methods and m != ours_column]
    
    # Reorder columns: ML methods + DL methods + Ours
    ordered_methods = []
    for m in ml_methods:
        if m in methods:
            ordered_methods.append(m)
    for m in dl_methods:
        ordered_methods.append(m)
    if ours_column in methods:
        ordered_methods.append(ours_column)
    
    df = df[ordered_methods]
    if avg_auc_row is not None:
        avg_auc_row = avg_auc_row[ordered_methods]
    if avg_rank_row is not None:
        avg_rank_row = avg_rank_row[ordered_methods]
    
    # Start building LaTeX
    latex_lines = []
    latex_lines.append("\\begin{table}[ht]")
    latex_lines.append("\\centering")
    latex_lines.append(f"\\caption{{{caption}}}")
    latex_lines.append(f"\\label{{{label}}}")
    latex_lines.append("\\vspace{-6pt}")
    latex_lines.append("\\resizebox{\\textwidth}{!}{%")
    
    # Table header
    num_ml = sum(1 for m in ordered_methods if m in ml_methods)
    num_dl = len(ordered_methods) - num_ml
    
    col_spec = "l" + "c" * len(ordered_methods)
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\toprule")
    
    # Multi-column header
    header1 = "& \\multicolumn{" + str(num_ml) + "}{c}{Machine Learning} & \\multicolumn{" + str(num_dl) + "}{c}{Deep Learning}"
    latex_lines.append(header1)
    latex_lines.append("\\\\")
    
    # Cmidrule
    cmidrule = f"\\cmidrule(lr){{2-{num_ml+1}}} \\cmidrule(lr){{{num_ml+2}-{len(ordered_methods)+1}}}"
    latex_lines.append(cmidrule)
    
    # Method names header
    header2_parts = ["Dataset"]
    for m in ordered_methods:
        if m == ours_column:
            header2_parts.append("\\textbf{Ours}")
        else:
            header2_parts.append(m)
    header2 = " & ".join(header2_parts) + " \\\\"
    latex_lines.append(header2)
    latex_lines.append("\\midrule")
    
    # Process each dataset row
    for dataset in datasets:
        row_data = df.loc[dataset]
        
        # Parse values (format: mean ± std)
        values = []
        stds = []
        for val in row_data:
            if pd.isna(val):
                values.append(np.nan)
                stds.append(np.nan)
            else:
                val_str = str(val).strip()
                if '±' in val_str:
                    parts = val_str.split('±')
                    values.append(float(parts[0].strip()))
                    stds.append(float(parts[1].strip()))
                else:
                    values.append(float(val_str))
                    stds.append(0.0)
        
        # Find best and second best
        valid_indices = [i for i, v in enumerate(values) if not np.isnan(v)]
        if len(valid_indices) >= 2:
            sorted_indices = sorted(valid_indices, key=lambda i: values[i], reverse=True)
            best_idx = sorted_indices[0]
            second_best_idx = sorted_indices[1]
        elif len(valid_indices) == 1:
            best_idx = valid_indices[0]
            second_best_idx = -1
        else:
            best_idx = -1
            second_best_idx = -1
        
        # Build row string
        row_parts = [dataset]
        for i, (val, std) in enumerate(zip(values, stds)):
            if np.isnan(val):
                cell_str = "-"
            else:
                cell_str = f"{val:.4f}$^{{\\pm{std:.4f}}}$"
                
                # Apply formatting
                if i == best_idx:
                    cell_str = f"\\textbf{{{cell_str}}}"
                elif i == second_best_idx:
                    cell_str = f"\\underline{{{cell_str}}}"
            
            row_parts.append(cell_str)
        
        row_str = " & ".join(row_parts) + " \\\\"
        latex_lines.append(row_str)
    
    latex_lines.append("\\midrule")
    
    # Average AUC-PR row
    avg_parts = ["Average AUC-PR"]
    
    if avg_auc_row is not None:
        # Use pre-calculated average from CSV
        avg_values = []
        avg_stds = []
        
        for method in ordered_methods:
            val = avg_auc_row[method]
            if pd.notna(val):
                val_str = str(val).strip()
                if '±' in val_str:
                    parts = val_str.split('±')
                    avg_values.append(float(parts[0].strip()))
                    avg_stds.append(float(parts[1].strip()))
                else:
                    avg_values.append(float(val_str))
                    avg_stds.append(0.0)
            else:
                avg_values.append(np.nan)
                avg_stds.append(np.nan)
    else:
        # Calculate average if not provided
        avg_values = []
        avg_stds = []
        
        for method in ordered_methods:
            method_data = df[method]
            valid_vals = []
            for val in method_data:
                if pd.notna(val):
                    val_str = str(val).strip()
                    if '±' in val_str:
                        valid_vals.append(float(val_str.split('±')[0].strip()))
                    else:
                        valid_vals.append(float(val_str))
            
            if valid_vals:
                avg_val = np.mean(valid_vals)
                std_val = np.std(valid_vals) / np.sqrt(len(valid_vals))
                avg_values.append(avg_val)
                avg_stds.append(std_val)
            else:
                avg_values.append(np.nan)
                avg_stds.append(np.nan)
    
    # Find best and second best for averages
    valid_avg_indices = [i for i, v in enumerate(avg_values) if not np.isnan(v)]
    if len(valid_avg_indices) >= 2:
        sorted_avg_indices = sorted(valid_avg_indices, key=lambda i: avg_values[i], reverse=True)
        best_avg_idx = sorted_avg_indices[0]
        second_best_avg_idx = sorted_avg_indices[1]
    else:
        best_avg_idx = -1
        second_best_avg_idx = -1
    
    for i, (avg_val, std_val) in enumerate(zip(avg_values, avg_stds)):
        if np.isnan(avg_val):
            cell_str = "-"
        else:
            cell_str = f"{avg_val:.4f}$^{{\\pm{std_val:.4f}}}$"
            
            if i == best_avg_idx:
                cell_str = f"\\textbf{{{cell_str}}}"
            elif i == second_best_avg_idx:
                cell_str = f"\\underline{{{cell_str}}}"
        
        avg_parts.append(cell_str)
    
    avg_row_str = " & ".join(avg_parts) + " \\\\"
    latex_lines.append(avg_row_str)
    
    # Average Rank row
    rank_parts = ["Average Rank"]
    
    if avg_rank_row is not None:
        # Use pre-calculated rank from CSV
        rank_values = []
        for method in ordered_methods:
            val = avg_rank_row[method]
            if pd.notna(val):
                rank_values.append(float(str(val).strip()))
            else:
                rank_values.append(np.nan)
    else:
        # Calculate rank if not provided
        rank_values = []
        for method in ordered_methods:
            method_ranks = df[method].rank(ascending=False, method='average')
            avg_rank = method_ranks.mean()
            rank_values.append(avg_rank)
    
    # Find best and second best ranks (lower is better)
    valid_rank_indices = [i for i, v in enumerate(rank_values) if not np.isnan(v)]
    if len(valid_rank_indices) >= 2:
        sorted_rank_indices = sorted(valid_rank_indices, key=lambda i: rank_values[i])
        best_rank_idx = sorted_rank_indices[0]
        second_best_rank_idx = sorted_rank_indices[1]
    else:
        best_rank_idx = -1
        second_best_rank_idx = -1
    
    for i, rank_val in enumerate(rank_values):
        if np.isnan(rank_val):
            cell_str = "-"
        else:
            # Format rank to match precision in example
            if rank_val == int(rank_val):
                cell_str = f"{int(rank_val)}"
            else:
                cell_str = f"{rank_val:.4f}".rstrip('0').rstrip('.')
            
            if i == best_rank_idx:
                cell_str = f"\\textbf{{{cell_str}}}"
            elif i == second_best_rank_idx:
                cell_str = f"\\underline{{{cell_str}}}"
        
        rank_parts.append(cell_str)
    
    rank_row_str = " & ".join(rank_parts) + " \\\\"
    latex_lines.append(rank_row_str)
    
    # Close table
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("} ")
    latex_lines.append("\\end{table}")
    
    # Write to file
    latex_output = "\n".join(latex_lines)
    with open(output_file, 'w') as f:
        f.write(latex_output)
    
    print(f"LaTeX table written to {output_file}")
    return latex_output


# Example usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python csv_to_latex.py <metric>")
        print("metric: aucpr, aucroc, or f1")
        sys.exit(1)
    
    metric = sys.argv[1]
    if metric == 'aucpr':
        csv_file = f"metrics/ratio_1.0_AUCPR_T.csv"
    elif metric == 'aucroc':
        csv_file = f"metrics/ratio_1.0_AUCROC_T.csv"
    elif metric == 'f1':
        csv_file = f"metrics/ratio_1.0_f1_T.csv"
    else:
        raise ValueError(f"Invalid metric: {metric}. Use 'aucpr', 'aucroc', or 'f1'")

    output_file = f"metrics/latex_{metric}.txt"
    
    latex_output = csv_to_latex(
        csv_file, 
        output_file,
        caption="Tabular anomaly detection results in terms of AUC-PR with standard deviation on 20 datasets, compared with baseline models. The rank indicates the relative AUC-PR performance within each dataset. The best results are shown in \\textbf{bold} and the second best in \\underline{underlined}.",
        label="table:std_main_table",
        ours_column="LATTE-patience-tuned"
    )