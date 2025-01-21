import numpy as np

def format_data(file_path):
    # Load the data from the file
    data = np.genfromtxt(file_path, dtype=str, delimiter='\n')
    
    # Define a function to round errors to 1 significant figure
    def round_error(value):
        exponent = np.floor(np.log10(abs(value)))
        factor = 10**exponent
        return round(value / factor) * factor

    formatted_rows = []

    for row in data:
        if not row.strip():
            continue
        
        # Split the row into parameter and its values
        param, values = row.split(':')
        val, plus_err, minus_err = map(float, values.split())
        
        # Round errors to 1 significant figure
        plus_err_rounded = round_error(plus_err)
        minus_err_rounded = round_error(minus_err)
        
        # Determine the number of decimal places
        max_error = max(plus_err_rounded, minus_err_rounded)
        decimal_places = abs(int(np.floor(np.log10(max_error))))
        val_str = f"{val:.{decimal_places}f}"
        plus_err_str = f"{plus_err_rounded:.{decimal_places}f}"
        minus_err_str = f"{minus_err_rounded:.{decimal_places}f}"
        
        formatted_rows.append(f"${val_str}^{{+{plus_err_str}}}_{{-{minus_err_str}}}$")
    
    return ' & '.join(formatted_rows)

# Example usage
file_path = './data/GRB1/GRB1_control_results.txt'  # Replace with your input file path
formatted_string = format_data(file_path)
print(f"\\textbf{{Control}} & {formatted_string} \\\\")
