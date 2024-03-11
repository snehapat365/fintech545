import csv


def read_csv(file_path):
    """
    Read a CSV file containing numbers.
    
    Args:
    - file_path: Path to the CSV file.
    
    Returns:
    - A list of strings containing header cols and list of lists containing the numbers.
    """
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        header = [str(cell) for cell in next(reader)]
        # skip header row
        next(reader)
        for row in reader:
            data.append([float(cell) for cell in row])
    return header, data


def compare_csv_contents(csv1, csv2, tolerance=1e-9):
    """
    Compare two CSV files containing numbers with a specified tolerance.
    
    Args:
    - csv1: List of lists containing numbers from the first CSV file.
    - csv2: List of lists containing numbers from the second CSV file.
    - tolerance: Tolerance for floating-point comparisons.
    
    Returns:
    - True if the CSV files are identical within the specified tolerance, False otherwise.
    """
    if len(csv1) != len(csv2):
        return False

    for row1, row2 in zip(csv1, csv2):
        if len(row1) != len(row2):
            return False

        for num1, num2 in zip(row1, row2):
            if abs(num1 - num2) > tolerance:
                return False

    return True


def check_output(file1, file2):
    csv1_headers, csv1_data = read_csv(file1)
    csv2_headers, csv2_data = read_csv(file2)
    if csv1_headers != csv2_headers:
        print("The CSV files do not have identical headers.")
        return False
    return compare_csv_contents(csv1_data, csv2_data, tolerance=1e-9)


# Example usage:
if __name__ == "__main__":
    file_names = [
        "testout_1.1.csv", # Covariance Missing data, skip missing rows
        "testout_1.2.csv", # Correlation Missing data, skip missing rows
        "testout_1.3.csv", # Covariance Missing data, Pairwise
        "testout_1.4.csv", # Correlation Missing data, pairwise
        "testout_2.1.csv", # EW Covariance, lambda=0.97
        "testout_2.2.csv", # EW Correlation, lambd=0.94
        "testout_2.3.csv", # Covariance with EW Variance (l=0.94), EW Correlation (l=0.97)
        "testout_3.1.csv", # near_psd covariance
        "testout_3.2.csv", # near_psd correlation
        # "testout_3.3.csv", # Higham covariance
        "testout_3.4.csv", # Higham correlation
        # "testout_4.1.csv", # chol_psd
        # "testout_5.1.csv", # Normal Simulation PD Input 0 mean - 100,000 simulations, compare input vs output covariance
        # "testout_5.2.csv", # Normal Simulation PSD Input 0 mean - 100,000 simulations, compare input vs output covariance
        # "testout_5.3.csv", # Normal Simulation nonPSD Input, 0 mean, near_psd fix - 100,000 simulations, compare input vs output covariance
        # "testout_5.4.csv", # Normal Simulation PSD Input, 0 mean, higham fix - 100,000 simulations, compare input vs output covariance
        # "testout_5.5.csv", # PCA Simulation, 99% explained, 0 mean - 100,000 simulations compare input vs output covariance
        "testout7_1.csv", # Fit Normal Distribution
        "testout7_2.csv", # Fit T Distribution
        # "testout7_3.csv", # T Regression
        "testout8_1.csv", # Var from Normal Distribution
        "testout8_2.csv", # Var from T Distribution
        "testout8_3.csv", # VaR from Simulation -- compare to 8.2 values
        "testout8_4.csv", # ES From Normal Distribution
        # "testout8_5.csv", # ES from T Distribution
        "testout8_6.csv", # ES from Simulation -- compare to 8.5 values
        # "testout9_1.csv" # VaR/ES on 2 levels from simulated values - Copula
    ]
    for file in file_names:
        print(file)
        result = check_output("data/output/" + file, "data/expected_output/" + file)
        if result == True:
            print("\033[32mPassed Test " + file + "!\033[0m")
        else:
            print("\033[91mFailed Test " + file + " ...\033[0m")
            
