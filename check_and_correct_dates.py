import pandas as pd

def check_and_correct_dates(file_path, output_path):
    """
    Check and correct the date format in the dataset.

    Args:
    - file_path: Path to the input CSV file.
    - output_path: Path to save the corrected CSV file.

    Returns:
    - None
    """
    try:
        # Load the dataset
        print("Loading dataset...")
        data = pd.read_csv(file_path)

        # Ensure 'date' column exists
        if 'date' not in data.columns:
            raise KeyError("The dataset does not have a 'date' column. Please verify the file.")

        # Attempt to convert the 'date' column to the correct format
        print("Checking and correcting date format...")
        data['date'] = pd.to_datetime(data['date'], errors='coerce')  # Convert to datetime
        invalid_dates = data[data['date'].isnull()]  # Identify rows with invalid dates

        if not invalid_dates.empty:
            print("Invalid dates detected. Dropping rows with invalid dates...")
            print(invalid_dates)
            data = data.dropna(subset=['date'])  # Drop rows with invalid dates

        # Convert to standard yyyy-mm-dd format
        data['date'] = data['date'].dt.strftime('%Y-%m-%d')

        # Save the corrected dataset
        data.to_csv(output_path, index=False)
        print(f"Dataset with corrected dates saved to: {output_path}")

    except Exception as e:
        print(f"Error: {e}")

# Paths
input_file = r"E:\CodeFusion\combined_stock_data_2_backup.csv"  # Input file path
output_file = r"E:\CodeFusion\combined_stock_data_2_corrected.csv"  # Corrected output file path

# Execute the function
check_and_correct_dates(input_file, output_file)
