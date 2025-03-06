#madaki script to convert txt data into dataframe stored in csv 
"""
How to Use
Input File: Save your text sample as clean_diseases.txt in the input folder (e.g., /content/drive/MyDrive/input/).
Run: Execute the script and enter paths when prompted.
Output: Check sythetic_data.csv in the output folder for the processed data.
"""
import csv  # For writing CSV files
import re   # For regular expression pattern matching
import os   # For file and directory operations

def process_text_to_csv(input_file_path, output_csv_path):
    """
    Process a text file containing disease data and write it to a CSV file line by line.
    
    Args:
        input_file_path (str): Path to the input .txt file containing disease data.
        output_csv_path (str): Path where the output .csv file will be saved.
    """
    # Extract the directory path from the output file path
    output_dir = os.path.dirname(output_csv_path)
    # Create the output directory if it doesn’t exist and isn’t empty
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Check if the input file exists; raise an error if it doesn’t
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Input file not found: {input_file_path}")
    
    # Initialize counters and storage for processing
    data_count = 0           # Tracks the number of disease entries processed
    current_disease = None   # Stores the current disease name being processed
    current_entry = {}       # Dictionary to hold Symptoms, Diagnosis, and Treatment for the current disease
    total_lines = 0          # Counts total lines read from the input file
    
    # Define regex patterns to match fields without requiring hyphens
    patterns = {
        'Symptoms': r'Symptoms[:\s]*(.*)',   # Matches "Symptoms:" followed by content, optional colon/whitespace
        'Diagnosis': r'Diagnosis[:\s]*(.*)', # Matches "Diagnosis:" followed by content
        'Treatment': r'Treatment[:\s]*(.*)'  # Matches "Treatment:" followed by content
    }
    
    # Open the output CSV file in write mode with UTF-8 encoding
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)  # Create a CSV writer object
        # Write the header row with column names in the specified order
        writer.writerow(['Disease', 'Treatment', 'Diagnosis', 'Symptoms'])
        
        try:
            # Open the input text file for reading with UTF-8 encoding
            with open(input_file_path, 'r', encoding='utf-8') as txt_file:
                # Iterate over each line in the file, tracking line numbers
                for line_num, line in enumerate(txt_file, 1):
                    total_lines += 1  # Increment total lines counter
                    
                    try:
                        line = line.strip()  # Remove leading/trailing whitespace
                        if not line:  # If the line is empty, skip it
                            continue
                        
                        # Check if the line matches any field (Symptoms, Diagnosis, Treatment)
                        matched = False
                        for field, pattern in patterns.items():
                            match = re.match(pattern, line)  # Try to match the regex pattern
                            if match:
                                # Store the matched content in the current entry dictionary
                                current_entry[field] = match.group(1).strip()
                                matched = True
                                break
                        
                        # If no field matches, assume the line is a new disease name
                        if not matched:
                            # If there’s a previous disease with data, write it to CSV
                            if current_disease and current_entry:
                                writer.writerow([
                                    current_disease,                          # Disease name
                                    current_entry.get('Treatment', ''),      # Treatment (default to empty if missing)
                                    current_entry.get('Diagnosis', ''),      # Diagnosis (default to empty if missing)
                                    current_entry.get('Symptoms', '')        # Symptoms (default to empty if missing)
                                ])
                                data_count += 1  # Increment processed disease count
                                # Print progress every 100 diseases
                                if data_count % 100 == 0:
                                    print(f"Processed {data_count} diseases at line {line_num}...")
                            
                            # Set the new disease name and reset the entry dictionary
                            current_disease = line
                            current_entry = {}
                            print(f"New disease found at line {line_num}: {current_disease}")
                        
                        # Handle special case: Treatment line before any disease
                        elif field == 'Treatment' and not current_disease:
                            # Warn user and skip this line since there’s no disease to associate it with
                            print(f"Warning: Treatment found before disease at line {line_num}. Associating with previous disease if exists.")
                            continue
                        
                    except Exception as e:
                        # Log any errors encountered while processing the line
                        print(f"Error at line {line_num}: {str(e)}")
                        print(f"Problematic line: {line}")
                        continue
                
                # After the loop, write the last disease entry if it exists
                if current_disease and current_entry:
                    writer.writerow([
                        current_disease,
                        current_entry.get('Treatment', ''),
                        current_entry.get('Diagnosis', ''),
                        current_entry.get('Symptoms', '')
                    ])
                    data_count += 1
        
        except UnicodeDecodeError:
            # Handle files with invalid UTF-8 encoding
            print("Error: File contains non-UTF-8 characters. Please ensure the file is UTF-8 encoded.")
            return
    
    # Print summary of processing
    print(f"Conversion complete! Processed {data_count} diseases.")
    print(f"Total lines read: {total_lines}")
    print(f"Output saved to: {output_csv_path}")

def main():
    """Main function to prompt user for input/output paths and execute the processing"""
    # Prompt user for the folder containing the input text file
    input_folder = input("Enter input folder path (where diseases.txt is located): ")
    # Prompt user for the folder where the output CSV will be saved
    output_folder = input("Enter output folder path (where sythetic_diseases_data.csv will be saved): ")
    
    # Construct full file paths using os.path.join for platform independence
    input_file_path = os.path.join(input_folder, 'clean_diseases.txt')  # Input file name
    output_csv_path = os.path.join(output_folder, 'sythetic_data.csv')  # Output file name
    
    try:
        # Call the processing function with the provided paths
        process_text_to_csv(input_file_path, output_csv_path)
        
        # Count the number of rows in the output CSV (excluding header)
        with open(output_csv_path, 'r', encoding='utf-8') as f:
            line_count = sum(1 for line in f) - 1  # Subtract 1 for the header row
            print(f"Total rows in CSV (excluding header): {line_count}")
            
    except FileNotFoundError as e:
        # Handle case where the input file is not found
        print(f"Error: {str(e)}")
    except Exception as e:
        # Handle any other unexpected errors
        print(f"An error occurred: {str(e)}")

# Entry point of the script
if __name__ == "__main__":
    main()  # Execute the main function when the script is run directly
