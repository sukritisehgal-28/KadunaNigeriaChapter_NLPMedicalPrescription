#@madakixo 
"""
Parse the text using regular expressions to identify disease names and their respective fields
Organize the data into the required 4-column format
Create a CSV file named 'diseases.csv'To use this script:
Save it (e.g., as convert_to_csv.py)
Create your input text file named 'diseases.txt' 
  run script 
  When prompted:
Enter the input folder path (e.g., "C:/Users/YourName/Documents/input")
Enter the output folder path (e.g., "C:/Users/YourName/Documents/output")
"""

import csv
import re
import os

def process_text_to_csv(input_file_path, output_csv_path):
    """Process text file with hyphen format and write to CSV line by line"""
    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Check if input file exists
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Input file not found: {input_file_path}")
    
    # Initialize variables
    data_count = 0
    current_disease = None
    current_entry = {}
    
    # Regex patterns for hyphen format
    patterns = {
        'Symptoms': r'-\s*Symptoms: (.*)',
        'Diagnosis': r'-\s*Diagnosis: (.*)',
        'Treatment': r'-\s*Treatment: (.*)'
    }
    
    # Open CSV file for writing
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Disease', 'Treatment', 'Diagnosis', 'Symptoms'])
        
        # Process input file line by line
        try:
            with open(input_file_path, 'r', encoding='utf-8') as txt_file:
                for line_num, line in enumerate(txt_file, 1):
                    try:
                        line = line.strip()
                        if not line:  # Skip empty lines
                            continue
                        
                        # New disease entry (no hyphen)
                        if not line.startswith('-'):
                            # Write previous entry if complete
                            if current_disease and current_entry:
                                writer.writerow([
                                    current_disease,
                                    current_entry.get('Treatment', ''),
                                    current_entry.get('Diagnosis', ''),
                                    current_entry.get('Symptoms', '')
                                ])
                                data_count += 1
                                if data_count % 100 == 0:  # Progress update
                                    print(f"Processed {data_count} diseases...")
                            
                            # Start new entry
                            current_disease = line
                            current_entry = {}
                        
                        # Match Symptoms, Diagnosis, or Treatment
                        else:
                            for field, pattern in patterns.items():
                                match = re.match(pattern, line)
                                if match:
                                    current_entry[field] = match.group(1)
                                    break
                    
                    except Exception as e:
                        print(f"Error at line {line_num}: {str(e)}")
                        print(f"Problematic line: {line}")
                        continue
                
                # Write the last entry
                if current_disease and current_entry:
                    writer.writerow([
                        current_disease,
                        current_entry.get('Treatment', ''),
                        current_entry.get('Diagnosis', ''),
                        current_entry.get('Symptoms', '')
                    ])
                    data_count += 1
        except UnicodeDecodeError:
            print("Error: File contains non-UTF-8 characters. Please ensure the file is UTF-8 encoded.")
            return
    
    print(f"Conversion complete! Processed {data_count} diseases.")
    print(f"Output saved to: {output_csv_path}")

def main():
    # Get folder paths from user
    input_folder = input("Enter input folder path (where diseases1.txt is located): ")
    output_folder = input("Enter output folder path (where sythetic_diseases_data.csv will be saved): ")
    
    # Construct full file paths
    input_file_path = os.path.join(input_folder, 'diseases.txt')
    output_csv_path = os.path.join(output_folder, 'sythetic_diseases_data1.csv')
    
    try:
        process_text_to_csv(input_file_path, output_csv_path)
        
        # Verify output line count
        with open(output_csv_path, 'r', encoding='utf-8') as f:
            line_count = sum(1 for line in f) - 1  # Subtract header
            print(f"Total rows in CSV (excluding header): {line_count}")
            
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
