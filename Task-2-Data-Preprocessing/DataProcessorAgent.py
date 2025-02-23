import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from groq import Agent  # Import Groq Agent

# Download necessary NLTK data (do this once, maybe outside the agent)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def preprocess_text(text):
    """Preprocesses text (lowercase, punctuation removal, tokenization, stopwords)."""
    if pd.isna(text) or isinstance(text, (int, float)):  # Handle NaN or numbers
        return ''

    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)


def process_dataframe(df):
    """Processes the DataFrame, adding preprocessed columns and stats."""
    columns = ['disease_name', 'description', 'diagnosis', 'treatment', 'symptoms', 'source', 'table_name']

    for col in columns:
        df[f'{col}_processed'] = df[col].apply(preprocess_text)
        df[f'{col}_word_count'] = df[f'{col}_processed'].apply(lambda x: len(x.split()) if x else 0)
        df[f'{col}_char_length'] = df[f'{col}_processed'].str.len().fillna(0)  # Handle NaNs in length

        #  No need to print inside the agent, handle in Groq app
        # print(f"Summary statistics for {col}:")
        # print(df[f'{col}_char_length'].describe())
        # print(df.shape)  # Shape can be accessed after processing

    return df


class DataProcessorAgent(Agent):
    def __init__(self, name="DataProcessor"):
        super().__init__(name=name)
        self.df = None  # Store the DataFrame

    async def handle_message(self, message):
        """Handles messages, assumes message.content is a path to CSV."""
        file_path = message.content  # Get file path from message
        try:
            self.df = pd.read_csv(file_path) # Read the CSV
            self.df = process_dataframe(self.df) # Process the dataframe
            # Instead of saving to a file, you can now send the processed dataframe
            # back to your Groq application to save it there.
            await self.send_message(self.df.to_json())  # Send the dataframe as JSON
            # Or send a success message
            # await self.send_message("Data processed successfully.")

        except FileNotFoundError:
            await self.send_message("File not found.")
        except pd.errors.EmptyDataError:
            await self.send_message("Empty CSV file")
        except Exception as e: # Catch any other exceptions
            await self.send_message(f"An error occurred: {e}")

async def main():
    processor_agent = DataProcessorAgent()
    await processor_agent.start()

    # Example interaction (replace with your actual Groq message handling)
    input_file_path = "/content/drive/MyDrive/01-nlp/raw data/cg-drugs-com.csv" # Example
    await processor_agent.handle_message(type('obj', (object,), {'content': input_file_path})()) # Mock message

    await processor_agent.stop()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())


"""
Groq Agent: The code is structured as a DataProcessorAgent class inheriting from groq.Agent.

handle_message: This method now receives a Groq message.  
It assumes that the message.content contains the file path to the CSV file. 
It reads the CSV, processes the data, and sends the processed DataFrame back (as a JSON string) using self.send_message().

Error Handling:  Added try...except blocks to handle FileNotFoundError, pd.errors.EmptyDataError, and other potential exceptions during file reading and processing.  
This is crucial for a robust agent.

DataFrame Storage: The self.df attribute in the agent now stores the processed DataFrame. This is useful if you want to perform further operations on the DataFrame 
within the agent later.

No File Saving Inside Agent: The agent no longer saves the processed data to a file directly.  Instead, it sends the DataFrame back to your Groq application.  
This gives your application control over how and where to save the data.  This is important because agents are designed to be modular.

Sending DataFrame as JSON: The processed DataFrame is converted to a JSON string using self.df.to_json() before being sent.  This is a good way to transmit data within Groq.  
Your Groq application can then parse the JSON back into a DataFrame.

NLTK Downloads: The NLTK downloads are now wrapped in try...except blocks. This prevents errors if the data has already been downloaded. 
It also might be better to handle these downloads outside the agent initialization, perhaps in your main Groq application setup.

Example Interaction: The example interaction is updated to send a file path as the message content.

How to Use in Your Groq Application:

Replace Mock Message: Replace the mock message in the example with your actual Groq Message object.
Send File Path: In your Groq application, when you want to use the DataProcessorAgent, send a message where the message.content is the path to the CSV file you want to process.
Receive Processed DataFrame: After sending the message, your Groq application should receive a message back from the agent. The message.content of this message will be the JSON string representing the processed DataFrame. Parse this JSON string back into a Pandas DataFrame using pd.read_json().
Save DataFrame in Your App: Your Groq application is now responsible for saving the received DataFrame to a file or database as needed.

"""
