import os
import json
import re
from bill_extractor import parse_data , create_pandas

if __name__ == "__main__":
    response = parse_data("/Users/prateekpuri/Documents/utility/alectra-june.pdf")
    print(response)
    create_pandas(response)
    #print(response)
    # Create a DataFrame from the JSON data
    #create_pandas(response)

# Display the DataFrame
