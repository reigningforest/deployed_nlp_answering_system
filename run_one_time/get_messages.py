"""
Fetches all messages from a paginated API endpoint and saves them to a JSON file.

Handles resuming from previously saved data.
"""


import requests
import json
import time
import os

# The base URL for the API endpoint
BASE_URL = "https://november7-730026606190.europe-west1.run.app/messages/"

# Parameters
LIMIT = 100
DATA_DIR = "data"
OUTPUT_FILENAME = "all_messages.json"
TOTAL_COUNT_FILENAME = "total_messages.txt"

OUTPUT_PATH = os.path.join(DATA_DIR, OUTPUT_FILENAME)
TOTAL_COUNT_PATH = os.path.join(DATA_DIR, TOTAL_COUNT_FILENAME)

def get_total_count():
    """
    Pings the API with a minimal request to get the total number of items
    and saves it to a file.
    """
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        # Ping with limit=1 (or 0) just to get the 'total' count
        params = {"skip": 0, "limit": 1} 
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        
        total = response.json().get("total")
        
        if total is None:
            print("Error: API response did not contain a 'total' key.")
            return None
            
        # Save the total count to our file
        with open(TOTAL_COUNT_PATH, "w", encoding="utf-8") as f:
            f.write(str(total))
        print(f"API reports a total of {total} messages.")
        return int(total)
        
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error getting total count: {http_err}")
    except requests.exceptions.RequestException as err:
        print(f"Error getting total count: {err}")
    except IOError as e:
        print(f"Error saving total count file: {e}")
    
    return None

def load_existing_messages():
    """Loads messages from the JSON file if it exists."""
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                messages = json.load(f)
                print(f"Loaded {len(messages)} existing messages from {OUTPUT_PATH}")
                return messages
        except json.JSONDecodeError:
            print(f"Warning: {OUTPUT_PATH} seems corrupted. Starting from scratch.")
            return []
        except IOError as e:
            print(f"Warning: Could not read {OUTPUT_PATH}. Starting from scratch. Error: {e}")
            return []
    return []

def save_messages(messages):
    """Saves the list of messages to the JSON file."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)
        print(f"\nSuccessfully saved {len(messages)} messages to {OUTPUT_PATH}")
    except IOError as e:
        print(f"Error saving data to file: {e}")

def fetch_all_messages():
    """
    Fetches all messages from the paginated API, resuming from
    previously saved data if it exists.
    """
    # 1. Get the official total from the API
    total_to_fetch = get_total_count()
    if total_to_fetch is None:
        print("Could not determine total message count. Exiting.")
        return

    # 2. Load what we already have
    all_messages = load_existing_messages()
    
    # 3. Determine where to start
    skip = len(all_messages)
    
    # 4. Check if we're already done
    if skip >= total_to_fetch:
        print(f"Data is already complete. {skip} messages found (total expected: {total_to_fetch}). Nothing to fetch.")
        return
        
    if skip > 0:
        print(f"Resuming fetch from item offset {skip}...")
    else:
        print("Starting to fetch all messages from scratch...")

    # 5. Start the fetch loop
    while True:
        try:
            # Set the query parameters for this page
            params = {"skip": skip, "limit": LIMIT}
            
            # Make the GET request
            print(f"Fetching {LIMIT} messages from skip={skip}...")
            response = requests.get(BASE_URL, params=params)
            
            # Raise an exception for bad status codes
            response.raise_for_status()
            
            # Get the full JSON response object
            response_data = response.json()
            
            # Extract the list of messages from the "items" key
            messages = response_data.get("items", [])
            
            # If the "items" list is empty, we've reached the end
            if not messages:
                print("No more messages returned by API. Stopping.")
                break
                
            # Add the fetched messages to our list
            all_messages.extend(messages)
            
            print(f"Fetched {len(messages)} messages (total: {len(all_messages)})...")
            
            # Increment 'skip' to get the next page
            skip += LIMIT
            
            # A small delay to be polite to the API
            time.sleep(0.5)

        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            print("Stopping fetch. Run the script again to resume.")
            break
        except requests.exceptions.ConnectionError as conn_err:
            print(f"Connection error occurred: {conn_err}")
            print("Stopping fetch. Run the script again to resume.")
            break
        except requests.exceptions.Timeout as timeout_err:
            print(f"Timeout error occurred: {timeout_err}")
            print("Stopping fetch. Run the script again to resume.")
            break
        except requests.exceptions.RequestException as err:
            print(f"An unexpected error occurred: {err}")
            print("Stopping fetch. Run the script again to resume.")
            break

    # After the loop finishes (either by success or error),
    # save all the messages we've accumulated.
    if all_messages:
        save_messages(all_messages)

if __name__ == "__main__":
    fetch_all_messages()