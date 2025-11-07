import os
import json

def load_json(path):
    """Safely load JSON file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def find_json_files(directory):
    """Find all JSON files starting with 'result_'."""
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith("result_") and file.endswith(".json"):
                json_files.append(os.path.join(root, file))

    return json_files

def compare_json(data1, data2):
    # print(len(data1))
    """Compare two JSON objects deeply (ignoring key order)."""
    return data1 == data2
    # return json.dumps(data1, sort_keys=True) == json.dumps(data2, sort_keys=True)

def find_matching_json_files(directory, reference_data):
    """Find JSON files that match the provided reference array."""
    json_files = find_json_files(directory)
    matching_files = []

    for json_file in json_files:
        data = load_json(json_file)
        if data is not None and compare_json(data, reference_data):
            matching_files.append(json_file)

    return matching_files


if __name__ == "__main__":
    # Example usage
    # directory = input("Enter directory to search: ").strip()
    directory = "./results"

    # Provide the reference array directly here
    # Example:
    # reference_json = [0, 4, 9, 11, 30, 39, 47, 50, 53, 56, 78, 103, 104, 112, 116, 131, 140, 149, 156, 168, 180, 186, 195, 199, 202, 204, 211, 216, 217, 221, 228, 248, 251, 259, 260, 263, 272, 279, 284, 293]
    # reference_json = [40, 86, 116, 143, 147, 194, 200, 202, 211, 213, 248, 250, 271, 291, 302, 323, 345, 354, 361, 380, 382, 393, 396, 400, 414, 419, 479, 491]
    # reference_json = [60, 84, 117, 125, 148, 172, 179, 211, 248, 251, 255, 259, 266, 289, 319, 336, 345, 348, 376, 380, 394, 407, 409, 429, 433, 460, 468, 479]
    reference_json = [1, 12, 17, 28, 31, 43, 44, 62, 70, 74, 75, 76, 77, 100, 121, 123, 161, 170, 181, 183, 189, 221, 224, 226, 247, 248, 272, 284, 315, 333, 340, 342, 348, 352, 355, 358, 387, 393, 398, 408, 439, 452, 487]

    matches = find_matching_json_files(directory, reference_json)

    if matches:
        print("\nMatching JSON files found:")
        for match in matches:
            print(" -", match)
    else:
        print("\nNo matching JSON files found.")
