import json
from pathlib import Path

def main():
    filtered_dir = Path("JSONs/filtered")
    raw_dir = Path("JSONs/raw")
    
    if not filtered_dir.exists():
        print(f"Directory {filtered_dir} does not exist.")
        return

    removed_count = 0
    files_to_check = list(filtered_dir.glob("*.json"))
    
    for json_file in files_to_check:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    is_empty = True
                else:
                    data = json.loads(content)
                    # Check if it is an empty list or if it's a list containing only empty lists/dicts?
                    # The user said "contain empty lists". 
                    # For now, we assume it means the result of extraction was an empty list.
                    is_empty = isinstance(data, list) and len(data) == 0
            
            if is_empty:
                print(f"Removing empty JSON: {json_file.name}")
                
                # Remove from filtered
                json_file.unlink()
                
                # Remove corresponding file from raw (assuming this is what "corresponding" refers to)
                raw_file = raw_dir / json_file.name
                if raw_file.exists():
                    raw_file.unlink()
                    print(f"  Also removed corresponding raw file: {raw_file.name}")
                else:
                    print(f"  Corresponding raw file not found in {raw_dir}: {raw_file.name}")
                
                removed_count += 1
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    print(f"Cleanup complete. Removed {removed_count} file pairs.")

if __name__ == "__main__":
    main()
