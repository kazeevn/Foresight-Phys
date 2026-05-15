import json
import re
from pathlib import Path

def check_leakage():
    filtered_dir = Path("JSONs/filtered")
    leaked_files = []

    for json_file in filtered_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.loads(f.read())
            
            file_leaked = False
            for exp_idx, exp in enumerate(data):
                desc = exp.get("experiment_description", "")
                
                results = exp.get("experiment_results", {})
                for res_key, res_val in results.items():
                    res_value = res_val.get("result")
                    res_desc = res_val.get("description", "")
                    
                    if isinstance(res_value, (int, float)):
                        val_str = str(res_value)
                        pattern = r'\b' + re.escape(val_str) + r'\b'
                        if re.search(pattern, desc):
                            # print(f"LITERAL LEAKAGE in {json_file.name}: Value {val_str} found in experiment_description")
                            file_leaked = True
                        if re.search(pattern, res_desc):
                            # print(f"RESULT DESC LEAKAGE in {json_file.name}: Value {val_str} found in result description of {res_key}")
                            file_leaked = True
            
            if file_leaked:
                leaked_files.append(json_file.name)
        except Exception as e:
            pass

    return leaked_files

if __name__ == "__main__":
    leaked = check_leakage()
    print(f"\nFiles with any literal leakage: {leaked}")
