import os
import re
import pandas as pd

def extract_episode_data(file_path):
    pattern = re.compile(
        r"Agent\s(\d+)/Episode Return\s=\s([\d\.]+)")
    
    data = []
    with open(file_path, 'r') as file:
        file_name = os.path.basename(file_path)
        episode_number = 0
        
        for line in file:
            if 'Episode Return' in line:
                matches = pattern.findall(line)
                if matches:
                    episode_number += 1
                    episode_data = {"File Name": file_name, "Episode": episode_number}
                    for agent_id, return_value in matches:
                        episode_data[f"Agent {agent_id} Return"] = float(return_value)
                    data.append(episode_data)
    
    return data

def process_logs(directory, output_csv):
    all_data = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".txt"):
            file_path = os.path.join(directory, file_name)
            all_data.extend(extract_episode_data(file_path))
    
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)
    print(f"CSV saved to {output_csv}")

# Usage example
log_directory = "logs"
output_csv = "evaluation_results.csv"
process_logs(log_directory, output_csv)
