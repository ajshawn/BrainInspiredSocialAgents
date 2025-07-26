import re
import os
import glob
import pandas as pd
from collections import defaultdict
import argparse

def parse_log_file(filepath):
    """
    Parses a single log file.
    Returns a dict: { field_name: [values] }
    """
    # Match the part like: [Coop Mining28] ...
    coop_pattern = re.compile(r'\[Coop Mining[^\]]*\](.*)')
    # Match key = value pairs
    keyval_pattern = re.compile(r'([^=|]+)=([^|]+)')

    fields = defaultdict(list)

    with open(filepath, 'r') as f:
        for line in f:
            match = coop_pattern.search(line)
            if match:
                log_content = match.group(1)
                kvs = keyval_pattern.findall(log_content)
                for k, v in kvs:
                    key = k.strip()
                    val = v.strip()
                    try:
                        val = float(val)
                        fields[key].append(val)
                    except ValueError:
                        continue  # skip non-numeric

    return fields

def compute_mean_std(fields_dict):
    """
    For each field, compute mean and std.
    """
    result = {}
    for k, v in fields_dict.items():
        if len(v) > 0:
            mean = sum(v) / len(v)
            std = (sum((x - mean)**2 for x in v) / len(v))**0.5
            result[f'{k} Mean'] = round(mean, 2)
            result[f'{k} Std'] = round(std, 2)
    return result

def main(log_dir, output_csv='aggregated_results.csv'):
    """
    Parse all .txt logs in the directory.
    Output a CSV with file name as row name, columns as all fields.
    """
    all_results = []
    filenames = []

    for filepath in glob.glob(os.path.join(log_dir, '*.txt')):
        print(f'Processing: {filepath}')
        fields = parse_log_file(filepath)
        if not fields:
            print(f'  ⚠️  No [Coop Mining] lines found in: {filepath}')
        stats = compute_mean_std(fields)
        all_results.append(stats)
        filenames.append(os.path.basename(filepath))

    df = pd.DataFrame(all_results)
    df.insert(0, 'FileName', filenames)
    output_csv = os.path.join(log_dir, output_csv)
    df.to_csv(output_csv, index=False)
    print(f'✅ Results saved to {output_csv}')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Parse log files and compute statistics.')
    argparser.add_argument('--log_dir', '-p', type=str, required=True, help='Directory containing log files')

    args = argparser.parse_args()
    main(args.log_dir) 
