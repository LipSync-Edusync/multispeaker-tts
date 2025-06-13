import os
import csv

data_root = "/home/oem/Lipsync-Edusync/datasets/speaker_verification_test"
output_csv = "/home/oem/Lipsync-Edusync/datasets/speaker_verification_test/metadata.csv"

with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter='|')
    
    for speaker_id in os.listdir(data_root):
        speaker_dir = os.path.join(data_root, speaker_id)
        
        if not os.path.isdir(speaker_dir):
            continue
        
        for utterance_file in os.listdir(speaker_dir):
            if utterance_file.endswith('.wav'):
                # Format: filename|speaker_id
                writer.writerow([
                    os.path.join(speaker_id, utterance_file),
                    speaker_id
                ])

print(f"Generated {output_csv} with {sum(1 for _ in open(output_csv))} entries")