import time
import subprocess
import sys

def execute_command_with_timing(command):
    start_time = time.time()  # Inizio del timer

    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    end_time = time.time()  # Fine del timer
    elapsed_time = end_time - start_time  # Tempo trascorso in secondi

    # Conversione in hh:mm:ss
    hours, remainder = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"Execution time: {hours:02}:{minutes:02}:{seconds:02}")
    #print(f"Output:\n{result.stdout}")
    #print(f"Errors:\n{result.stderr}")

if len(sys.argv) < 2:
    print("Usage: python script.py <dataset_path>")
    sys.exit(1)

dataset_path = sys.argv[1]

print("Files to txt conversion")
execute_command_with_timing(f"python3 scripts/file_to_text_converter.py {dataset_path}")
print("\n\n\n")
print("Questions generation")
execute_command_with_timing(f"python3 scripts/questions_generator.py {dataset_path}")
print("\n\n\n")
print("Passages generation")
execute_command_with_timing(f"python3 scripts/passages_generator.py {dataset_path}")
print("\n\n\n")
print("Indexes creation")
execute_command_with_timing(f"python3 scripts/indexer.py {dataset_path}")
print("\n\n\n")
print("Retrieving passages for answering questions")
execute_command_with_timing(f"python3 scripts/passages_retriever.py {dataset_path}")
print("\n\n\n")
print("Questions answering")
execute_command_with_timing(f"python3 scripts/questions_answering.py {dataset_path}")
print("\n\n\n")
print("Evaluation")
execute_command_with_timing(f"python3 scripts/q\&a_evaluator.py {dataset_path}")
print(f"Final figure at {dataset_path}/answers/generated_questions/accuracy.png")
