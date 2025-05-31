from datetime import datetime


def insert_run_date_comment(script_path):
    # Read the contents of the script
    with open(script_path, 'r') as file:
        content = file.readlines()

    # Create a unique run date comment
    run_date_comment = f"# RUN DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

    # Insert the run date at the top of the script (after any existing shebang lines)
    if content[0].startswith("#!"):  # Keep shebang (if present) at the top
        content.insert(1, run_date_comment)
    else:
        content.insert(0, run_date_comment)

    # Write the modified content back to the file
    with open(script_path, 'w') as file:
        file.writelines(content)

    print(f"Inserted RUN DATE comment: {run_date_comment.strip()}")

