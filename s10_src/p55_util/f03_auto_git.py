from datetime import datetime

import git


def auto_commit_and_get_hash(script_path):
    try:
        # # Insert a unique RUN DATE comment into the script
        # insert_run_date_comment(script_path)

        # Initialize the repository
        repo = git.Repo(search_parent_directories=True)

        # Stage the modified script (and any other files you want to track)
        repo.git.add(A=True)

        # Check if there are changes to commit
        if repo.is_dirty(untracked_files=True):
            # Commit with a message
            commit_message = f"Auto-commit before training run on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            repo.index.commit(commit_message)
            print(f"Code committed: {commit_message}")
        else:
            print("No changes to commit.")

        # Get the latest commit hash
        commit_hash = repo.head.commit.hexsha
        print(f"Git commit hash: {commit_hash}")
        return commit_hash
    except git.exc.GitError as e:
        print(f"Git error: {e}")
        return None