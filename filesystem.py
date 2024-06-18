import subprocess


def git_root():
    try:
        # Execute the git command to get the top-level directory
        git_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True, stderr=subprocess.STDOUT
        ).strip()
        return git_root
    except subprocess.CalledProcessError:
        # Handle the case where the command fails (e.g., not in a git repo)
        print("Error: Not a git repository (or any of the parent directories)")
        return None
