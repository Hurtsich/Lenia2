import subprocess
import sys
import os

def test_smoke_test_launches_successfully():
    """Runs the main application with --smoke-test and checks for a zero exit code."""
    # Construct the path to the main.py file relative to this test file
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_script_path = os.path.join(project_root, 'main.py')

    command = [
        sys.executable,  # The current python interpreter
        main_script_path,
        '--smoke-test'
    ]

    # Execute the command. check=True will raise an exception for non-zero exit codes.
    subprocess.run(command, check=True, capture_output=True, text=True)
