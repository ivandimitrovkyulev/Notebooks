#! /usr/bin/env python3
import re
import sys
import shlex
import subprocess
from argparse import ArgumentParser, RawTextHelpFormatter


parser = ArgumentParser(
    usage="./%(prog)s \n",
    formatter_class=RawTextHelpFormatter,
    description="Automatically update repository version across 'pyproject.toml' and 'README.md' files.\n"
                "The version is then interactively added, commited, tagged and pushed using Git.\n",
)


class SubprocessReturnError(Exception):
    pass


def run_subprocess(
        command: str,
        capture_output: bool = False,
        print_stdout: bool = False,
        print_command: bool = False,
) -> str | None:
    """
    Run a command in a subprocess.
    :param command: Command to execute. If string it will be split into a list of words
    :param capture_output: Capture both stdout & stderr. Note: this will not print process stdout in real time
    :param print_stdout: Print Subprocess whole stdout at termination?
    :param print_command: Print Subprocess command and exit code at termination?
    :raises SubprocessReturnError: If subprocess return code is != 0
    """
    # Make sure command is split and constructed as a list for security purposes
    command = shlex.split(command)

    try:
        proc = subprocess.run(command, capture_output=capture_output)

        if print_command:
            print(f"Subprocess {command} exited with code: {proc.returncode}\n")

        if proc.returncode != 0:
            raise SubprocessReturnError(
                f"Command: {' '.join(command)}; stdout: {proc.stdout}; stderror: {proc.stderr}"
            )

        if capture_output:
            output = proc.stdout.decode("utf-8").strip()
            if print_stdout:
                print(output)

            return output

    except KeyboardInterrupt:
        sys.exit(f"\nProgram Terminated with KeyboardInterrupt")


parser.parse_args()
try:
    print(f">>> Updating Version for 'Notebooks' project.")
    last_version = run_subprocess(f"git describe --tags", capture_output=True)
    version = input(f">> Latest version is {last_version}. Specify new version: ")

    run_subprocess(f"poetry version {version}")
    print(f"--> Updated 'pyproject.toml' file to {version}")

    # ------------  Update 'README.md' file ---------------------------------------------------------------------------
    with open(f"README.md", 'r') as file:
        old_text = file.read()

    old_version = re.findall(r"### version \S*", old_text)[0]

    if old_version:
        # Replace the all target strings
        new_text = re.sub(old_version, f"### version {version}", old_text)

        with open(f"README.md", 'w') as file:
            file.write(new_text)
            print(f"--> Updated 'README.md' file to {version}")

    run_subprocess(f"git commit --interactive")
    run_subprocess(f"git tag -a {version} HEAD")
    run_subprocess(f"git push origin {version}")
    run_subprocess(f"git push")
    run_subprocess(f"git status")

except SubprocessReturnError as ex:
    print(ex)

except KeyboardInterrupt as ex:
    sys.exit("\n>>> KeyboardInterrupt. Closing down.")
