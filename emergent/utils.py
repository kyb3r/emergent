import sys
import threading
import itertools
import time
from colorama import init, Fore, Style

init(autoreset=True)


class Spinner:
    def __init__(self, message="Loading...", delay=0.1):
        self.spinner = itertools.cycle(["-", "/", "|", "\\"])
        self.delay = delay
        self.message = message
        self.running = False
        self.spinner_thread = None

    def spin(self):
        while self.running:
            sys.stdout.write(next(self.spinner) + " " + self.message + "\r")
            sys.stdout.flush()
            time.sleep(self.delay)
            sys.stdout.write("\b" * (len(self.message) + 2))

    def __enter__(self):
        self.running = True
        self.spinner_thread = threading.Thread(target=self.spin)
        self.spinner_thread.start()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.running = False
        self.spinner_thread.join()
        sys.stdout.write("\r" + " " * (len(self.message) + 2) + "\r")
        sys.stdout.flush()


def print_colored(text, color=Fore.RESET):
    print(color + text, end="", flush=True)


def process_response(response):
    thinking = False
    using_tool = False
    for token in response:
        if (token == "<hidden" or token == "<") and not thinking:
            print_colored("[thinking...]\n", Fore.YELLOW)
            thinking = True
        elif token == "__" and not using_tool:
            print_colored("[using tool...] ", Fore.GREEN)
            using_tool = True
        elif not using_tool and not thinking and isinstance(token, str):
            print_colored(token)

        if isinstance(token, dict):
            if "tool_name" in token:
                print_colored(f'[{token["tool_name"]}] ', Fore.GREEN)
                print_colored(f'args {token["tool_params"]}\n', Fore.LIGHTGREEN_EX)
                using_tool = False
                thinking = False
            if "tool_result" in token:
                using_tool = False
                thinking = False

        if (token in ['">', " >", ">", "> "]) and thinking:
            thinking = False
