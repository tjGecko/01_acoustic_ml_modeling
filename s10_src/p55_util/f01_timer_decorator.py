import time
from functools import wraps


def exe_duration(func):
    """Decorator that prints the runtime duration of the function in hours, minutes, and seconds."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate elapsed time

        # Convert elapsed time into hours, minutes, and seconds
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)

        # Format the output
        print(f"Function '{func.__name__}' executed in {int(hours)}h {int(minutes)}m {seconds:.4f}s")
        return result  # Return the result of the original function

    return wrapper

