import pydoc


def get_function(function_fqn: str):
    function = pydoc.locate(function_fqn)
    if function is None or not callable(function):
        raise ValueError(f"Expected to find a function by FQN = {function_fqn}, but found: {function}")
    return function
