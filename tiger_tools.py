def add(a, b):
    """Add two numbers

    Parameters
    ----------
    a : numeric
        Any number
    b : numeric
        Any number

    Returns
    -------
    Numeric
        Sum of a and b
    """
    return a + b

def _add_test():
    """Simple addition test
    """
    c = add(1, 2)
    print(f'1 + 2 = {str(c)}')
