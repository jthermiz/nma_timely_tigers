from nma_timely_tigers import utils

def test_add():
    """Simple addition test
    """
    c = utils.add(1, 2)
    print(f'1 + 2 = {str(c)}')