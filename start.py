# Generic imports
import os
import sys
import time
sys.path.append(os.path.abspath('.'))

# Custom imports
from lbm.src.app.app      import *
from lbm.src.core.lattice import *
from lbm.src.core.run     import *

########################
# Run lbm simulation
########################
if __name__ == '__main__':

    # Check command-line input
    if (len(sys.argv) == 2):
        app_name = sys.argv[1]
    else:
        print('Command line error, please use as follows:')
        print('python3 start.py app_name')
        sys.exit()

    app_name = 'random3'

    app_factory = factory()
    app_factory.register('turek', turek)
    app_factory.register('array', array)
    app_factory.register('random3', random3)

    # Instanciate app
    app = app_factory.create(app_name)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    base_output_dir = os.path.join(".", "results", timestamp)
    os.makedirs(base_output_dir, exist_ok=True)

    # Instanciate lattice
    ltc = lattice(app, base_output_dir)

    # Run
    run(ltc, app)
