from pathlib import Path
import argparse
import sys
from support.test_docker import test_docker
#----- ADD YOUR IMPORTS HERE IF NEEDED -----
import numpy as np
import cv2 as cv
from support.sift_matchers import *
from support.main import *

def run_project(input_dir, output_dir):
    """
    Main entry point for your project code.

    DO NOT MODIFY THE SIGNATURE OF THIS FUNCTION.
    """
    #---- FILL ME IN ----

    # Add your code here...
    print("\nStart Program")
    s = Stitch()
    s.stitching()
    print("\n######################")
    print ("Done")
    # s.pano = s.pano[413:703,737:4534]
    # s.pano = cv.resize(s.pano,(2100,500),interpolation=cv.INTER_CUBIC)
    cv.destroyAllWindows()

    #--------------------


# Command Line Arguments
parser = argparse.ArgumentParser(description='ROB501 Final Project.')
parser.add_argument('--input_dir', dest='input_dir', type=str, default="./input",
                    help='Input Directory that contains all required rover data')
parser.add_argument('--output_dir', dest='output_dir', type=str, default="./output",
                    help='Output directory where all outputs will be stored.')


if __name__ == "__main__":

    # Parse command line arguments
    args = parser.parse_args()

    # Uncomment this line if you wish to test your docker setup
    # test_docker(Path(args.input_dir), Path(args.output_dir))

    # Run the project code
    run_project(args.input_dir, args.output_dir)
