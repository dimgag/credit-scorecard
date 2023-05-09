# Utility functions for the project

import os
import sys
import json
import time
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Data Cleaning template')
    parser.add_argument('--input', type=str, default='data.csv', help='Input file name')
    parser.add_argument('--output', type=str, default='clean_data.csv', help='Output file name')
    args = parser.parse_args()
    return args

