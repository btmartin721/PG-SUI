#!/usr/bin/env python

import sys

import numpy as np
import pandas as pd
import scipy.stats as stats

from simulation.simulation import SimGenotypeData

def main():
    t = "((A, B), C);"
    sim = SimGenotypeData(
        poptree=t,
        n_to_sample=4
    )


if __name__ == "__main__":
    main()
