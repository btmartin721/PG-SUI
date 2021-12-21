#!/bin/bash

TESTDIR=~/Desktop/pgsuipiptest
SRCDIR=~/Desktop/PG-SUI

cd $SRCDIR

pip uninstall pgsui

python setup.py install --force

cd $TESTDIR

./pg_sui.py -h
