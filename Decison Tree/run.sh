#!/bin/sh

echo "Training and Test errors for Car DataSet"
python3 Car_Decision_Tree.py

echo "Training and Test errors for Bank DataSet -unknown as feature value"
python3 Bank_Decision_Tree.py

echo " Training and Test errors for Bank DataSet -unknown as missing value"
python3 Unknown_Bank_Decision_Tree.py
