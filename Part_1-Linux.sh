#!/bin/bash

# 1. File Management
echo "Listing CSV files with details:"
ls -l *.csv

echo "
# The permission string (e.g., -rw-r--r--) means:
# -: regular file
# rw-: owner can read and write
# r--: group can read
# r--: others can read
"

# 2. Data Download
CDC_URL="https://data.cdc.gov/resource/q3t8-zr7t.csv"

wget -O cdc_covid_cases.csv "$CDC_URL"

echo "Verifying file integrity with sha256sum:"
sha256sum cdc_covid_cases.csv

# 3. Text Processing
echo "Searching for 'BMI >=30' in NHANES dataset and counting occurrences:"
grep 'BMI >=30' NHANES_Body_Measure.csv| wc -l > bmi30_results.txt

# 4. Memory Management
echo "Checking available RAM (in MB):"
free -m

echo "
# Linux uses swap space when RAM is full.
# Swap is slower disk-based memory, used to prevent crashes when physical RAM is exhausted.
"

# 5. Permissions
echo "Granting execute permission to all users for script.py, keeping read/write for owner:"
chmod u=rwx script.py #Permitting user (owner) to read, write and execute
chmod go=x script.py #Permitting Groups and Others to Execute

ls -l script.py
