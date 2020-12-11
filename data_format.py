"""
CSC110 Final Project - Ozone Layer Visualization

Tobey Brizuela, Daniel Lazaro, Matthew Parvaneh, Michael Umeh
"""
import datetime
import json
import os
import re
from typing import Dict, List

# Getting the names of all the files in the directory.
files = os.listdir('data')

ozone_dict = {}

# Iterating through each file in the directory.
for file in files:
    file = f'data/{file}'

    with open(file, 'r') as f:
        data = f.read()

        data = data.replace('\n', '')

        date_regex = re.compile(r'\b\w{3}\s{2}1, \d{4}\b')
        data_regex = re.compile(r'\b(\d\d\d|\s\s0)(.*?\.5)\b') # Works more or less

        matches = data_regex.findall(data)
        date = date_regex.search(data)

        # date = format_date(date)

        # Structure of the ozone data dictionary.
        for match in matches:
            match = str(match)
            full_row = ''.join(match)
            row = ''.join(full_row)
            print(row)

            # # TODO: Read in latitude values for each chunk - Commented out b/c doesn't work properly
            # lat_regex = re.compile(r'lat =\s+(-?\d{1,2}\.5)')
            # lat_match = lat_regex.search(match)
            # lat = int(lat_match.group(1))
            #
            # ozone_dict[date] = {lat: []}

            # count = 0
            # longitude = -179.5
            # while count < 1092:


# TODO: implement a function to format the date properly
def format_date(date_string: str) -> datetime.datetime:
    """
    Takes the string representation of a date and turns it into the numerical
    representation in the form mm/dd/yyyy
    """
    ...


# TODO: implement a function that
def convert_data() -> None:
    """
    Idk what this does yet...
    """
    ...
