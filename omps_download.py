"""
CSC110 Final Project - Ozone Layer Visualization

Tobey Brizuela, Daniel Lazaro, Matthew Parvaneh, Michael Umeh
"""
import json
import re
import wget
from typing import Dict, List


base_url = 'https://ozonewatch.gsfc.nasa.gov/data/omps/'


def read_data(filename: str) -> List:
    """Reading in the JSON data as a list."""
    with open(filename) as file:
        years = json.load(file)

    return years


def format_ozone_data(raw_data: List) -> Dict[str, List]:
    """
    Function that formats a list of dictionaries in a json file into
    one single dictionary.
    """
    ozone_data = {}

    for lst in raw_data:
        year = list(lst.keys())[0]
        ozone_data[year] = list(lst.values())[0]

    return ozone_data


def extract_filenames(data: Dict[str, List]) -> List:
    """
    Extract the relevant data from the
    """
    first_of_month = re.compile(r'\S+m\d{2}01_\d{4}\w+\.txt')
    text_files = []

    for year in data:
        for file in data[year]:
            mo = first_of_month.search(str(file))

            if mo is not None:
                path = year + '/' + mo.group()
                text_files.append(path)

    return text_files


def download_files(file_list: List[str], path: str) -> None:
    """
    Use wget module to automatically download

    "path" is a string representation of the filepath containing the folder where
    the downloaded files should be stored.
    """
    print(f"Beginning file download from: {base_url}")

    try:
        for file in file_list:
            url = base_url + file
            wget.download(url, path)

            print('Download Successful!')
    except Exception as e:
        print("Something went wrong. Please try again.")
