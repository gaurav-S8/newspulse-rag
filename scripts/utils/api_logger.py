# Import Libraries
import os
import pandas as pd
from typing import Dict

def log_api_request(
    filePath: str,
    paramsUsed: Dict
) -> None:
    """
    Logs details of a NewsAPI request into a CSV log file.

    Reads the existing log file, appends a new row with request metadata and statistics,
    and writes the updated DataFrame back to the same file.

    Parameters
    ----------
    filePath: str
        Path to the CSV log file.
    paramsUsed: Dict
        Dictionary containing request metadata with the following keys:
        [0] Endpoint Used (str)
        [1] Keyword Searched (str or None)
        [2] SearchIn (str or None)
        [3] Country (str or None)
        [4] Category (str or None)
        [5] Sources (str or None)
        [6] Domains (str or None)
        [7] Excluded Domains (str or None)
        [8] From Date (str or None)
        [9] To Date (str or None)
        [10] Sort By (str)
        [11] Language (str)
        [12] Status (str: 'ok' or 'error')
        [13] Articles Fetched (int)
        [14] Time Queried (str: e.g. '2025-06-04 14:03:52')
        [15] Date (str: e.g. '2025-06-04')
        [16] Error, if any (str)

    Returns
    -------
    None
    """

    try:
        df = pd.read_csv(filePath)
    except Exception:
        df = pd.DataFrame([])

    newRow = pd.DataFrame(
        [{
            'Endpoint Used': paramsUsed[0],
            'Keyword Searched': paramsUsed[1],
            'SearchIn': paramsUsed[2],
            'Country': paramsUsed[3],
            'Category': paramsUsed[4],
            'Sources': paramsUsed[5],
            'Domains': paramsUsed[6],
            'Excluded Domains': paramsUsed[7],
            'From Date': paramsUsed[8],
            'To Date': paramsUsed[9],
            'Sort By': paramsUsed[10],
            'Language': paramsUsed[11],
            'Status': paramsUsed[12],
            'Articles Fetched': paramsUsed[13],
            'Time Queried': paramsUsed[14],
            'Date': paramsUsed[15],
            'Error': paramsUsed[16]
        }]
    )
    df = pd.concat([df, newRow], ignore_index = True)
    os.makedirs(os.path.dirname(filePath), exist_ok = True)
    df.to_csv(filePath, index = False)