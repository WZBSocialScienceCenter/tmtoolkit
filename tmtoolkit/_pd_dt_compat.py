"""
Module for pandas DataFrame / datatable Frame compatibility.
"""

import numpy as np

try:
    import datatable as dt
    USE_DT = True
    FRAME_TYPE = dt.Frame
except ImportError:
    import pandas as pd
    USE_DT = False
    FRAME_TYPE = pd.DataFrame


def pd_dt_frame(data, colnames=None):
    """
    Create a new datatable Frame or pandas DataFrame from data `data`.
    """

    if USE_DT:
        return dt.Frame(data, names=colnames)
    else:
        # if isinstance(data, np.ndarray):
        #     data = data.T
        # elif isinstance(data, list):
        #     data = list(map(list, zip(*data)))

        return pd.DataFrame(data, columns=colnames)


def pd_dt_concat(frames, axis=0):
    """
    Concatenate sequence of datatable Frames or pandas DataFrames `frames` along `axis` (0 means rows, 1 means columns).
    """

    if USE_DT:
        if axis == 0:
            return dt.rbind(*frames)
        elif axis == 1:
            return dt.cbind(*frames)
        else:
            raise ValueError('invalid axis:', axis)
    else:
        return pd.concat(frames, axis=axis)


def pd_dt_sort(frame, cols):
    """
    Sort datatable Frame or pandas DataFrame `frame` along columns `cols`.
    """

    if USE_DT:
        return frame[:, :, dt.sort(*cols)]
    else:
        return frame.sort_values(list(cols))


def pd_dt_colnames(frame):
    """
    Return column names from datatable Frame or pandas DataFrame `frame`.
    """

    if USE_DT:
        return list(frame.names)
    else:
        return list(frame.columns)


def pd_dt_frame_to_list(frame):
    """
    Convert data in datatable Frame or pandas DataFrame `frame` to nested list.
    """

    if USE_DT:
        return frame.to_list()
    else:
        return frame.to_numpy().T.tolist()
