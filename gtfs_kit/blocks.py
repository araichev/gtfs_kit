'''
Functions about blocks
'''

from functools import reduce
from . import helpers as hp
import datetime as dt
import pandas as pd

def datestr_to_dt(x: str | None, format_str: str = "%Y%m%d") -> dt.date | None:
    """
    Convert a date string to a datetime.date.
    Return ``None`` if ``x is None``.
    """
    if x is None:
        return None
    return dt.datetime.strptime(x, format_str)

def timestr_to_min(x):
    return hp.timestr_to_seconds(x, mod24=True) // 60

def timestr_to_dt(x, datetime):
    minutes = timestr_to_min(x)
    return datetime + dt.timedelta(minutes=minutes)


def compute_block_times(
    feed: "Feed",
    date: str,
    trip_stats: pd.DataFrame | None = None,
    freq: str = "h",
) -> pd.DataFrame:
    if trip_stats is None:
        trip_stats = feed.compute_trip_stats()
    
    # get relevant trips with connected block_id
    trip_stats = trip_stats.merge(feed.trips[['trip_id', 'block_id']]) # add block_ids to trip stats
    date_trips = feed.get_trips(date)['trip_id']
    trip_stats = trip_stats[trip_stats['trip_id'].isin(date_trips)]

    # attach datetimes to trips
    block_dt = datestr_to_dt(date)
    trip_stats[['start_dt', 'end_dt']] = trip_stats[['start_time', 'end_time']].map(lambda x: timestr_to_dt(x, block_dt))

    # find start and end times for all blocks
    block_times = trip_stats.groupby('block_id').agg(
        start_dt = ('start_dt', min),
        end_dt = ('end_dt', max))
    
    return block_times


def active_blocks_by_freq_0(
    block_times, 
    freq='1h', 
    block_list: list = None, 
    block_filt: 'function' = None) -> pd.DataFrame:
    
    if block_list: # filter for list of blocks
        block_times = block_times[block_times.index.isin(block_list)]
    if block_filt: # filter blocks using function
        block_times = block_times[list(map(block_filt, block_times.index))]

    # return none if no blocks in block times
    if not block_times.shape[0]:
        return pd.DataFrame(columns=['active_blocks', 'block_starts', 'block_ends'])
    
    # create dt index to check for active blocks
    first_block = pd.Timestamp(block_times['start_dt'].min())
    last_block = pd.Timestamp(block_times['end_dt'].max())
    day_start = first_block.floor('1D')
    range_start = first_block.floor(freq)
    range_end = last_block.ceil(freq)
    dr = pd.date_range(start=range_start, end=range_end, freq=freq)
    full_dr = pd.date_range(start=day_start, end=range_end, freq=freq)

    # count active blocks by hour
    active_blocks = []
    for s in dr:
        e = (s + pd.Timedelta(minutes=1)).ceil(freq) # use (s)tart and (e)nd of freq
        # print(s,e)
        ab = block_times[(block_times['start_dt'] < e) & (block_times['end_dt'] >= s)]
        active_blocks.append(ab.shape[0])

    active_blocks = pd.Series(index=dr, data=active_blocks, name='active_blocks').reindex(full_dr, fill_value=0)

    # count block starts and ends by hour
    block_starts = pd.Series(index=block_times['start_dt'], data=1, name='block_starts')
    block_starts = block_starts.resample(freq).count().reindex(full_dr, fill_value=0)
    
    block_ends = pd.Series(index=block_times['end_dt'], data=1, name='block_ends')
    block_ends = block_ends.resample(freq).count().reindex(full_dr, fill_value=0)
    
    return pd.concat([active_blocks, block_starts, block_ends], axis=1)


def active_blocks_by_freq(feed, dates, freq, trip_stats: pd.DataFrame | None = None, block_list=None, block_filt=None) -> pd.DataFrame:
    
    active_block_results = []
    for d in dates:
        cbt = compute_block_times(feed, d, trip_stats, freq=freq)
        active_blocks = active_blocks_by_freq_0(cbt, freq=freq, block_list=block_list, block_filt=block_filt)
        if active_blocks.shape[0]:
            active_block_results.append(active_blocks)
        else:
            empty_dt = pd.Timestamp(datestr_to_dt(d))
            columns=['active_blocks', 'block_starts', 'block_ends']
            index = pd.date_range(start=empty_dt, end=empty_dt + pd.Timedelta(days=1), freq=freq)
            empty_df = pd.DataFrame(index=index, columns=columns, data=0)
            active_block_results.append(empty_df)            

    all_active_blocks = reduce(lambda a, b: a.add(b, fill_value=0), active_block_results)

    # fill in any missing periods
    dr = pd.date_range(start=all_active_blocks.index.min(), end=all_active_blocks.index.max(), freq=freq)

    all_active_blocks = all_active_blocks.reindex(dr, fill_value=0)

    return all_active_blocks




