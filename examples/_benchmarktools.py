from datetime import datetime


timings = []
timing_labels = []


def add_timing(label):
    timings.append(datetime.today())
    timing_labels.append(label)


def print_timings():
    print('timings:')
    t_sum = 0
    prev_t = None
    for i, (t, label) in enumerate(zip(timings, timing_labels)):
        if i > 0:
            t_delta = (t - prev_t).total_seconds()
            print('%s: %.2f sec' % (label, t_delta))
            t_sum += t_delta

        prev_t = t

    print('total: %.2f sec' % t_sum)
