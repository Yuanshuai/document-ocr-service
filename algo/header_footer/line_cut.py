import numpy as np
from scipy.signal import butter, filtfilt
from config import slope_width, cutoff, order, window, throttle, hot_area

hot_header = 0.128
hot_footer = 0.1


# 高通滤波
def img_arr_highpass_filter(arr) -> list:
    new = np.sum(arr, axis=1) // slope_width
    fs = len(new)
    normal_cutoff = 2 * cutoff / fs
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    new = filtfilt(b, a, new)
    old, new = new, []
    for i in range(0, len(old) - window):
        temp_var = min(200, np.var(old[i:i + window]))
        new.append(temp_var)
    avg = max(np.mean(new), 0.25)
    header = 0
    footer = len(new) - 1
    header_score = avg * throttle
    footer_score = avg * throttle
    # plt.plot(new)
    # avg_list = np.array([avg*throttle]*len(new))
    # plt.plot(avg_list)
    # plt.show()
    for y in range(0, int(hot_header * len(new))):
        if new[y] >= header_score:
            header = y + window // 2
            header_score = max(new[y] * 0.8, header_score)
    for y in range(0, int(hot_footer * len(new))):
        if new[len(new) - y - 1 - window] > footer_score:
            footer = len(new) - y - 1 - window // 2
            footer_score = new[len(new) - y - 1 - window]
    header_score = avg * throttle * 2
    footer_score = avg * throttle * 2
    for y in range(int(hot_header * len(new)), int(hot_area * len(new))):
        if new[y] >= header_score:
            header = y + window // 2
            header_score = max(new[y] * 0.8, header_score)
    for y in range(int(hot_footer * len(new)), int(hot_area * len(new))):
        if new[len(new) - y - 1 - window] > footer_score:
            footer = len(new) - y - 1 - window // 2
            footer_score = new[len(new) - y - 1 - window]
    if header / len(arr) > 0.10:
        header = 0
    return [header / len(arr), footer / len(arr), "img_arr_highpass_filter"]
