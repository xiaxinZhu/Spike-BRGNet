import numpy
import numpy as np
import torch
import cv2
# y_k_size = 6
# x_k_size = 6
y_k_size = 3
x_k_size = 3

# def gen_edge(label, edge_pad=True, edge_size=4):
def gen_edge(label, edge_pad=True, edge_size=2):
    edge = cv2.Canny(label, 0.1, 0.2)
    kernel = np.ones((edge_size, edge_size), np.uint8)
    if edge_pad:
        edge = edge[y_k_size:-y_k_size, x_k_size:-x_k_size]
        edge = np.pad(edge, ((y_k_size,y_k_size),(x_k_size,x_k_size)), mode='constant')
    edge = (cv2.dilate(edge, kernel, iterations=1)>50)*1.0
    return edge

def generate_input_representation(events, event_representation, shape, nr_temporal_bins=5, separate_pol=True):
    """
    Events: N x 4, where cols are x, y, t, polarity, and polarity is in {-1, 1}. x and y correspond to image
    coordinates u and v.
    """
    if event_representation == 'histogram':
        return generate_event_histogram(events, shape)
    elif 'voxel_grid' in event_representation:
        return generate_voxel_grid(events, shape, nr_temporal_bins, separate_pol)
    elif event_representation == 'MDOE':
        return generate_MDOE(events, shape, nr_temporal_bins)
    elif 'SBT' in event_representation or 'SBE' in event_representation:
        flag = event_representation.split('_')[-1]
        return generate_SBT_SBE(flag, events, shape, nr_temporal_bins)
    elif event_representation == 'ev_segnet':
        return generate_ev_segnet(events, shape, nr_temporal_bins)


def generate_event_histogram(events, shape):
    """
    Events: N x 4, where cols are x, y, t, polarity, and polarity is in {-1, 1}. x and y correspond to image
    coordinates u and v.
    """
    height, width = shape
    x, y, t, p = events.T
    x = x.astype(np.int32)
    y = y.astype(np.int32)
    p[p == 0] = -1  # polarity should be +1 / -1
    img_pos = np.zeros((height * width,), dtype="float32")
    img_neg = np.zeros((height * width,), dtype="float32")

    np.add.at(img_pos, x[p == 1] + width * y[p == 1], 1)
    np.add.at(img_neg, x[p == -1] + width * y[p == -1], 1)

    histogram = np.stack([img_neg, img_pos], 0).reshape((2, height, width))

    return histogram


def normalize_voxel_grid(events):
    """Normalize event voxel grids"""
    nonzero_ev = (events != 0)
    num_nonzeros = nonzero_ev.sum()
    if num_nonzeros > 0:
        # compute mean and stddev of the **nonzero** elements of the event tensor
        # we do not use PyTorch's default mean() and std() functions since it's faster
        # to compute it by hand than applying those funcs to a masked array
        mean = events.sum() / num_nonzeros
        stddev = torch.sqrt((events ** 2).sum() / num_nonzeros - mean ** 2)
        mask = nonzero_ev.float()
        events = mask * (events - mean) / stddev

    return events


def generate_voxel_grid(events, shape, nr_temporal_bins, separate_pol=True):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.
    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param nr_temporal_bins: number of bins in the temporal axis of the voxel grid
    :param shape: dimensions of the voxel grid
    """
    height, width = shape
    assert(events.shape[1] == 4)
    assert(nr_temporal_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid_positive = np.zeros((nr_temporal_bins, height, width), np.float32).ravel()
    voxel_grid_negative = np.zeros((nr_temporal_bins, height, width), np.float32).ravel()
    
    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 2]
    first_stamp = events[0, 2]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    # events[:, 2] = (nr_temporal_bins - 1) * (events[:, 2] - first_stamp) / deltaT
    xs = events[:, 0].astype(np.int32)
    ys = events[:, 1].astype(np.int32)
    # ts = events[:, 2]
    # print(ts[:10])
    ts = (nr_temporal_bins - 1) * (events[:, 2] - first_stamp) / deltaT

    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(np.int32)
    dts = ts - tis
    vals_left = np.abs(pols) * (1.0 - dts)
    vals_right = np.abs(pols) * dts
    pos_events_indices = pols == 1

    # Positive Voxels Grid
    valid_indices_pos = np.logical_and(tis < nr_temporal_bins, pos_events_indices)
    valid_pos = (xs < width) & (xs >= 0) & (ys < height) & (ys >= 0) & (ts >= 0) & (ts < nr_temporal_bins)
    valid_indices_pos = np.logical_and(valid_indices_pos, valid_pos)

    np.add.at(voxel_grid_positive, xs[valid_indices_pos] + ys[valid_indices_pos] * width +
              tis[valid_indices_pos] * width * height, vals_left[valid_indices_pos])

    valid_indices_pos = np.logical_and((tis + 1) < nr_temporal_bins, pos_events_indices)
    valid_indices_pos = np.logical_and(valid_indices_pos, valid_pos)
    np.add.at(voxel_grid_positive, xs[valid_indices_pos] + ys[valid_indices_pos] * width +
              (tis[valid_indices_pos] + 1) * width * height, vals_right[valid_indices_pos])

    # Negative Voxels Grid
    valid_indices_neg = np.logical_and(tis < nr_temporal_bins, ~pos_events_indices)
    valid_indices_neg = np.logical_and(valid_indices_neg, valid_pos)

    np.add.at(voxel_grid_negative, xs[valid_indices_neg] + ys[valid_indices_neg] * width +
              tis[valid_indices_neg] * width * height, vals_left[valid_indices_neg])

    valid_indices_neg = np.logical_and((tis + 1) < nr_temporal_bins, ~pos_events_indices)
    valid_indices_neg = np.logical_and(valid_indices_neg, valid_pos)
    np.add.at(voxel_grid_negative, xs[valid_indices_neg] + ys[valid_indices_neg] * width +
              (tis[valid_indices_neg] + 1) * width * height, vals_right[valid_indices_neg])

    voxel_grid_positive = np.reshape(voxel_grid_positive, (nr_temporal_bins, height, width))
    voxel_grid_negative = np.reshape(voxel_grid_negative, (nr_temporal_bins, height, width))

    if separate_pol:
        return np.concatenate([voxel_grid_positive, voxel_grid_negative], axis=0)

    voxel_grid = voxel_grid_positive - voxel_grid_negative
    return voxel_grid


# MDOE
def trilinear_kernel(ts, num_channels):
    gt_values = np.zeros_like(ts)

    gt_values[ts >= 0] = (1 - (num_channels-1) * ts)[ts >= 0] # original code [ts > 0]
    gt_values[ts < 0] = ((num_channels-1) * ts + 1)[ts < 0]

    gt_values[ts < -1.0 / (num_channels-1)] = 0
    gt_values[ts > 1.0 / (num_channels-1)] = 0

    return gt_values

def generate_MDOE(events, shape, nr_temporal_bins):
    epsilon = 10e-3 # avoid dividing by zero
    # event.size = [num,4]
    vox_moe = np.zeros((2, nr_temporal_bins, shape[0], shape[1]), np.float32).ravel()
    vox_doe = np.zeros((2, nr_temporal_bins, shape[0], shape[1]), np.float32).ravel()
    C, H, W = nr_temporal_bins, shape[0], shape[1]
    
    x, y, t, p = events.T # p = 0/1
    # normalize the event timestamps so that they lie between 0 and 1
    last_stamp = t[-1]
    first_stamp = t[0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0
    
    t = (t - first_stamp) / deltaT
    
    # get values for each channel
    for i_bin in range(C):
        # calculate density of events 
        index = (t > i_bin / C) & (t <= (i_bin + 1) / C)
        x1 = x[index]
        y1 = y[index]
        t1 = t[index]
        p1 = p[index]
        
        # [2,C,H,W]
        idx_count = W*H*C*p1 +  W*H*i_bin + W*y1 + x1 
        val_counts = np.zeros_like(x1) + 1
        np.add.at(vox_doe, idx_count, val_counts)
            
        # calculate the magnitude of events (moe) on the same pixel
        index = (t <= (i_bin + 1) / C)
        x1 = x[index]
        y1 = y[index]
        t1 = t[index]
        p1 = p[index]
        
        idx_count = W*H*C*p1 +  W*H*i_bin + W*y1 + x1 
        val = trilinear_kernel(t1 - (i_bin+1)/C, 2)
        np.add.at(vox_moe, idx_count, val)
                    
    # normalize, 不再使用后续预处理的normalize操作!
    vox_moe = vox_moe / (vox_moe.max() + epsilon)
    vox_doe = vox_doe / (vox_doe.max() + epsilon)

    # vox_moe/vox_doe: [2,C,H,W]
    vox_moe = np.reshape(vox_moe, (2, C, H, W))
    vox_doe = np.reshape(vox_doe, (2, C, H, W))

    vox_MDOE = np.concatenate([vox_moe, vox_doe], axis=0)
    
    vox_new = np.zeros((C, 4, H, W), np.float32)
    vox_mid = np.zeros((4, H, W), np.float32)
    for i in range(C):
        for j in range(4):
            vox_mid[j] = vox_MDOE[j, i, ...]
        vox_new[i] = vox_mid 
    vox_MDOE = np.reshape(vox_new, (C*4, H, W))
    
    return vox_MDOE


# ev_segnet
# ev_segnet
def generate_ev_segnet(events, shape):
    """
    Events: N x 4, where cols are x, y, t, polarity, and polarity is in {-1, 1}. x and y correspond to image
    coordinates u and v.
    """
    height, width = shape
    x, y, t, p = events.T
    x = x.astype(np.int32)
    y = y.astype(np.int32)
    p[p == 0] = -1  # polarity should be +1 / -1
    
    img_pos = np.zeros((height * width,), dtype="float32")
    img_neg = np.zeros((height * width,), dtype="float32")
    
    M_pos_ts = np.zeros((height * width,), dtype="float32")
    M_neg_ts = np.zeros((height * width,), dtype="float32")
    
    S_pos_ts = np.zeros((height * width,), dtype="float32")
    S_neg_ts = np.zeros((height * width,), dtype="float32")
    
    # calculate Hist(x,y,p)
    np.add.at(img_pos, x[p == 1] + width * y[p == 1], 1)
    np.add.at(img_neg, x[p == -1] + width * y[p == -1], 1)
    
    # calculate M(x,y,p)
    for i in range(len(p)):
        if p[i] == 1:
            M_pos_ts[x[i] + width * y[i]] += t[i]
        else:
            M_neg_ts[x[i] + width * y[i]] += t[i]
    M_pos_ts /= (img_pos + np.finfo(float).eps) 
    M_neg_ts /= (img_neg + np.finfo(float).eps) 
    
    # calculate S(x,y,p)
    for i in range(len(p)):
        if p[i] == 1:
            S_pos_ts[x[i] + width * y[i]] += np.square(t[i]-M_pos_ts[x[i] + width * y[i]])
        else:
            S_neg_ts[x[i] + width * y[i]] += np.square(t[i]-M_neg_ts[x[i] + width * y[i]])
    S_pos_ts = np.sqrt(S_pos_ts / (img_pos - 1 + np.finfo(float).eps))
    S_neg_ts = np.sqrt(S_neg_ts / (img_neg - 1 + np.finfo(float).eps))

    ev_segnet = np.stack([img_neg, img_pos, M_pos_ts, M_neg_ts, S_pos_ts, S_neg_ts], 0).reshape((6, height, width))

    return ev_segnet


# SBT/SBE
def generate_SBT_SBE(flag, events, shape, nr_temporal_bins):
    C, H, W = nr_temporal_bins, shape[0], shape[1]
    assert(events.shape[1] == 4)
    assert(C > 0)
    assert(W > 0)
    assert(H > 0)

    SBT_positive = np.zeros((C, H, W), np.float32).ravel()
    SBT_negtive = np.zeros((C, H, W), np.float32).ravel()
    
    # normalize the event timestamps so that they lie between 0 and 1
    last_stamp = events[-1, 2]
    first_stamp = events[0, 2]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    x = events[:, 0].astype(np.int32)
    y = events[:, 1].astype(np.int32)
    t = (events[:, 2] - first_stamp) / deltaT
    p = events[:, 3]
    
    p[p == 0] = -1  # polarity should be +1 / -1
    pos_events_indices = p == 1
    neg_events_indices = p == -1

    # get values for each frame
    for i_bin in range(C):
        index = (t > i_bin / C) & (t <= (i_bin + 1) / C)
        # positive_SBT:[C,H,W]
        valid_indices_pos = np.logical_and(index, pos_events_indices)
        x1 = x[valid_indices_pos]
        y1 = y[valid_indices_pos]
        
        idx_count = W*H*i_bin + W*y1 + x1 
        val_counts = np.zeros_like(x1) + 1
        np.add.at(SBT_positive, idx_count, val_counts)
        
        # negtive_SBT:[C,H,W]
        valid_indices_neg = np.logical_and(index, neg_events_indices)
        x1 = x[valid_indices_neg]
        y1 = y[valid_indices_neg]
        
        idx_count = W*H*i_bin + W*y1 + x1 
        val_counts = np.zeros_like(x1) - 1
        np.add.at(SBT_negtive, idx_count, val_counts)

    SBT_positive = np.reshape(SBT_positive, (C, H, W))
    SBT_negtive = np.reshape(SBT_negtive, (C, H, W))

    if flag == '1':
        SBT_new = np.zeros((C, 2, H, W), np.float32)
        SBT_mid = np.zeros((2, H, W), np.float32)
        for i in range(C):
            SBT_mid[0] = SBT_positive[i]
            SBT_mid[1] = SBT_negtive[i]
            SBT_new[i] = SBT_mid
        SBT = np.reshape(SBT_new, (C*2, H, W))
        return SBT
    if flag == '2':
        SBT = np.concatenate((SBT_positive, SBT_negtive), 0)
        return SBT