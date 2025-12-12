import torch
import json
import numpy as np
import os
import zlib


def feat_dataloader_tensor(tens_ds, train=True, device="cuda", batch_size=256):
    return torch.utils.data.DataLoader(
        tens_ds,
        batch_size=batch_size,
#        pin_memory=(True if device == 'cuda' else False),
        shuffle=train
    )

def feat_dataloader(X, y, train=True, device="cuda", batch_size=256):
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    return feat_dataloader_tensor(torch.utils.data.TensorDataset(X, y), train, device, batch_size)

def feat_ag_dataloader(X, ag, y, train=True, device="cuda", batch_size=256):
    X = torch.from_numpy(X)
    ag = torch.from_numpy(ag)
    y = torch.from_numpy(y)
    return feat_dataloader_tensor(torch.utils.data.TensorDataset(X, ag, y), train, device, batch_size)

def train_and_val_sets(split, train_set, strategy, device="cuda", batch_size=256):
    X, ag, y = split[train_set]
    #feature_dim = X.shape[1]
    out_channels = y.shape[1]
    if strategy == "classif":
        train_ds = feat_dataloader(X, y, train=True, device=device, batch_size=batch_size)
    else:
        train_ds = feat_ag_dataloader(X, ag, y, train=True, device=device, batch_size=batch_size)
    X, ag, y = split["val"]
    if strategy == "classif":
        val_ds = feat_dataloader(X, y, train=False, device=device, batch_size=batch_size)
    else:
        val_ds = feat_ag_dataloader(X, ag, y, train=False, device=device, batch_size=batch_size)
    return out_channels, train_ds, val_ds

def test_set(split, strategy, device="cuda", batch_size=256):
    X, ag, y = split["test"]
    out_channels = y.shape[1]
    if strategy == "classif":
        test_ds = feat_dataloader(X, y, train=False, device=device, batch_size=batch_size)
    else:
        test_ds = feat_ag_dataloader(X, ag, y, train=False, device=device, batch_size=batch_size)
    return out_channels, test_ds

# compact semi-unique identifier for a model
# usage is similar to UUID-s
def metadata_hash(metadata):
    s = json.dumps(metadata, sort_keys=True)
    h = zlib.crc32(bytes(s, encoding="utf-8"))
    return f"{h:x}"[-8:]

def save_model_split(model_name, silo_name, out_path, split_num, feats):
    path_pref = os.path.join(out_path, silo_name)
    os.makedirs(path_pref, exist_ok=True)
    filenames = {}
    for k in feats:
        X, ag, y = feats[k]
        out_fn = f"{model_name}_split{split_num}_{k}.npz"
        np.savez_compressed(os.path.join(path_pref, out_fn), X=X, ag=ag, y=y)
        filenames[k] = out_fn
    return filenames

# load for one model
def load_split_features(dir_pref, silo_metadata, split_num, strategy="classif"):
    split_data = {}
    filenames = silo_metadata["splits"][strategy][str(split_num)]
    path_pref = os.path.join(dir_pref, silo_metadata["silo_name"])
    for ds in filenames:
        with np.load(os.path.join(path_pref, filenames[ds])) as data:
            split_data[ds] = (data["X"], data["ag"], data["y"])
    test_y = split_data["test"][-1]
    return split_data, test_y

