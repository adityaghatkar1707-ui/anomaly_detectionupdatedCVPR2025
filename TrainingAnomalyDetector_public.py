import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Disable GPU usage/use only CPU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import glob
import csv
from os import listdir
from os.path import isfile, join, basename
from datetime import datetime

print("🚀 Script started successfully — now loading model and data...")

import numpy as np
from scipy.io import loadmat, savemat
import tensorflow as tf

def load_dataset_Train_batch(AbnormalPath, NormalPath):
    print("Loading training batch...")

    # Dynamically count how many .txt files are actually present
    Abnormal_files = [f for f in os.listdir(AbnormalPath) if f.endswith(".txt")]
    Normal_files = [f for f in os.listdir(NormalPath) if f.endswith(".txt")]

    Num_abnormal = len(Abnormal_files)
    Num_Normal = len(Normal_files)

    if Num_abnormal == 0 or Num_Normal == 0:
        raise ValueError(
            f"No .txt feature files found! Abnormal={Num_abnormal}, Normal={Num_Normal}"
        )

    # Automatically set batch size to avoid out-of-range errors
    batchsize = min(2, Num_abnormal + Num_Normal)
    n_exp = batchsize // 2  # Number of abnormal and normal per batch

    print(f"Found {Num_abnormal} abnormal and {Num_Normal} normal files.")
    print(f"Using batchsize={batchsize} (Abnormal={n_exp}, Normal={n_exp})")

    # Helper to list *_C.txt files
    def listdir_nohidden(folder_path):
        file_dir_extension = os.path.join(folder_path, "*_C.txt")
        return sorted(
            [
                os.path.basename(f)
                for f in glob.glob(file_dir_extension)
                if not os.path.basename(f).startswith(".")
            ]
        )

    # ---------------- Abnormal ----------------
    print("Loading Abnormal videos Features...")
    AllVideos_Path = AbnormalPath
    All_Videos = listdir_nohidden(AllVideos_Path)

    AllFeatures = None
    video_count = 0

    # Randomly pick available indexes safely
    Abnor_list_iter = np.random.choice(range(len(All_Videos)), size=min(n_exp, len(All_Videos)), replace=False)

    for iv in Abnor_list_iter:
        VideoPath = os.path.join(AllVideos_Path, All_Videos[iv])
        with open(VideoPath, "r") as f:
            words = f.read().split()

        num_feat = len(words) // 4096  # 32 segments per video
        VideoFeatures = np.array(words, dtype=np.float32).reshape(num_feat, 4096)

        if AllFeatures is None:
            AllFeatures = VideoFeatures
        else:
            AllFeatures = np.vstack((AllFeatures, VideoFeatures))

        print(f"  Abnormal features loaded: {All_Videos[iv]}")

    # ---------------- Normal ----------------
    print("Loading Normal videos Features...")
    AllVideos_Path = NormalPath
    All_Videos = listdir_nohidden(AllVideos_Path)

    Norm_list_iter = np.random.choice(range(len(All_Videos)), size=min(n_exp, len(All_Videos)), replace=False)

    for iv in Norm_list_iter:
        VideoPath = os.path.join(AllVideos_Path, All_Videos[iv])
        with open(VideoPath, "r") as f:
            words = f.read().split()

        num_feat = len(words) // 4096
        VideoFeatures = np.array(words, dtype=np.float32).reshape(num_feat, 4096)
        AllFeatures = np.vstack((AllFeatures, VideoFeatures))

        print(f"  Normal features loaded: {All_Videos[iv]}")

    print("All features loaded successfully.")

    # ---------------- Labels ----------------
    AllLabels = np.zeros(32 * batchsize, dtype="uint8")
    th_loop1 = n_exp * 32
    th_loop2 = n_exp * 32 - 1

    for iv in range(32 * batchsize):
        if iv < th_loop1:
            AllLabels[iv] = 0  # abnormal
        if iv > th_loop2:
            AllLabels[iv] = 1  # normal

        # Safety check — make sure data loaded properly
    if AllFeatures is None or len(AllFeatures) == 0:
        raise ValueError("❌ No valid feature data loaded. Check your *_C.txt files in Abnormal/Normal folders.")

    print(f"✅ Loaded feature array shape: {AllFeatures.shape}")
    return AllFeatures, AllLabels
  






# ----------------------------------------
# ✅ Step 1: Confirm imports done
print("✅ Step 1: Imports and function definitions complete")

# ----------------------------------------
# ✅ Step 2: Define Model
print("✅ Step 2: Defining model...")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adagrad
import numpy as np

model = Sequential()
model.add(Dense(512, input_dim=4096, kernel_initializer="glorot_normal",
                kernel_regularizer=l2(0.001), activation="relu"))
model.add(Dropout(0.6))
model.add(Dense(32, kernel_initializer="glorot_normal",
                kernel_regularizer=l2(0.001), activation="linear"))
model.add(Dropout(0.6))
model.add(Dense(1, kernel_initializer="glorot_normal",
                kernel_regularizer=l2(0.001), activation="sigmoid"))

print("✅ Step 3: Model created successfully")

# ----------------------------------------
# ✅ Step 3: Compile Model
adagrad = Adagrad(learning_rate=0.01, epsilon=1e-08)
model.compile(loss="binary_crossentropy", optimizer=adagrad)
print("✅ Step 4: Model compiled successfully")

# ----------------------------------------
# ✅ Step 4: Load Data Paths
AllClassPath = r"C:\Users\ADITYA\OneDrive\Desktop\anomaly_detection\AnomalyDetectionCVPR2018\C3D_Features_Txt\Train"
AbnormalPath = os.path.join(AllClassPath, "Abnormal")
NormalPath = os.path.join(AllClassPath, "Normal")

print("✅ Step 5: Loading data...")
inputs, targets = load_dataset_Train_batch(AbnormalPath, NormalPath)
print("✅ Step 6: Data loaded successfully")

# ----------------------------------------
# ✅ Step 5: Training
print("✅ Step 7: Starting training on CPU...")
history = model.fit(inputs, targets, batch_size=8, epochs=5, verbose=1)

print("🎉 Training completed successfully!")


# ----------------------------------------
# ✅ Save trained model
model.save("trained_anomaly_model.h5")
print("💾 Model saved as trained_anomaly_model.h5")

