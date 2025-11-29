import os
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from scipy.io import savemat

print("🚀 Starting anomaly testing...")

# -----------------------------
# 1️⃣ Paths (edit these if needed)
# -----------------------------
# Folder containing your 32x4096 *_C.txt feature files for test videos
AllTest_Video_Path = r"C:\Users\ADITYA\OneDrive\Desktop\anomaly_detection\AnomalyDetectionCVPR2018\C3D_Features_Txt\Test"

# Folder where you want to save predictions (.mat files)
Results_Path = r"C:\Users\ADITYA\OneDrive\Desktop\anomaly_detection\AnomalyDetectionCVPR2018\Eval_Res"

# Your trained model path (from training step)
Model_path = r"C:\Users\ADITYA\OneDrive\Desktop\anomaly_detection\AnomalyDetectionCVPR2018\trained_anomaly_model.h5"

# -----------------------------
# 2️⃣ Create output directory if missing
# -----------------------------
os.makedirs(Results_Path, exist_ok=True)

# -----------------------------
# 3️⃣ Load trained model
# -----------------------------
print("✅ Loading trained model...")
model = load_model(Model_path)
print("✅ Model loaded successfully.")

# -----------------------------
# 4️⃣ Helper to load one video’s features
# -----------------------------
def load_dataset_One_Video_Features(Test_Video_Path):
    with open(Test_Video_Path, "r") as f:
        words = f.read().split()
    num_feat = len(words) // 4096
    VideoFeatures = np.array(words, dtype=np.float32).reshape(num_feat, 4096)
    return VideoFeatures

# -----------------------------
# 5️⃣ Loop through test files
# -----------------------------
All_Test_files = sorted([
    f for f in os.listdir(AllTest_Video_Path)
    if f.endswith(".txt") or f.endswith("_C.txt")
])

print(f"🧾 Found {len(All_Test_files)} test feature files.")
time_before = datetime.now()

for iv, fname in enumerate(All_Test_files, 1):
    Test_Video_Path = os.path.join(AllTest_Video_Path, fname)
    inputs = load_dataset_One_Video_Features(Test_Video_Path)

    # Predict anomaly scores (one score per segment)
    predictions = model.predict(inputs, verbose=0)
    avg_score = float(np.mean(predictions))

    # Save results
    video_name = os.path.splitext(fname)[0]
    result_path = os.path.join(Results_Path, f"{video_name}.mat")
    savemat(result_path, {"predictions": predictions})

    # Display result
    label = "ABNORMAL" if avg_score > 0.5 else "NORMAL"
    print(f"[{iv}/{len(All_Test_files)}] {video_name}: "
          f"mean_score={avg_score:.4f} → {label}")

print("✅ All test videos processed.")
print("⏱️ Total time:", datetime.now() - time_before)

