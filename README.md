🚀 Anomaly Detection in Surveillance Videos (Updated CVPR 2025)

This is an updated and working version of the CVPR 2018 Anomaly Detection project by Waqas Sultani et al., rewritten for modern Python 3.13, TensorFlow 2.20.0, and Keras.
The original code used Theano and outdated Keras 1.x; this version runs successfully on Windows using CPU.

🧠 About the Project

The system detects abnormal activities in surveillance videos (e.g., accidents, explosions, thefts) using C3D feature representations.
Our implementation performs both training and testing from pre-computed feature text files and achieves accurate anomaly prediction via CMD interface.

📂 Folder Structure
AnomalyDetectionUpdatedCVPR2025/
│
├── C3D_Features_Txt/
│   ├── Train/
│   │   ├── Abnormal/
│   │   └── Normal/
│   └── Test/
│
├── TrainingAnomalyDetector_public.py
├── Test_Anomaly_Detector_public.py
├── trained_anomaly_model.h5
└── README.md

🧩 Dataset (Existing Pre-computed)

This implementation uses C3D pre-computed feature datasets from the original CVPR 2018 work:
🔗 UCF Anomaly Detection Dataset (Official Link)

Each video is divided into 32 segments, and each segment contains a 4096-dimensional feature vector extracted from a C3D model.

⚙️ Setup & Run
1️⃣ Install dependencies
pip install tensorflow numpy scipy

2️⃣ Train model
python TrainingAnomalyDetector_public.py

3️⃣ Test model
python Test_Anomaly_Detector_public.py

🧾 Example Output (CMD)
🧾 Found 9 test feature files.
[1/9] Explosion008_C: mean_score=0.5379 → ABNORMAL
[9/9] Shoplifting028_C: mean_score=0.4916 → NORMAL
✅ All test videos processed.

📈 Improvements (Over Original 2018 Version)
Feature	Original (2018)	Updated (2025)
Backend	Theano	TensorFlow 2.20
Keras Version	1.1.0	Modern Keras API
Compatibility	Linux only	Windows + CPU compatible
Code Quality	Legacy	Clean, modular, and debugged
GPU Dependency	Required	Optional (CPU-only mode added)


🧩 Future Scope

Real-time visualization and alert system

Integration with LPR (License Plate Recognition)

Object and behavior detection

Dashboard for anomaly analytics

Migration from C3D → I3D / Transformer-based models

🧑‍💻 Author

Aditya Ghatkar
Third-Year Engineering Student | Research Enthusiast (AI & Deep Learning)

🧾 Reference

Sultani, W., Chen, C., & Shah, M. (2018). Real-World Anomaly Detection in Surveillance Videos.
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).



