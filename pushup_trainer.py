# import os
# import cv2
# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, TimeDistributed
# from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import train_test_split

# # Paths
# dataset_dir = "Dataset"
# categories = ["Correct sequence", "Wrong sequence"]

# IMG_SIZE = 64        # Resize frames
# SEQUENCE_LENGTH = 20 # Number of frames per video
# EPOCHS = 10          # Increase later (for real training)

# def extract_frames(video_path, max_frames=SEQUENCE_LENGTH):
#     frames = []
#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frame_interval = max(1, total_frames // max_frames)

#     while len(frames) < max_frames:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
#         frame = frame / 255.0
#         frames.append(frame)
#         for _ in range(frame_interval - 1):
#             cap.read()  # skip frames

#     cap.release()
#     return np.array(frames)

# # Load data
# data, labels = [], []
# for idx, category in enumerate(categories):
#     folder = os.path.join(dataset_dir, category)
#     for file in os.listdir(folder):
#         if file.endswith(".mp4"):
#             path = os.path.join(folder, file)
#             frames = extract_frames(path)
#             if len(frames) == SEQUENCE_LENGTH:
#                 data.append(frames)
#                 labels.append(idx)
#                 print(f"Loaded {file} from {category}")

# data = np.array(data)
# labels = to_categorical(np.array(labels), num_classes=len(categories))

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# # CNN + LSTM model
# model = Sequential([
#     TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3)),
#     TimeDistributed(MaxPooling2D(2, 2)),
#     TimeDistributed(Conv2D(64, (3, 3), activation='relu')),
#     TimeDistributed(MaxPooling2D(2, 2)),
#     TimeDistributed(Flatten()),
#     LSTM(64),
#     Dense(32, activation='relu'),
#     Dense(len(categories), activation='softmax')
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# print("\n✅ Training started...")
# model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_test, y_test))

# model.save("pushup_cnn_lstm.h5")
# print("\n✅ Model saved successfully as pushup_cnn_lstm.h5")




















import cv2
import mediapipe as mp
import numpy as np
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split

# =============================================
# SETUP
# =============================================
DATASET_PATH = "Dataset"
MODEL_PATH = "pushup_cnn_lstm.h5"
IMG_SIZE = 64
EPOCHS = 10
BATCH_SIZE = 8

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# =============================================
# DATA PREPARATION
# =============================================

def load_video_frames_from_folder(folder, label):
    frames = []
    labels = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        cap = cv2.VideoCapture(file_path)
        frame_list = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = img_to_array(frame) / 255.0
            frame_list.append(frame)
        cap.release()
        if len(frame_list) > 0:
            frames.append(frame_list)
            labels.append(label)
    return frames, labels

def load_dataset():
    print("📂 Loading dataset...")
    correct_frames, correct_labels = load_video_frames_from_folder(os.path.join(DATASET_PATH, "Correct sequence"), 1)
    wrong_frames, wrong_labels = load_video_frames_from_folder(os.path.join(DATASET_PATH, "Wrong sequence"), 0)

    frames = correct_frames + wrong_frames
    labels = correct_labels + wrong_labels

    # Pad/truncate sequences to same length
    max_len = max(len(f) for f in frames)
    X = np.zeros((len(frames), max_len, IMG_SIZE, IMG_SIZE, 3))
    for i, seq in enumerate(frames):
        for j in range(min(len(seq), max_len)):
            X[i, j] = seq[j]
    y = np.array(labels)
    print("✅ Dataset loaded successfully.")
    return X, y

# =============================================
# MODEL CREATION
# =============================================

def create_cnn_lstm_model(input_shape):
    model = Sequential([
        TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=input_shape),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Conv2D(64, (3, 3), activation='relu')),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Flatten()),
        LSTM(64),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# =============================================
# TRAINING / LOADING
# =============================================

if os.path.exists(MODEL_PATH):
    print("🧠 Pre-trained model found! Loading existing model...")
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
else:
    print("⚙️ No pre-trained model found. Starting training...")
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    input_shape = (X_train.shape[1], IMG_SIZE, IMG_SIZE, 3)
    model = create_cnn_lstm_model(input_shape)

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)
    model.save(MODEL_PATH)
    print(f"✅ Model trained and saved as {MODEL_PATH}")

# =============================================
# REAL-TIME PUSHUP DETECTION
# =============================================

print("🚀 Starting real-time push-up detection...")

cap = cv2.VideoCapture(0)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("AI Push-up Trainer", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
