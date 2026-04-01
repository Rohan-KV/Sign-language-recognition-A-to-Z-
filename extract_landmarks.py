import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import mediapipe as mp

# ==============================
# CONFIG
# ==============================
DATASET_PATH = "Dataset/train"   # change to Dataset/test later
OUTPUT_FILE = "train_landmarks.csv"
MAX_IMAGES_PER_CLASS = None  # set to limit (e.g., 500) or keep None

# ==============================
# INIT MEDIAPIPE
# ==============================
mp_hands = mp.solutions.hands

# ==============================
# FUNCTION: EXTRACT LANDMARKS
# ==============================
def extract_landmarks(image, hands):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None

    landmarks = results.multi_hand_landmarks[0]

    # Normalize relative to wrist (landmark 0)
    base_x = landmarks.landmark[0].x
    base_y = landmarks.landmark[0].y
    base_z = landmarks.landmark[0].z

    data = []
    for lm in landmarks.landmark:
        data.extend([
            lm.x - base_x,
            lm.y - base_y,
            lm.z - base_z
        ])

    return data


# ==============================
# MAIN
# ==============================
def main():
    all_data = []
    skipped_images = 0
    total_images = 0

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:

        for label in sorted(os.listdir(DATASET_PATH)):
            label_path = os.path.join(DATASET_PATH, label)

            if not os.path.isdir(label_path):
                continue

            images = os.listdir(label_path)

            if MAX_IMAGES_PER_CLASS:
                images = images[:MAX_IMAGES_PER_CLASS]

            print(f"\n🔹 Processing class: {label} ({len(images)} images)")

            for img_name in tqdm(images):
                img_path = os.path.join(label_path, img_name)

                image = cv2.imread(img_path)
                if image is None:
                    skipped_images += 1
                    continue

                total_images += 1

                landmarks = extract_landmarks(image, hands)

                if landmarks is None:
                    skipped_images += 1
                    continue

                # Append label
                landmarks.append(label)
                all_data.append(landmarks)

    # ==============================
    # SAVE CSV
    # ==============================
    df = pd.DataFrame(all_data)

    # Optional: name columns
    columns = []
    for i in range(21):
        columns += [f"x{i}", f"y{i}", f"z{i}"]
    columns.append("label")

    df.columns = columns

    df.to_csv(OUTPUT_FILE, index=False)

    print("\n✅ DONE!")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Total images processed: {total_images}")
    print(f"Skipped (no hand detected): {skipped_images}")
    print(f"Final dataset size: {len(df)}")


# ==============================
# ENTRY POINT
# ==============================
if __name__ == "__main__":
    main()