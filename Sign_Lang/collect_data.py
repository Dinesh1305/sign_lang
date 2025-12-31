import cv2
import os

# CONFIGURATION: Only collecting for 'hello'
labels = ['done']
base_dir = "dataset"

if not os.path.exists(base_dir):
    os.makedirs(base_dir)

for label in labels:
    path = os.path.join(base_dir, label)
    if not os.path.exists(path):
        os.makedirs(path)

cap = cv2.VideoCapture(0)

print(f"Collecting data for: {labels}")
print("Press 's' to save an image.")
print("Press 'q' to quit.")

current_label_index = 0
count = 0

# Check existing files to avoid overwriting if you run this multiple times
save_path = os.path.join(base_dir, labels[0])
existing_files = os.listdir(save_path)
if existing_files:
    # Set count to the number of existing files so we append instead of overwrite
    count = len(existing_files)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # ROI box
    roi = frame[100:350, 100:350]
    cv2.rectangle(frame, (100, 100), (350, 350), (255, 0, 0), 2)

    # Display information
    current_label = labels[0]
    cv2.putText(frame, f"Collecting: {current_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Saved Count: {count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Data Collection - Hello Only", frame)

    key = cv2.waitKey(1) & 0xFF

    # Press 's' to save
    if key == ord('s'):
        img_save = cv2.resize(roi, (64, 64))

        filename = f"{base_dir}/{current_label}/{current_label}_{count}.jpg"
        cv2.imwrite(filename, img_save)
        print(f"Saved {filename}")
        count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()