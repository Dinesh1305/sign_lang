import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("asl_model.h5")

# LABELS: Must match training output {'A': 0, 'hello': 1}
labels = ['A', 'hello']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # ROI box
    roi = frame[100:350, 100:350]

    # Preprocess
    img = cv2.resize(roi, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    preds = model.predict(img)
    class_id = np.argmax(preds)
    pred_label = labels[class_id]
    confidence = preds[0][class_id]

    # --- CHECK FOR HELLO LOGIC ---
    if pred_label == 'hello':
        # Green box and text if Hello is found
        color = (0, 255, 0)
        display_text = "HELLO DETECTED!"
    else:
        # Red box for other letters (A)
        color = (0, 0, 255)
        display_text = f"{pred_label} ({confidence:.2f})"

    # Draw the rectangle and text
    cv2.rectangle(frame, (100, 100), (350, 350), color, 2)
    cv2.putText(frame, display_text, (100, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("SIGN LANGUAGE RECOGNITION", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()