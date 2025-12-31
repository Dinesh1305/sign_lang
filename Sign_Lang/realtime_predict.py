import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("digit_detector.keras")
labels = list(model.signatures.keys()) if hasattr(model, 'signatures') else None

# OR we define manually:
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    roi = frame[150:350, 150:350]  # smaller box

    img = cv2.resize(roi, (64, 64)) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]
    class_id = np.argmax(preds)
    digit = labels[class_id]
    confidence = preds[class_id]

    color = (0, 255, 0)
    cv2.rectangle(frame, (100, 100), (350, 350), color, 2)
    cv2.putText(frame, f"{digit} ({confidence:.2f})", (100, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Digit Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
