import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("mnist_model.keras")

url = "http://192.168.0.202:8080/videofeed"

# Start webcam
cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert full frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize full frame to 28x28
    resized = cv2.resize(gray, (28, 28))

    # Threshold for cleaner digit (better than simple invert)
    _, resized = cv2.threshold(resized, 120, 255, cv2.THRESH_BINARY_INV)

    # Normalize
    resized = resized.astype("float32") / 255.0

    # Reshape to (1,28,28,1)
    input_img = resized.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(input_img, verbose=0)
    digit = np.argmax(prediction)
    confidence = np.max(prediction)

    # Display prediction
    cv2.putText(frame, f"Prediction: {digit} ({confidence:.2f})",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,0,255), 2)

    cv2.imshow("Live MNIST Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
