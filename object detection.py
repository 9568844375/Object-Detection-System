import cv2
import matplotlib.pyplot as plt
import os

# Model paths (Ensure the files exist)
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph(1).pb'



# Load the model
model = cv2.dnn_DetectionModel(frozen_model, config_file)

# Load class labels (Ensure the file exists)
file_name = "labels.txt"
if not os.path.exists(file_name):
    raise FileNotFoundError(f"Label file '{file_name}' not found!")

with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

print("Loaded class labels:", classLabels)
print("Number of classes:", len(classLabels))

# Set model parameters
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# Load image
img_path = "boy.jpg"
if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image file '{img_path}' not found!")

img = cv2.imread(img_path)

# Check if image is loaded correctly
if img is None:
    raise ValueError("Error loading image. Check the file format and path.")

# Convert image to RGB for matplotlib display
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# Object detection on the image
ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.5)
print("Detected Objects:", ClassIndex)

# Define font settings
font_scale = 1
font = cv2.FONT_HERSHEY_PLAIN

# Draw bounding boxes on image
for ClassInd, conf, box in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    if 1 <= ClassInd <= len(classLabels):  # Ensure valid class index
        cv2.rectangle(img, box, (0, 255, 0), 2)
        cv2.putText(img, classLabels[ClassInd - 1], (box[0] + 10, box[1] + 40),
                    font, font_scale, (0, 0, 255), thickness=2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# Video Processing
video_path = "pexels_geoge_morina-5330833.mp4"
cap = cv2.VideoCapture(video_path)

# If video file not found, switch to webcam
if not cap.isOpened():
    print("Video file not found. Switching to webcam...")
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open video or webcam!")

# Video processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or camera error.")
        break  # Exit loop if video ends

    # Object detection on video frame
    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)

    if len(ClassIndex) != 0:
        for ClassInd, conf, box in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if 1 <= ClassInd <= len(classLabels):  # Ensure valid class index
                cv2.rectangle(frame, box, (0, 255, 0), 2)
                cv2.putText(frame, classLabels[ClassInd - 1], (box[0] + 10, box[1] + 40),
                            font, font_scale, (0, 0, 255), thickness=2)

    cv2.imshow("Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
