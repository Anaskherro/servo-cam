# Configuration file for Jetson Servo Camera Tracker

# ===== CAMERA SETTINGS =====
camera_index = 0  # Camera index (0=default) or GStreamer pipeline string
frame_width = 1280
frame_height = 720

# ===== YOLO MODEL SETTINGS =====
model_name = "yolov5su"  # Options: yolov5n, yolov5s, yolov5m, yolov5l, yolov5x, yolov5su
use_gpu = True  # Use GPU acceleration (CUDA)
target_class = 0  # Class 0 = person (COCO dataset)
confidence_threshold = 0.5  # Detection confidence (0-1)

# ===== SERVO GPIO SETTINGS =====
pan_pin = 33  # GPIO pin for pan servo (left-right)
servo_frequency = 50  # PWM frequency in Hz (standard servo: 50Hz)

# ===== SERVO ANGLE LIMITS =====
pan_min = 0  # Minimum pan angle (degrees)
pan_max = 180  # Maximum pan angle (degrees)

pan_neutral = 90  # Neutral pan angle (center position)

# ===== PID CONTROL SETTINGS =====
kp_pan = 0.5  # Pan proportional gain (responsiveness)

# Tuning tips:
# - Increase kp for faster response
# - Decrease kp for smoother motion
# - Typical range: 0.1 to 1.0

# ===== DISPLAY SETTINGS =====
display_enabled = True  # Show live video with detections
show_bounding_box = True  # Draw bounding box around detected person
show_center_crosshair = True  # Draw frame center reference
show_servo_angles = True  # Display current servo angles
show_fps = True  # Display frames per second

# ===== SMOOTHING SETTINGS =====
bbox_buffer_size = 5  # Number of frames for bounding box smoothing
# Larger values = smoother but more latency
# Smaller values = faster but more jitter

# ===== PERFORMANCE TUNING =====
camera_buffer_size = 1  # Camera frame buffer (1 = minimum latency)
# Lower = lower latency but may drop frames
# Higher = more buffering but higher latency

# GStreamer CSI Camera Pipeline Example (Jetson)
# Uncomment to use instead of camera_index
# camera_index = (
#     "nvarguscamerasrc ! "
#     "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
#     "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
# )
