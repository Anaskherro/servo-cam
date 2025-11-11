# Jetson Servo Camera Tracker

Real-time person detection and tracking with servo motor control using YOLOv5su on Jetson devices.

## Features

- **YOLOv5su Detection**: Optimized YOLOv5 model for better performance on Jetson
- **PWM Servo Control**: Direct GPIO PWM control for pan servo
- **GPU Acceleration**: CUDA support for fast inference
- **Person Tracking**: Automatically keeps detected persons centered horizontally
- **Real-time Display**: Visual feedback with bounding boxes and servo angle

## Hardware Requirements

- Jetson device (Xavier, Nano, Orin, etc.)
- USB camera or CSI camera
- 1x servo motor (pan)
- Servo power supply (typically 5-6V)
- Jumper wires to connect GPIO

## Installation

### 1. Install Dependencies

```bash
# Install pip packages
pip install -r requirements.txt

# Note: Jetson.GPIO should already be available in NVIDIA Jetson Docker containers
# If not, install it:
pip install jetson-gpio
```

### 2. GPIO Pin Configuration

Update the GPIO pin number in `main.py`:
- `pan_pin`: GPIO pin for horizontal servo (default: 33)

Check your Jetson pinout documentation to find available GPIO pins.

### 3. Servo Calibration

The servo angle-to-duty cycle conversion assumes standard servo specs:
- 0° = 5% duty cycle (1ms pulse)
- 90° = 7.5% duty cycle (1.5ms pulse)
- 180° = 10% duty cycle (2ms pulse)

If your servos use different specs, modify the `angle_to_duty_cycle()` method in `JetsonPWMServo` class.

## Usage

```bash
# Run the tracker
python main.py
```

### Configuration

Edit the parameters in the `__main__` section:

```python
tracker = ServoCamera(
    model_name='yolov5su',       # YOLOv5 model (yolov5su recommended)
    camera_index=0,              # Camera index (0 = default, or GStreamer pipeline)
    pan_pin=33,                  # GPIO pin for pan servo
    target_class=0,              # Class 0 = person (COCO dataset)
    confidence_threshold=0.5,    # Detection confidence (0-1)
    use_gpu=True                 # Use GPU acceleration
)
```

### Tuning

Adjust tracking responsiveness with PID gain:

```python
# In ServoCamera.__init__()
self.kp_pan = 0.5   # Pan responsiveness (increase = faster response)
```

### Controls

- **q**: Quit the application
- **Ctrl+C**: Interrupt and exit

## Visual Feedback

- **Green crosshair**: Frame center (reference)
- **Green box**: Detected person
- **Green dot**: Person center
- **Red crosshair**: Searching (no person detected)
- **Pan angle display**: Current pan servo position
- **FPS counter**: Real-time performance metric

## Performance Tips

1. **Model Size**: Use `yolov5su` or smaller for better FPS
2. **Confidence Threshold**: Increase to reduce false positives (trades recall)
3. **Buffer Size**: Camera buffer set to 1 for lowest latency
4. **GPU**: Ensure CUDA is enabled for faster inference

## Example GStreamer Pipeline (for CSI Camera)

If using a CSI camera on Jetson, use a GStreamer pipeline:

```python
camera_pipeline = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
)

tracker = ServoCamera(
    model_name='yolov5su',
    camera_index=camera_pipeline,  # Use pipeline instead of 0
    # ... other parameters
)
```

## Troubleshooting

### GPIO Permission Errors
```bash
# Run with sudo
sudo python main.py
```

### CUDA Out of Memory
- Reduce image resolution
- Use smaller model (yolov5n instead of yolov5su)
- Reduce confidence threshold

### Servo Not Moving
- Check GPIO pin numbers
- Verify servo power supply
- Test GPIO with blink example first
- Check servo PWM frequency (50Hz for standard servos)

## Notes

- First frame may take longer due to model loading
- The tracker continuously adjusts servo angle based on person position
- If no person is detected, servo maintains last known position
- Bounding box smoothing (5-frame buffer) reduces jitter

## License

MIT

## References

- [YOLOv5 Documentation](https://github.com/ultralytics/yolov5)
- [Jetson GPIO Documentation](https://github.com/NVIDIA/jetson-gpio)
- [COCO Dataset Classes](https://cocodataset.org/)
