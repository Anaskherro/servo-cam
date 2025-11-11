"""
Advanced usage examples for Jetson Servo Camera Tracker
"""

# Example 1: Custom servo angle limits and PID tuning
from main import ServoCamera

def example_custom_tuning():
    """Configure tracker with custom angle limits and PID gains"""
    tracker = ServoCamera(
        model_name='yolov5su',
        camera_index=0,
        pan_pin=33,
        target_class=0,
        confidence_threshold=0.45,  # Lower threshold for more detections
        use_gpu=True
    )
    
    # Customize angle limits (e.g., servo doesn't rotate full 180°)
    tracker.pan_min = 45
    tracker.pan_max = 135
    
    # Increase PID gain for faster tracking
    tracker.kp_pan = 0.8
    
    tracker.run(display=True)


# Example 2: Using CSI camera with GStreamer
def example_csi_camera():
    """Use Jetson CSI camera with GStreamer pipeline"""
    import cv2
    
    # GStreamer pipeline for Jetson CSI camera
    pipeline = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
    )
    
    tracker = ServoCamera(
        model_name='yolov5su',
        camera_index=pipeline,  # GStreamer pipeline
        pan_pin=33,
        confidence_threshold=0.5,
        use_gpu=True
    )
    
    tracker.run(display=True)


# Example 3: Headless mode (no display)
def example_headless():
    """Run tracker without display (useful for deployment)"""
    tracker = ServoCamera(
        model_name='yolov5su',
        camera_index=0,
        pan_pin=33,
        target_class=0,
        confidence_threshold=0.5,
        use_gpu=True
    )
    
    # Run without display output
    tracker.run(display=False)


# Example 4: Manual servo angle control (for testing)
def example_test_servo():
    """Test servo motor with manual angles"""
    from main import JetsonPWMServo
    import time
    
    servo = JetsonPWMServo(pan_pin=33, frequency=50)
    
    # Test pan servo (0° to 180°)
    print("Testing pan servo...")
    for angle in [0, 45, 90, 135, 180]:
        servo.set_angles(angle)
        print(f"Pan: {angle}°")
        time.sleep(1)
    
    # Return to center
    servo.set_angles(90)
    servo.cleanup()
    print("Done!")


# Example 5: Low-latency configuration
def example_low_latency():
    """Configure for minimum latency tracking"""
    tracker = ServoCamera(
        model_name='yolov5n',  # Smaller model = faster
        camera_index=0,
        pan_pin=33,
        target_class=0,
        confidence_threshold=0.6,  # Higher threshold = fewer detections = faster
        use_gpu=True
    )
    
    # Reduce smoothing buffer for faster response
    tracker.bbox_buffer = tracker.bbox_buffer.__class__(maxlen=2)
    
    # Increase PID gain for immediate response
    tracker.kp_pan = 1.0
    
    tracker.run(display=True)


# Example 6: High accuracy configuration
def example_high_accuracy():
    """Configure for maximum detection accuracy"""
    tracker = ServoCamera(
        model_name='yolov5su',  # Larger model = better accuracy
        camera_index=0,
        pan_pin=33,
        target_class=0,
        confidence_threshold=0.3,  # Lower threshold = more detections
        use_gpu=True
    )
    
    # Increase smoothing for stable tracking
    # (default maxlen=5 is good)
    
    # Reduce PID gain for smoother motion
    tracker.kp_pan = 0.3
    
    tracker.run(display=True)


# Example 7: Multiple person tracking (track nearest person)
def example_multi_person():
    """
    Tracker automatically tracks the person closest to frame center.
    To track a different person, manually adjust the frame or modify
    the distance calculation in detect_persons() method.
    """
    tracker = ServoCamera(
        model_name='yolov5su',
        camera_index=0,
        pan_pin=33,
        target_class=0,
        confidence_threshold=0.5,
        use_gpu=True
    )
    
    # The detect_persons() method returns the person closest to center
    # Modify calculate_servo_angles() to change tracking strategy
    
    tracker.run(display=True)


if __name__ == "__main__":
    # Choose which example to run
    print("Jetson Servo Camera Tracker - Examples")
    print("1. Custom tuning")
    print("2. CSI camera")
    print("3. Headless mode")
    print("4. Test servo")
    print("5. Low latency")
    print("6. High accuracy")
    print("7. Multi-person")
    
    choice = input("Select example (1-7): ").strip()
    
    examples = {
        '1': example_custom_tuning,
        '2': example_csi_camera,
        '3': example_headless,
        '4': example_test_servo,
        '5': example_low_latency,
        '6': example_high_accuracy,
        '7': example_multi_person,
    }
    
    if choice in examples:
        examples[choice]()
    else:
        print("Invalid choice!")
