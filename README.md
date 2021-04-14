# Driver-Drowsiness-Detection
A real-time drowsiness detection system for drivers, which alerts the driver if they fall asleep due to fatigue while still driving. The computer vision algorithm used for the implementation uses a trifold approach to detect drowsiness, including the measurement of forward head tilt angle, measurement of eye aspect ratio (to detect closure of eyes) and measurement of mouth aspect ratio (to detect yawning).

## Installation
The software runs on Python and primarily uses OpenCV to perform the head pose estimation and Dlib for facial feature landmark detection.

1. Install the python packages mentioned in 'Requirements.txt' file via the command "pip install -r Requirements.txt"
2. Execute the program through the CLI command "python Driver\ Drowsiness\ Detection.py"
3. Press the 'Q' key to stop the program execution
