import cv2
import numpy as np
import pandas as pd
import os
color_ranges = {
    "Red": ([0, 120, 70], [10, 255, 255]),
    "Red2": ([170, 120, 70], [180, 255, 255]),  
    "Green": ([40, 40, 70], [80, 255, 255]),
    "Blue": ([90, 50, 70], [130, 255, 255]),
    "Yellow": ([20, 100, 100], [30, 255, 255]),
    "Purple": ([130, 50, 50], [160, 255, 255]),
    "Cyan": ([80, 100, 100], [90, 255, 255]),
    "Magenta": ([140, 100, 100], [170, 255, 255]),
    "White": ([0, 0, 200], [180, 50, 255]),     
    "Orange": ([5, 150, 150], [15, 255, 255]),   
    "DarkGreen": ([50, 100, 100], [70, 255, 255]) 
}
def track_balls(hsv_frame, lower, upper):
    mask = cv2.inRange(hsv_frame, np.array(lower), np.array(upper))
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    balls = []
    for contour in contours:
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        area = cv2.contourArea(contour)
        if 20 < radius < 100 and area / (np.pi * radius ** 2) > 0.5: 
            balls.append((int(x), int(y), int(radius)))
    return balls
def get_quadrant(x, y, width, height, buffer=20):
    if x < width // 2 - buffer and y < height // 2 - buffer:
        return 1
    elif x >= width // 2 + buffer and y < height // 2 - buffer:
        return 2
    elif x < width // 2 - buffer and y >= height // 2 + buffer:
        return 3
    else:
        return 4

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_video_path = os.path.join('Path of directory', f'{base_name}_processed.mp4')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    event_data = []
    prev_quadrants = {color: None for color in color_ranges.keys()}
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for color_name, (lower, upper) in color_ranges.items():
            balls = track_balls(hsv, lower, upper)
            for (x, y, r) in balls:
                quadrant = get_quadrant(x, y, width, height)

                timestamp = frame_number / fps
                prev_quadrant = prev_quadrants[color_name]
                if prev_quadrant != quadrant:
                    event_type = 'Exit' if prev_quadrant else 'Entry'
                    event_data.append([timestamp, quadrant, color_name, event_type])
                    prev_quadrants[color_name] = quadrant
                    cv2.putText(frame, f"{color_name} Ball {event_type} Q{quadrant} at {timestamp:.2f}s", 
                                (x - r, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Frame', width, height)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        out.write(frame)
        frame_number += 1
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    event_data_sorted = sorted(event_data, key=lambda x: x[0])
    output_log_path = os.path.join('C:\\Users\\Administrator\\Desktop\\Final\\Internships\\AI_internship_task', f'{base_name}_event_log.txt')
    with open(output_log_path, 'w') as file:
        for event in event_data_sorted:
            file.write(f"{event[0]}\t{event[1]}\t{event[2]}\t{event[3]}\n")

    print(f"Processing completed. Processed video saved as {output_video_path}. Event log saved as {output_log_path}.")
video_path = 'AI Assignment video.mp4'  
main(video_path)
