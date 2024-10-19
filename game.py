from ultralytics import YOLO
import cv2
import numpy as np
import time

# Load model
model = YOLO('.\\best.pt')

# Initialize
grid = np.zeros((3, 3), dtype=str)
turn = 'X'
score = {'X': 0, 'O': 0, 'Draws': 0}

def draw_grid(frame):
    h, w, _ = frame.shape
    # Draw grid lines
    cv2.line(frame, (w//3, 0), (w//3, h), (255, 255, 255), 2)
    cv2.line(frame, (2*w//3, 0), (2*w//3, h), (255, 255, 255), 2)
    cv2.line(frame, (0, h//3), (w, h//3), (255, 255, 255), 2)
    cv2.line(frame, (0, 2*h//3), (w, 2*h//3), (255, 255, 255), 2)
    # Display score on screen
    cv2.putText(frame, f"player_1: {score['X']}", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"player_2: {score['O']}", (w//3+10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Draws: {score['Draws']}", (2 *w//3+10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Draw 'X' or 'O' on the grid
    for i in range(3):
        for j in range(3):
            if grid[i][j] != '':
                center_x = int(j * w // 3 + w // 6)
                center_y = int(i * h // 3 + h // 6)
                if grid[i][j] == 'X':
                    cv2.putText(frame, 'X', (center_x - 30, center_y + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)
                else:
                    cv2.putText(frame, 'O', (center_x - 30, center_y + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)

    return frame

def check_winner_or_draw():
    # Check rows, columns, and diagonals
    for i in range(3):
        if grid[i][0] == grid[i][1] == grid[i][2] != '':
            return grid[i][0]
        if grid[0][i] == grid[1][i] == grid[2][i] != '':
            return grid[0][i]
    if grid[0][0] == grid[1][1] == grid[2][2] != '':
        return grid[0][0]
    if grid[0][2] == grid[1][1] == grid[2][0] != '':
        return grid[0][2]

    # Check for draw
    for i in range(3):
        for j in range(3):
            if grid[i][j] == '':
                return None
    return 'Draw'

def update_grid(gesture, cell_x, cell_y):
    global turn
    if grid[cell_y][cell_x] == '' and gesture == turn:
        grid[cell_y][cell_x] = turn
        turn = 'O' if turn == 'X' else 'X'

def get_grid_position(x, y, frame):
    h, w, _ = frame.shape
    col = x // (w // 3)
    row = y // (h // 3)
    return int(row), int(col)

def detect_gesture(frame):
    results = model(frame)

    detected_gesture = None
    center_x, center_y = None, None

    detections = results[0].boxes.xyxy.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()

    confidence_threshold_o = 0.7
    confidence_threshold_x = 0.5

    for i, detection in enumerate(detections):
        x1, y1, x2, y2 = detection
        cls = int(class_ids[i])
        conf = confidences[i]

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        if cls == 1 and conf >= confidence_threshold_x:
            detected_gesture = 'X'
            break
        elif cls == 0 and conf >= confidence_threshold_o:
            detected_gesture = 'O'
            break

    return detected_gesture, center_x, center_y

# Function to display the winner or draw and the current score on the frame
def display_winner_and_score(frame, result):
    h, w, _ = frame.shape
    if result == 'Draw':
        cv2.putText(frame, "It's a Draw!", (w // 5, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
    elif result=='X':
        cv2.putText(frame, f"player_1 wins!", (w // 5, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
    else    :
        cv2.putText(frame, f"player_2 wins!", (w // 5, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)


# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = draw_grid(frame)

    gesture_detected, center_x, center_y = detect_gesture(frame)

    if gesture_detected and center_x and center_y:
        row, col = get_grid_position(center_x, center_y, frame)
        update_grid(gesture_detected, col, row)
        frame = draw_grid(frame)

    result = check_winner_or_draw()
    if result:
        if result == 'Draw':
            score['Draws'] += 1
        else:
            score[result] += 1

        display_winner_and_score(frame, result)
        cv2.imshow('Tic-Tac-Toe with Gesture Detection', frame)
        cv2.waitKey(1)
        time.sleep(2)
        grid = np.zeros((3, 3), dtype=str)

    cv2.imshow('Tic-Tac-Toe with Gesture Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
