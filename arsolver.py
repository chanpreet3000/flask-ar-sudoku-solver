import cv2
import numpy as np
import tensorflow
import solver


model = tensorflow.keras.models.load_model('ocr_model.h5')
height, width = 630, 630
input_size = 48

solved_sudoku_board = []


def get_InvPerspective(img, masked_num, location):
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    pts2 = np.float32([location[0], location[3], location[1], location[2]])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(
        masked_num, matrix, (img.shape[1], img.shape[0]))
    return result


def display_numbers(img, numbers, color=(0, 0, 255)):
    W = int(height/9)
    H = int(width/9)
    for i in range(9):
        for j in range(9):
            if numbers[(j*9)+i] != 0:
                cv2.putText(img, str(numbers[(j*9)+i]), (i*W+int(W/2)-int((W/4)), int(
                    (j+0.7)*H)), cv2.FONT_HERSHEY_COMPLEX, 2, color, 2, cv2.LINE_AA)
    return img


def predict_sudoku_boxes(rois):
    prediction = model.predict(rois)
    predicted_numbers = []
    for i in prediction:
        index = (np.argmax(i))
        predicted_number = index
        predicted_numbers.append(predicted_number)

    return predicted_numbers


def split_boxes(board):
    rows = np.vsplit(board, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            box = cv2.resize(box, (input_size, input_size))/255.0
            boxes.append(box)
    return boxes


def get_predicted_board(img):
    gray_warp_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rois = split_boxes(gray_warp_img)
    rois = np.array(rois).reshape(-1,
                                  input_size, input_size, 1)

    # predicting wrap image
    predicted_numbers = predict_sudoku_boxes(rois)
    board_num = np.array(predicted_numbers).astype(
        'uint8').reshape(9, 9)
    print("Predicted Numbers : ", predicted_numbers)
    return board_num


def solve_warped_board(board_num):
    try:
        return solver.get_board(board_num.copy())
    except:
        return []


def check_sudoku(img):
    contours, _ = cv2.findContours(
        img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    num = 0
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 15, True)
        if len(approx) == 4:
            num += 1
    # print(num)
    if (num >= 10):
        return True
    else:
        return False


def get_perspective(img, location):
    pts1 = np.float32([location[0], location[3], location[1], location[2]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (width, height))
    return result


def solve(frame):
    global solved_sudoku_board
    img = frame.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 20)
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    for cnt in contours:
        cnt_position = cv2.approxPolyDP(cnt, 15, True)
        if len(cnt_position) == 4:
            cnt_img = img.copy()
            cv2.drawContours(cnt_img, cnt, -1, (0, 255, 0), 3)

            warp_img = get_perspective(img, cnt_position)
            warp_thesh_img = get_perspective(thresh, cnt_position)

            if (check_sudoku(warp_thesh_img)):
                if (len(solved_sudoku_board) == 0):
                    predicted_board = get_predicted_board(warp_img)
                    solved_board = solve_warped_board(predicted_board)
                    print("Predicted Board : ", predicted_board)
                    if (len(solved_board) != 0):
                        binArr = np.where(
                            np.array(predicted_board.flatten()) > 0, 0, 1)
                        temp = np.array(
                            solved_board.copy()).flatten()
                        flatten_solved_board = temp*binArr
                        solved_sudoku_board = flatten_solved_board.copy()
                        print("Solved Board : ", solved_sudoku_board)
                else:

                    black = np.zeros((height, width, 3), dtype="uint8")

                    text_img = display_numbers(
                        black, solved_sudoku_board, (0, 255, 0))

                    inv = get_InvPerspective(img, text_img, cnt_position)

                    temp = img.copy()
                    combined = cv2.addWeighted(temp, 0.5, inv, 0.5, 0)
                    return combined

    return []


def resetSudoku():
    global solved_sudoku_board
    solved_sudoku_board = []


def solveSudoku(img):
    if (len(img) != 0):
        combined = solve(img)
        if (len(combined) != 0):
            return combined
    return img
