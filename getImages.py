import cv2
import glob
import numpy as np
import yaml
def activate_webcam():
    # Access the webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Couldn't open the webcam.")
        return

    found = 0
    # Loop to continuously capture frames from the webcam
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if the frame was captured successfully
        if not ret:
            print("Error: Couldn't capture the frame.")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Try to find the chessboard corners in the grayscale frame
        retC, corners = cv2.findChessboardCorners(gray, (8, 6), None)

        # If the chessboard corners are found, save the frame when 's' is pressed
        if retC:
            # Draw the chessboard corners on the frame
            #cv2.drawChessboardCorners(frame, (7, 7), corners, retC)
            #print(gray.shape[::-1])
            # Display the frame with the chessboard corners
            cv2.imshow('Webcam', frame)

            # Save the frame if 's' is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                cv2.imwrite(f'frames/chessboard_frame{found}.jpg', frame)
                print(f"Chessboard frame saved as 'chessboard_frame{found}.jpg'")
                found+=1

        # Display the frame without the chessboard corners
        cv2.imshow('Webcam', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


def calibration(image_path, debug: bool = False, chessboard_grid_size=(8, 6)):
    """
    Perform camera calibration on the previously collected images.
    Creates `calibration_matrix.yaml` with the camera intrinsic matrix and the distortion coefficients.

    :param image_path: path to all png images
    :param every_nth: only use every n_th image
    :param debug: preview the matched chess patterns
    :param chessboard_grid_size: size of chess pattern
    :return:
    """

    x, y = chessboard_grid_size

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((y * x, 3), np.float32)
    objp[:, :2] = np.mgrid[0:x, 0:y].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob(f'{image_path}/*.jpg')
    found = 0

    src2 = cv2.imread('frames/chessboard_frame0.jpg')

    gray = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)
    for fname in images:
        img = cv2.imread(fname)  # Capture frame-by-frame
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (x, y))

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)  # Certainly, every loop objp is the same, in 3D.
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            found += 1

            if debug:
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, chessboard_grid_size, corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(100)

    print("Number of images used for calibration: ", found)

    # When everything done, release the capture
    cv2.destroyAllWindows()

    # calibration
    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print('rms', rms)

    # transform the matrix and distortion coefficients to writable lists
    data = {
        'rms': np.asarray(rms).tolist(),
        'camera_matrix': np.asarray(mtx).tolist(),
        'dist_coeff': np.asarray(dist).tolist()
    }

    # and save it to a file
    with open("calibration_matrix.yaml", "w") as f:
        yaml.dump(data, f)

    print(data)



calibration('./frames', debug=False)
#activate_webcam()




# Call the function to activate the webcam
