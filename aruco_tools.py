import cv2
import sys
import time
import numpy as np

# Pre-set positions of Aruco markers in the map frame.
aruco_locations = [[0,0,0.092]]

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50, "DICT_4X4_100": cv2.aruco.DICT_4X4_100, "DICT_4X4_250": cv2.aruco.DICT_4X4_250, "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50, "DICT_5X5_100": cv2.aruco.DICT_5X5_100, "DICT_5X5_250": cv2.aruco.DICT_5X5_250, "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50, "DICT_6X6_100": cv2.aruco.DICT_6X6_100, "DICT_6X6_250": cv2.aruco.DICT_6X6_250, "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50, "DICT_7X7_100": cv2.aruco.DICT_7X7_100, "DICT_7X7_250": cv2.aruco.DICT_7X7_250, "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5, "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9, "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

aruco_type = "DICT_5X5_100"
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])
arucoParams = cv2.aruco.DetectorParameters_create()

def aruco_display(corners, ids, rejected, image):
    """
    Annotate an image with the locations of detected ArUco markers.
    
    Parameters:
    - corners: list of numpy.ndarray
        Detected marker corners in a 4x2 array of corner points (x, y).
        
    - ids: numpy.ndarray
        A 1D array containing the IDs of the detected ArUco markers.
          
    - image: numpy.ndarray
        The input image on which to annotate the ArUco markers. The image should be in BGR color format.
        
    Returns:
    - image: numpy.ndarray
        The input image annotated with the locations of the detected ArUco markers, including their IDs and centers.
    """ 
    if len(corners) > 0:
        ids = ids.flatten()
        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
            
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
            
            cv2.putTex_marker(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print("[Inference] ArUco marker ID: {}".format(markerID))
            
    return image


def info(i, ids, rvec, tvec, robotPose):
    print(f"{'ArUco Marker ID:':<20} {ids[i]}")
    # World to aruco information - Predefined
    print(f"{'Rotation:':<20}")
    print(f"{'theta_x:':<20} {rvec[0][0][0]}")
    print(f"{'theta_y:':<20} {rvec[0][0][1]}")
    print(f"{'theta_z:':<20} {rvec[0][0][2]}")
    
    print(f"{'Translation:':<20}")
    print(f"{'xWA:':<20} {round(aruco_locations[i][0]*1000)} cm")
    print(f"{'yWA:':<20} {aruco_locations[i][1]*1000} cm")
    print(f"{'zWA:':<20} {aruco_locations[i][2]*1000} cm")

    # Camera to aruco information - measured by the camera
    print(f"{'xCA:':<20} {tvec[0][0][0]*1000} cm")
    print(f"{'yCA:':<20} {tvec[0][0][1]*1000} cm")
    print(f"{'zCA:':<20} {tvec[0][0][2]*1000} cm")

    print(f"{'dx, WC:':<20} {robotPose[0]*1000} cm")
    print(f"{'dy, WC:':<20} {robotPose[1]*1000} cm")
    print(f"{'dz, WC:':<20} {robotPose[2]*1000} cm")
    print("-----------------------------------")

    # print("Euclidean distance", euc_dist)
    # time.sleep(0.5)
    # cv2.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  
      

if __name__ == '__main__':
    # intrinsic_camera = np.array(((933.15867, 0, 657.59),(0,933.1586, 400.36993),(0,0,1)))
    intrinsic_camera = np.array(((800, 0, 600),(0,800, 400),(0,0,1))) # Yet to calibrate the intrinsic for real sense 3d
    distortion = np.array((-0.43948,0.18514,0,0)) # Yet to find the distorsion coeffecients for real sense 3d

    # Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while cap.isOpened():    
        ret, img = cap.read()
        output = pose_estimation(img, ARUCO_DICT[aruco_type], intrinsic_camera, distortion)
        cv2.imshow('Estimated Pose', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()