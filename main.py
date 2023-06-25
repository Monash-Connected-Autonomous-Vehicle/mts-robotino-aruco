from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO
import time
import numpy as np
import cv2

from aruco_ekf import ArUcoEnv, ArUcoEKF, CameraAgent

UPDATE_INTERVAL = 0.1
ENV = ArUcoEnv(np.array([[1,0,1,0,np.pi/2,0], [5,0,0,0,-np.pi/2,0], [-1,0,3,0,0,0], [0,0,-2,0,np.pi,0]]))

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50, "DICT_4X4_100": cv2.aruco.DICT_4X4_100, "DICT_4X4_250": cv2.aruco.DICT_4X4_250, "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50, "DICT_5X5_100": cv2.aruco.DICT_5X5_100, "DICT_5X5_250": cv2.aruco.DICT_5X5_250, "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50, "DICT_6X6_100": cv2.aruco.DICT_6X6_100, "DICT_6X6_250": cv2.aruco.DICT_6X6_250, "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50, "DICT_7X7_100": cv2.aruco.DICT_7X7_100, "DICT_7X7_250": cv2.aruco.DICT_7X7_250, "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5, "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9, "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

CAMERA_INTRINSIC = np.array([[608.58924596, 0,           314.4157201],
                             [0,            609.1273208, 234.53093946],
                             [0,            0,           1]])
CAMERA_DISTORTION = np.array([4.88590894e-02, 5.80568720e-01, 1.22917285e-03, -1.98513739e-03, -2.41165314e+00])


if __name__ == '__main__':
    aruco_type = "DICT_5X5_100"
    ekf = ArUcoEKF(ENV, np.array([0.8994711026693413, -1.6668953491807064, 1.074997912251441]).T)
    camera_agent = CameraAgent(ENV, ARUCO_DICT[aruco_type], CAMERA_INTRINSIC, CAMERA_DISTORTION)

    t = time.time()
    while True:
        try:
            # Get responses from Rest API
            img_response = requests.get('http://192.168.0.1/cam0')
            odom_response = requests.get('http://192.168.0.1/data/odometry')
            # Convert data from response
            camera_agent.last_image = np.array(Image.open(BytesIO(img_response.content)))

            odom = odom_response.json()
            dt, t = time.time()-t, time.time()
            vel = np.array(odom[3:6]).T

            # Perform kalman filtering
            ekf.predict(vel, dt)
            ekf.update(camera_agent)

            print('state, odom', ekf.state, odom[:3])
            print('dt', dt)
            
            time.sleep(UPDATE_INTERVAL)

        except UnidentifiedImageError:
            print('Unable to retrieve camera image')
        
        except requests.exceptions.JSONDecodeError:
            print('Unable to retrieve odometry data')
        
        except KeyboardInterrupt:
            print(' Keyboard Interrupt')
            break