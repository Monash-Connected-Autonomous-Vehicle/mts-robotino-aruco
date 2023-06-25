import numpy as np
import cv2


class ArUcoEnv:
    '''
    Specification for ArUco marker layout in environment in world frame coordinates

    Parameters
    ----------
    markers : np.ndarray
        6xn array containing (x, y, z, rx, ry, rz) coordinates of ArUco markers

    Attributes
    ----------
    mats : list[np.ndarray]
        List of 4x4 homogeneous transformation matrices of each marker in world frame, in ID order.

    '''
    def __init__(self, markers) -> None:
        self.markers = markers  # x, y, z, rx, ry, rx for all markers in ArUco dict
        self.mats = []
        for marker in markers:
            R_mc, _ = cv2.Rodrigues(marker[3:])  # Rotation of camera in marker frame
            T_mc = np.eye(4)              # Transformation of camera in marker frame
            T_mc[:3, :3] = R_mc
            T_mc[:3, 3] = marker[:3]
            self.mats.append(T_mc)


class CameraAgent:
    '''
    Processing of camera feed from agent in ArUco environment

    Parameters
    ----------
    env : ArUcoEnv
        Specification of marker positions in environment.

    aruco_dict_type : cv2.aruco_Dictionary
        ArUco marker dictionary to be used for marker detection.

    intrinsic_mat : numpy.ndarray
        3x3 intrinsic camera matrix with the focal lengths and optical centers information.     

    distortion_coeffs: numpy.ndarray
        Lens distortion coefficients (k1, k2, p1, p2, k3, ..., kn) of camera.

    '''
    def __init__(self, env, aruco_dict_type, intrinsic_mat, distortion_coeffs) -> None:
        self.env = env
        self.intrinsic_mat = intrinsic_mat
        self.distorion_coeffs = distortion_coeffs
        self.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
        self.detector_parameters = cv2.aruco.DetectorParameters_create()
        self.last_image = None  # Last image seen by the camera, should be updated consistently


    def obs_to_pose(self, tvec, rvec, id):
        '''
        Calculation of robot pose with respect to the world frame, based on a single ArUco detection

        Parameters
        ----------
        tvec : np.ndarray
            (x, y, z) ranslation of ArUco marker in camera frame
        
        rvec : np.ndarray
            (rx, ry, rz) rotation of ArUco marker in camera frame

        id : int
            ID of ArUco marker

        Returns
        -------
        pose : np.ndarray
            Array containing (x, z, theta) planar coordinates of robot pose in world frame

        '''

        R_mc = cv2.Rodrigues(rvec)[0].T  # Rotation of camera in marker frame
        T_mc = np.eye(4)                 # Transformation of camera in marker frame
        T_mc[:3, :3] = R_mc              # Calculate inverse transformation
        T_mc[:3, 3] = (-R_mc @ tvec).T

        T_wm = self.env.mats[id]         # Get marker position on world frame from environment specification
        T_wc = T_wm @ T_mc               # Transformation of camera in world frame

        camera_rot, camera_pos = cv2.Rodrigues(T_wc[:3, :3], jacobian=False), T_wc[:3, 3]  # Convert back to angle specification
        return np.array([camera_pos[0], camera_pos[2], camera_rot[0][1][0]])               # Extract planar coordinates
    

    def aggregate_poses(self, poses):
        '''
        Calculates averages and standard deviations of robot pose estimates based on multiple ArUco markers

        Parameters
        ----------
        poses : np.ndarray
            3xn array containing planar coorindates of each pose estimation

        Returns
        -------
        est : np.ndarray
            1x3 vector containing planar coordinates of aggregate estimate
        x_std : float
            Standard deviation of x coordinates
        z_std : float
            Standard deviation of z coordinates

        '''
        x_mean, z_mean = np.mean(poses[0]), np.mean(poses[1])
        x_std, z_std = np.std(poses[0]), np.std(poses[1])
        th_mean = np.arctan2(np.mean(np.sin(poses[2])), np.mean(np.cos(poses[2])))  # Mean heading

        return np.array([x_mean, z_mean, th_mean]).T, x_std, z_std


    def pose_estimation(self):
        '''
        Estimation of robot pose based on last image seen by agent.

        Returns
        -------
        mean_est : np.ndarray
            1x3 vector containing planar coordinates of camera pose estimate in world frame, based on last image seen by agent.
            Returns None if no ArUco markers are detected in the image

        '''

        corners, ids, _ = cv2.aruco.detectMarkers(self.last_image, self.aruco_dict, parameters=self.detector_parameters)

        if ids is not None:
            ests = np.empty((3, len(ids)))
            for i, id in enumerate(ids):
                # Get pose estimate for each marker
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, self.intrinsic_mat, self.distorion_coeffs)
                ests[:, i] = self.obs_to_pose(tvec[0].T, rvec, id[0])

            # Collate average measurement
            mean_est, _, _ = self.aggregate_poses(ests)
            return mean_est

        return None


class ArUcoEKF:
    '''
    Extended Kalman Filter for tracking robotino pose based on ArUco observations and velocity odometry

    Parameters
    ----------
    env : ArUcoEnv
        Specification of marker positions in environment.

    init_state : np.ndarray, default=[0,0,0]
        Initial (x, z, theta) state of the robot

    init_cov : np.ndarray, default=[[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]]
        Initial covariance/undertainty of the robot's state
    
    process_noise : np.ndarray, default=[[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]]
        Process noise of the robot's movement based on velocity information

    measure_noise : np.ndarray, default=[[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]]
        Measurement noise of robot's position from ArUco markers

    '''
    def __init__(self, env, init_state=np.array([0,0,0]).T, init_cov=1e-3*np.eye(3),
                 process_noise=1e-3*np.eye(3), measure_noise=1e-3*np.eye(3)) -> None:
        self.env = env
        self.state = init_state
        self.markers = np.zeros((2,0))

        # Position Covariance matrix
        self.P = init_cov  # x, z, ry
        self.process_noise = process_noise
        self.measure_noise = measure_noise

    def predict(self, vel, dt):
        '''
        Predicts robot state/covariance evolution from velocity readings

        Arguments
        ---------
        vel : np.ndarray
            Velocity vector containing (vx, vz, vth)

        dt : float
            Time delta between measurements

        '''

        # Prediction of the next sate from the velocity
        self.state = self.state + vel*dt

        # Assumes no covariance between linear/angular velocities
        #J = dt*np.eye(3)  # Jacobian of dynamics

        F = np.eye(3) + dt*np.diag(vel)
        self.P = F @ self.P @ F + self.process_noise  # Update uncertainty in robot pose
    

    def update(self, camera_agent):
        '''
        Updates the robot state based on observation of ArUco markers

        Parameters
        ----------
        camera_agent : CameraAgent
            Specification of camera view of robot, performs calculation of pose estimate
        '''
        aruco_est = camera_agent.pose_estimation()
        print('aruco est', aruco_est)
        if aruco_est is None: return
        
        H = np.eye(3)           # Maps state space to observation space (identity)

        # Compute kalman gain
        S = H @ self.P @ H.T + self.measure_noise
        K = self.P @ H.T @ np.linalg.inv(S)

        # Make state correction
        y = aruco_est - H @ self.state         # Prediction and observation difference
        self.state = self.state + K @ y        # Updated state prediction
        self.P = (np.eye(3) - K @ H) @ self.P  # Updated state covariance



if __name__ == '__main__':
    '''
    Testing code, main script should be run from main.py
    '''
    ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50, "DICT_4X4_100": cv2.aruco.DICT_4X4_100, "DICT_4X4_250": cv2.aruco.DICT_4X4_250, "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50, "DICT_5X5_100": cv2.aruco.DICT_5X5_100, "DICT_5X5_250": cv2.aruco.DICT_5X5_250, "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50, "DICT_6X6_100": cv2.aruco.DICT_6X6_100, "DICT_6X6_250": cv2.aruco.DICT_6X6_250, "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50, "DICT_7X7_100": cv2.aruco.DICT_7X7_100, "DICT_7X7_250": cv2.aruco.DICT_7X7_250, "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5, "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9, "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
    }
    aruco_type = "DICT_5X5_100"


    markers = np.array([[1,0,1,0,np.pi/2,0], [5,0,0,0,-np.pi/2,0], [-1,0,3,0,0,0], [0,0,-2,0,np.pi,0]])
    env = ArUcoEnv(markers)
    ekf = ArUcoEKF(env, np.array([0,0,0]).T)
    intrin =  np.array([[608.58924596, 0, 314.4157201],
                         [0, 609.1273208, 234.53093946],
                         [0, 0, 1]])
    distort  = [[4.88590894e-02, 5.80568720e-01, 1.22917285e-03, -1.98513739e-03, -2.41165314e+00]]
    agent = CameraAgent(env, ARUCO_DICT[aruco_type], intrin, distort)


    print(env.mats[0])
    '''print(ekf.state)
    print(ekf.P)
    ekf.predict(np.array([10,0,0]), 0.1)
    print(ekf.state)
    print(ekf.P)
    ekf.update()
    print(ekf.state)
    print(ekf.P)'''


