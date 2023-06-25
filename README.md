# mts-robotino-aruco
ArUco marker based localization for Robotino.

Performs pose estimation via [Kalman filter](https://en.wikipedia.org/wiki/Extended_Kalman_filter) using decoupled velocity odometry, and observation of predefined ArUco marker locations.

Operates on remote machine connected on a Robotino network, via RestAPI. Running main.py after connecting performs pose estimation.
