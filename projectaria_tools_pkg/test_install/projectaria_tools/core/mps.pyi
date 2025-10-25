from __future__ import annotations
import projectaria_tools.core.calibration
import datetime
import numpy
import typing
__all__ = ['ClosedLoopTrajectoryPose', 'EyeGaze', 'GlobalPointPosition', 'MpsDataPaths', 'MpsDataPathsProvider', 'MpsEyegazeDataPaths', 'MpsSlamDataPaths', 'OnlineCalibration', 'OpenLoopTrajectoryPose', 'PointObservation', 'StaticCameraCalibration', 'StreamCompressionMode', 'get_eyegaze_point_at_depth', 'read_closed_loop_trajectory', 'read_eyegaze', 'read_global_point_cloud', 'read_online_calibration', 'read_open_loop_trajectory', 'read_point_observations', 'read_static_camera_calibrations']
class ClosedLoopTrajectoryPose:
    """
    
              Closed loop trajectory is the pose estimation output by our mapping process, in an arbitrary
      gravity aligned world coordinate frame. The estimation includes pose and dynamics (translational
      and angular velocities).
    
      Closed loop trajectories are fully bundle adjusted with detected loop closures, reducing the VIO
      drift which is present in the open loop trajectories. However, due to the loop closure
      correction, the “relative” and “local” trajectory accuracy within a short time span (i.e.
      seconds) might be worse compared to open loop trajectories.
    
      In some datasets we also share and use this format for trajectory pose ground truth from
      simulation or Optitrack
        
    """
    angular_velocity_device: numpy.ndarray[numpy.float64[3, 1]]
    device_linear_velocity_device: numpy.ndarray[numpy.float64[3, 1]]
    graph_uid: str
    gravity_world: numpy.ndarray[numpy.float64[3, 1]]
    quality_score: float
    tracking_timestamp: datetime.timedelta
    transform_world_device: SE3
    utc_timestamp: datetime.timedelta
    def __repr__(self) -> str:
        ...
class EyeGaze:
    """
    An object representing single Eye gaze output.
    """
    depth: float
    pitch: float
    pitch_high: float
    pitch_low: float
    session_uid: str
    tracking_timestamp: datetime.timedelta
    yaw: float
    yaw_high: float
    yaw_low: float
    def __repr__(self) -> str:
        ...
class GlobalPointPosition:
    distance_std: float
    graph_uid: str
    inverse_distance_std: float
    position_world: numpy.ndarray[numpy.float64[3, 1]]
    uid: int
    def __repr__(self) -> str:
        ...
class MpsDataPaths:
    """
    A struct that includes the file paths of all MPS data for a sequence.
    """
    eyegaze: ...
    slam: ...
    def __repr__(self) -> str:
        ...
class MpsDataPathsProvider:
    """
    This class is allows you to get all MPS data paths associated with an Aria sequence. 
    Note that all Aria open datasets will have MPS results which fit the format specified in this data provider.
    Use this data provider to avoid breaking changes in your code due to changes in MPS files
    """
    def __init__(self, arg0: str) -> None:
        ...
    def get_data_paths(self) -> MpsDataPaths:
        """
        Get the resulting data paths
        """
class MpsEyegazeDataPaths:
    """
    A struct that includes the file paths of all MPS eye gaze data for a sequence.
    """
    general_eyegaze: str
    personalized_eyegaze: str
    summary: str
    def __repr__(self) -> str:
        ...
class MpsSlamDataPaths:
    """
    A struct that includes the file paths of all MPS SLAM data for a VRS sequence processed by MPS.
    """
    closed_loop_trajectory: str
    online_calibrations: str
    open_loop_trajectory: str
    semidense_observations: str
    semidense_points: str
    summary: str
    def __repr__(self) -> str:
        ...
class OnlineCalibration:
    camera_calibs: list[projectaria_tools.core.calibration.CameraCalibration]
    imu_calibs: list[projectaria_tools.core.calibration.ImuCalibration]
    tracking_timestamp: datetime.timedelta
    utc_timestamp: datetime.timedelta
    def __repr__(self) -> str:
        ...
class OpenLoopTrajectoryPose:
    """
    
            Open loop trajectory is the odometry estimation output by the visual-inertial odometry (VIO), in
            an arbitrary odometry coordinate frame. The estimation includes pose and dynamics (translational
            and angular velocities).
    
            The open loop trajectory has good “relative” and “local” accuracy: the relative transformation
            between two frames is accurate when the time span between two frames is short (within a few
            minutes). However, the open loop trajectory has increased drift error accumulated over time spent
            and travel distance. Consider using closed loop trajectory if you are looking for trajectory
            without drift error.
        
    """
    angular_velocity_device: numpy.ndarray[numpy.float64[3, 1]]
    device_linear_velocity_odometry: numpy.ndarray[numpy.float64[3, 1]]
    gravity_odometry: numpy.ndarray[numpy.float64[3, 1]]
    quality_score: float
    session_uid: str
    tracking_timestamp: datetime.timedelta
    transform_odometry_device: SE3
    utc_timestamp: datetime.timedelta
    def __repr__(self) -> str:
        ...
class PointObservation:
    """
    2D observations of the point
    """
    camera_serial: str
    frame_capture_timestamp: datetime.timedelta
    point_uid: int
    uv: numpy.ndarray[numpy.float32[2, 1]]
    def __repr__(self) -> str:
        ...
class StaticCameraCalibration:
    """
    Static camera intrinsic calibration and extrinsics in the world frame
    """
    camera_uid: str
    end_frame_idx: int | None
    graph_uid: str
    height: int
    intrinsics: numpy.ndarray[numpy.float32[8, 1]]
    intrinsics_type: str
    start_frame_idx: int | None
    transform_world_cam: SE3
    width: int
    def __repr__(self) -> str:
        ...
class StreamCompressionMode:
    """
    Stream compression mode
    
    Members:
    
      NONE : No compression
    
      GZIP : GZIP compression
    """
    GZIP: typing.ClassVar[StreamCompressionMode]  # value = <StreamCompressionMode.GZIP: 1>
    NONE: typing.ClassVar[StreamCompressionMode]  # value = <StreamCompressionMode.NONE: 0>
    __members__: typing.ClassVar[typing.Dict[str, StreamCompressionMode]]  # value = {'NONE': <StreamCompressionMode.NONE: 0>, 'GZIP': <StreamCompressionMode.GZIP: 1>}
    def __eq__(self, other: object) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: object) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(arg0: StreamCompressionMode) -> int:
        ...
def get_eyegaze_point_at_depth(yaw_rads: float, pitch_rads: float, depth_m: float) -> numpy.ndarray[numpy.float64[3, 1]]:
    """
     Given the yaw and pitch angles of the eye gaze and a depth, return the gaze 3D point in CPF frame.
      Parameters
      __________
      yaw_rads: Yaw angle in radians in CPF frame.
      pitch_rads: Pitch angle in radians in CPF frame.
      depth_m: Depth of the point in meters.
    """
def read_closed_loop_trajectory(path: str) -> list[ClosedLoopTrajectoryPose]:
    """
    Read Closed loop trajectory.
      Parameters
      __________
      path: Path to the closed loop trajectory csv file. Usually named 'closed_loop_trajectory.csv'
    """
def read_eyegaze(path: str) -> list[EyeGaze]:
    """
    Read Eye Gaze from the eye gaze output generated via MPS.
      Parameters
      __________
      path: Path to the eye gaze csv file.
    """
@typing.overload
def read_global_point_cloud(path: str, compression: StreamCompressionMode) -> list[GlobalPointPosition]:
    """
    Read global point cloud.
      Parameters
      __________
      path: Path to the global point cloud file. Usually named 'global_pointcloud.csv.gz'
      compression: Stream compression mode for reading csv file.
    """
@typing.overload
def read_global_point_cloud(path: str) -> list[GlobalPointPosition]:
    """
    Read global point cloud.
      Parameters
      __________
      path: Path to the global point cloud file. Usually named 'global_pointcloud' with a '.csv' or '.csv.gz'
    """
def read_online_calibration(path: str) -> list[OnlineCalibration]:
    """
    Read estimated online calibrations.
      Parameters
      __________
      path: Path to the online calibration jsonl file. Usually named 'online_calibration.jsonl'
    """
def read_open_loop_trajectory(path: str) -> list[OpenLoopTrajectoryPose]:
    """
    Read Open loop trajectory.
      Parameters
      __________
      path: Path to the open loop trajectory csv file. Usually named 'open_loop_trajectory.csv'
    """
@typing.overload
def read_point_observations(path: str, compression: StreamCompressionMode) -> list[PointObservation]:
    """
    Read point observations.
      Parameters
      __________
      path: Path to the point observations file. Usually named 'semidense_observations.csv.gz'
      compression: Stream compression mode for reading csv file.
    """
@typing.overload
def read_point_observations(path: str) -> list[PointObservation]:
    """
    Read point observations.
      Parameters
      __________
      path: Path to the point observations file. Usually named 'semidense_observations' with a '.csv' or '.csv.gz'
    """
def read_static_camera_calibrations(path: str) -> list[StaticCameraCalibration]:
    """
    Read static camera calibrations.
      Parameters
      __________
      path: Path to the static camera calibrations file.
    """
