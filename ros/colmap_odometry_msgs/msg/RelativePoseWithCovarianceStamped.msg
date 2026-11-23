# Relative motion measurement for a graph between-factor.
#
# The measurement is the transform from the base frame at from_stamp to the
# base frame at to_stamp.  header.stamp is equal to to_stamp and
# header.frame_id is the configured common world-independent base frame.
#
# The covariance is serialized using geometry_msgs' ROS order
# [translation_x, translation_y, translation_z, rotation_x, rotation_y,
#  rotation_z].  Consumers that use GTSAM must reorder this to
# [rotation_x, rotation_y, rotation_z, translation_x, translation_y,
#  translation_z].  The covariance describes the local SE(3) error of the
# relative measurement, in metres/radians (and their cross units), using the
# same right-local convention as a GTSAM Pose3 BetweenFactor.
std_msgs/Header header
builtin_interfaces/Time from_stamp
builtin_interfaces/Time to_stamp
geometry_msgs/PoseWithCovariance relative_pose
uint32 inlier_count
float64 reprojection_rms_px
float64 information_condition_number
string covariance_source
