# One native feature observation in a COLMAP camera image.
builtin_interfaces/Time stamp
uint32 frame_index
uint32 keypoint_index
float32 x
float32 y

# Stereo geometry for this cam0 feature. The 3D point is expressed in the
# cam0 optical frame at this observation timestamp.
bool has_stereo_point
float32 stereo_x
float32 stereo_y
float64[3] stereo_point_camera
float32 triangulation_angle_rad
