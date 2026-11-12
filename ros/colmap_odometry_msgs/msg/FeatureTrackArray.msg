# Track updates emitted after processing one camera keyframe.
std_msgs/Header header
string feature_type
# The exact synchronized input pair used to produce this update. Keeping the
# compressed inputs with the tracks avoids trying to re-synchronize two large,
# independently delivered decompressed image topics after COLMAP processing.
bool has_stereo_images
sensor_msgs/CompressedImage left_image
sensor_msgs/CompressedImage right_image
FeatureTrack[] tracks
