# The first mature update contains the full history. Later updates contain
# only newly appended observations; completion is emitted exactly once.
uint64 track_id
string camera_name
bool is_complete
FeatureObservation[] observations
