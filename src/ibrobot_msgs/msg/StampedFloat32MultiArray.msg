# Header stamp is the environment-owned observation time used for policy sampling.
# All observation topics produced by one reset/step share exactly this stamp.
# Receive time must NOT replace header.stamp.
std_msgs/Header header
std_msgs/Float32MultiArray value
