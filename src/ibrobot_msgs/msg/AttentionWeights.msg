# AttentionWeights.msg
# ACT attention weight diagnostics shared by inference and visualization nodes.

std_msgs/Header header

# Flattened attention weight tensor.
float32[] attention_weights
int32[] tensor_shape

# Metadata.
int32 layer_idx
int32 batch_idx
int32 query_idx
int32 num_heads
int32 feature_map_height
int32 feature_map_width

# Camera metadata.
string[] camera_keys
string[] camera_topics
int32 num_non_image_tokens

# Visualization parameters.
bool average_heads
float64 blend_alpha
