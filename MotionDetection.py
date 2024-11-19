import cv2
import numpy as np
from sklearn.cluster import DBSCAN


# Step 1: Detect Motion Edges Between Frames
def compute_motion_edges(prev_frame, current_frame):
    """Generate edges highlighting motion using the frame difference."""
    difference = cv2.absdiff(current_frame, prev_frame)
    edge_map = cv2.Canny(difference, 50, 150)  # Apply Canny edge detection
    return edge_map


# Step 2: Identify Moving Regions via Clustering
def find_clusters(edge_map, max_distance=10, min_cluster_size=5):
    """Use DBSCAN to group edge points into clusters."""
    # Extract edge coordinates
    edge_points = np.column_stack(np.nonzero(edge_map))  # y, x format
    if edge_points.size == 0:
        return []

    # Perform clustering
    clustering = DBSCAN(eps=max_distance, min_samples=min_cluster_size).fit(edge_points)
    regions = []
    for cluster_id in np.unique(clustering.labels_):
        if cluster_id == -1:  # Ignore noise
            continue
        cluster_points = edge_points[clustering.labels_ == cluster_id]
        y_min, x_min = cluster_points.min(axis=0)
        y_max, x_max = cluster_points.max(axis=0)
        regions.append((x_min, y_min, x_max, y_max))  # Bounding box in OpenCV format
    return regions


# Step 3: Estimate Object Speeds
def measure_speeds(previous_positions, current_positions, frame_rate, scale_factor=1):
    """Calculate speeds of tracked objects."""
    object_speeds = {}
    for obj_id, (prev_x, prev_y) in previous_positions.items():
        if obj_id in current_positions:
            curr_x, curr_y = current_positions[obj_id]
            movement = ((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2) ** 0.5
            velocity = movement * scale_factor * frame_rate
            object_speeds[obj_id] = velocity
    return object_speeds


# Step 4: Process Video to Detect Motion and Track Objects
def analyze_video(video_file, fps, scale=1, max_distance=10, min_cluster_size=5):
    video = cv2.VideoCapture(video_file)
    success, first_frame = video.read()
    if not success:
        print("Error: Unable to open video.")
        return

    gray_prev_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    previous_positions = {}
    obj_id_counter = 0

    while True:
        success, current_frame = video.read()
        if not success:
            break

        gray_curr_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Detect motion and find clusters
        edges = compute_motion_edges(gray_prev_frame, gray_curr_frame)
        motion_regions = find_clusters(edges, max_distance=max_distance, min_cluster_size=min_cluster_size)

        # Match and track objects
        current_positions = {}
        for bbox in motion_regions:
            x_min, y_min, x_max, y_max = bbox
            center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2

            # Assign existing or new ID
            closest_id = None
            min_dist = float('inf')
            for obj_id, (prev_x, prev_y) in previous_positions.items():
                dist = ((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    closest_id = obj_id

            if closest_id is not None and min_dist < 50:
                current_positions[closest_id] = (center_x, center_y)
            else:
                current_positions[obj_id_counter] = (center_x, center_y)
                obj_id_counter += 1

        # Calculate speeds
        speeds = measure_speeds(previous_positions, current_positions, fps, scale)

        # Display results
        for obj_id, bbox in zip(current_positions.keys(), motion_regions):
            x_min, y_min, x_max, y_max = bbox
            speed = speeds.get(obj_id, 0)
            cv2.rectangle(current_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(current_frame, f"Speed: {speed:.2f} km/h", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        object_count = len(current_positions)
        cv2.putText(current_frame, f"Objects: {object_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

        cv2.imshow('Motion Detection & Tracking', current_frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        previous_positions = current_positions
        gray_prev_frame = gray_curr_frame

    video.release()
    cv2.destroyAllWindows()

# Run the analysis
analyze_video("Horse_running_around.mp4", fps=30, scale=0.05)
