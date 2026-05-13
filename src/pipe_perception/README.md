# pipe_perception

ROS2 pipe inspection perception pipeline.
Reads a video file, unwraps each frame cylindrically, accumulates coverage, and publishes a 3D heatmap to RViz2.

---

## Pipeline

```
pipe.mp4
   │
   ▼
video_ingest_node          /camera/image_raw  (sensor_msgs/Image)
   │
   ▼
cylinder_unwrap_node        /pipe/unwrapped    (sensor_msgs/Image)
   │
   ▼
coverage_mapper_node        /pipe/coverage              (sensor_msgs/Image)
                            /pipe/coverage_percentage   (std_msgs/Float32)
   │
   ▼
cylinder_visualizer_3d_node /visualization/cylinder     (visualization_msgs/Marker)
```

---

## Prerequisites

- ROS2 Humble (or later)
- Python 3.10+
- OpenCV + cv_bridge: `sudo apt install ros-humble-cv-bridge python3-opencv`
- NumPy: `pip install numpy`

---

## Build

```bash
# From the workspace root (~/inspectly)
cd ~/inspectly
colcon build --packages-select pipe_perception
source install/setup.bash
```

---

## Add your video

Place the inspection video at:

```
~/inspectly/videos/pipe.mp4
```

Any other path can be passed as a launch argument (see below).

---

## Run

### Full pipeline (all 4 nodes)

```bash
ros2 launch pipe_perception pipe_perception.launch.py
```

### Override defaults

```bash
ros2 launch pipe_perception pipe_perception.launch.py \
    video_path:=/abs/path/to/pipe.mp4 \
    fps:=25.0 \
    pipe_radius:=0.10 \
    pipe_length:=3.5
```

### Run nodes individually

```bash
ros2 run pipe_perception video_ingest_node
ros2 run pipe_perception cylinder_unwrap_node
ros2 run pipe_perception coverage_mapper_node
ros2 run pipe_perception cylinder_visualizer_3d_node
```

---

## Parameters

| Node | Parameter | Default | Description |
|------|-----------|---------|-------------|
| `video_ingest_node` | `video_path` | `~/inspectly/videos/pipe.mp4` | Path to video file |
| `video_ingest_node` | `fps` | `30.0` | Publish rate (frames/sec) |
| `cylinder_unwrap_node` | `radius` | `0.15` | Pipe inner radius (m) |
| `coverage_mapper_node` | `angle_bins` | `360` | Angular grid resolution |
| `coverage_mapper_node` | `distance_bins` | `480` | Depth grid resolution |
| `cylinder_visualizer_3d_node` | `pipe_radius` | `0.15` | Pipe inner radius (m) |
| `cylinder_visualizer_3d_node` | `pipe_length` | `2.0` | Pipe length (m) |

---

## Topics

| Topic | Type | Direction | Description |
|-------|------|-----------|-------------|
| `/camera/image_raw` | `sensor_msgs/Image` | published | Raw BGR frames |
| `/pipe/unwrapped` | `sensor_msgs/Image` | published | Cylindrical projection (H×360) |
| `/pipe/coverage` | `sensor_msgs/Image` | published | Greyscale heatmap (mono8) |
| `/pipe/coverage_percentage` | `std_msgs/Float32` | published | Coverage 0–100 % |
| `/visualization/cylinder` | `visualization_msgs/Marker` | published | RViz2 TRIANGLE_LIST |

---

## Visualise in RViz2

1. Open RViz2: `rviz2`
2. Set **Fixed Frame** to `world`
3. Add a **Marker** display → topic `/visualization/cylinder`
4. Add an **Image** display → topic `/pipe/coverage` (optional heatmap overlay)
