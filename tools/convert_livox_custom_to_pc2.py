#!/usr/bin/env python3
"""
Convert a ROS2 bag containing livox_ros_driver2/msg/CustomMsg topics
to sensor_msgs/msg/PointCloud2, preserving all timestamps.

Edit the parameters in the "USER PARAMETERS" section below, then run:
  source install/setup.bash
  python3 tools/convert_livox_custom_to_pc2.py
"""

# ════════════════════════════════════════════════════════════════════════════
# USER PARAMETERS – edit here before running
# ════════════════════════════════════════════════════════════════════════════

# Input bag directory
INPUT_BAG = "bag/rosbag_2026_0405_1300_37_379.00s"

# Output bag directory (will be created; must not already exist)
OUTPUT_BAG = "bag/rosbag_slam_pc2"

# CustomMsg topics to convert → PointCloud2 (dict: input_topic: output_topic)
# Set output_topic the same as input_topic to keep the same name.
CONVERT_MAP = {
    "/livox/lidar_192_168_1_183": "/livox/lidar_192_168_1_183",
    "/livox/lidar_192_168_1_133": "/livox/lidar_192_168_1_133",
}

# Topics to keep in the output bag (in addition to CONVERT_MAP keys).
# Set to None to keep ALL topics from the input bag.
KEEP_TOPICS = [
    "/livox/imu_192_168_1_183",
    "/livox/imu_192_168_1_133",
]

# Output storage plugin: "sqlite3" or "mcap".
# Set to None to auto-detect from the input bag.
STORAGE_OUT = None

# ════════════════════════════════════════════════════════════════════════════

import sys
import struct
import time

import rclpy
from rclpy.serialization import deserialize_message, serialize_message
import rosbag2_py

from livox_ros_driver2.msg import CustomMsg
from sensor_msgs.msg import PointCloud2, PointField


# PointCloud2 layout: x(f32) y(f32) z(f32) intensity(f32) tag(u8) line(u8) [pad 2]
_FIELDS = [
    PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
    PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
    PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
    PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    PointField(name='tag',       offset=16, datatype=PointField.UINT8,   count=1),
    PointField(name='line',      offset=17, datatype=PointField.UINT8,   count=1),
]
_POINT_STEP = 20  # 16 (4 floats) + 2 (tag+line) + 2 (padding)
_PACK_FMT = '<ffffBBxx'  # little-endian, 20 bytes per point


def custom_msg_to_pointcloud2(msg: CustomMsg) -> PointCloud2:
    buf = bytearray(_POINT_STEP * len(msg.points))
    offset = 0
    for pt in msg.points:
        struct.pack_into(_PACK_FMT, buf, offset,
                         pt.x, pt.y, pt.z, float(pt.reflectivity),
                         pt.tag, pt.line)
        offset += _POINT_STEP

    pc2 = PointCloud2()
    pc2.header = msg.header        # preserves frame_id and stamp
    pc2.height = 1
    pc2.width = len(msg.points)
    pc2.fields = _FIELDS
    pc2.is_bigendian = False
    pc2.point_step = _POINT_STEP
    pc2.row_step = _POINT_STEP * len(msg.points)
    pc2.data = bytes(buf)
    pc2.is_dense = False
    return pc2


def detect_storage(bag_path: str) -> str:
    """Guess storage plugin from file extension in the bag directory."""
    import os
    for f in os.listdir(bag_path):
        if f.endswith('.mcap'):
            return 'mcap'
    return 'sqlite3'


def convert_bag(input_path: str, output_path: str,
                convert_map: dict,      # {input_topic: output_topic}
                keep_set: set | None,   # None = keep all; set = whitelist
                storage_out: str) -> None:
    # ── reader ───────────────────────────────────────────────────────────────
    storage_in = detect_storage(input_path)
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=input_path, storage_id=storage_in),
        rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'),
    )

    all_topics = reader.get_all_topics_and_types()
    topic_type_map = {t.name: t.type for t in all_topics}

    for src in convert_map:
        if src not in topic_type_map:
            print(f"[ERROR] Topic '{src}' not found in bag.")
            print("Available topics:", list(topic_type_map.keys()))
            sys.exit(1)
        if topic_type_map[src] != 'livox_ros_driver2/msg/CustomMsg':
            print(f"[WARN] '{src}' has type '{topic_type_map[src]}', "
                  "expected 'livox_ros_driver2/msg/CustomMsg'. Proceeding anyway.")

    # topics that will actually be written
    def should_write(name: str) -> bool:
        if name in convert_map:          # always write (as converted)
            return True
        if keep_set is None:             # no whitelist → keep all
            return True
        return name in keep_set

    # ── writer ───────────────────────────────────────────────────────────────
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=output_path, storage_id=storage_out),
        rosbag2_py.ConverterOptions('', ''),
    )

    for t in all_topics:
        if not should_write(t.name):
            continue
        if t.name in convert_map:
            writer.create_topic(rosbag2_py.TopicMetadata(
                name=convert_map[t.name],
                type='sensor_msgs/msg/PointCloud2',
                serialization_format='cdr',
            ))
        else:
            writer.create_topic(t)

    # ── process messages ──────────────────────────────────────────────────────
    total = reader.get_metadata().message_count
    print(f"Processing {total} messages ...")
    counts: dict[str, int] = {}
    skipped = 0
    processed = 0
    t0 = time.monotonic()

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        processed += 1

        if not should_write(topic):
            skipped += 1
        elif topic in convert_map:
            msg = deserialize_message(data, CustomMsg)
            pc2_msg = custom_msg_to_pointcloud2(msg)
            out_topic = convert_map[topic]
            writer.write(out_topic, serialize_message(pc2_msg), timestamp)
            counts[topic] = counts.get(topic, 0) + 1
        else:
            writer.write(topic, data, timestamp)
            counts[topic] = counts.get(topic, 0) + 1

        if processed % 500 == 0 or processed == total:
            elapsed = time.monotonic() - t0
            pct = processed / total * 100 if total else 0
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0
            bar_filled = int(pct / 5)
            bar = '█' * bar_filled + '░' * (20 - bar_filled)
            print(f"\r  [{bar}] {pct:5.1f}%  {processed}/{total}  "
                  f"{rate:6.0f} msg/s  ETA {eta:5.1f}s   ",
                  end='', flush=True)

    print()  # newline after progress bar

    del reader
    del writer

    print(f"  Done in {time.monotonic() - t0:.1f}s")
    print("\nSummary:")
    for k, v in sorted(counts.items()):
        tag = f" → {convert_map[k]} (PointCloud2)" if k in convert_map else ""
        print(f"  {k}{tag}: {v} msgs")
    if skipped:
        print(f"  (dropped {skipped} messages from other topics)")
    print(f"\nOutput bag: {output_path}")


def main() -> None:
    keep_set = set(KEEP_TOPICS) if KEEP_TOPICS is not None else None
    storage_out = STORAGE_OUT or detect_storage(INPUT_BAG)

    rclpy.init(args=[])
    try:
        convert_bag(INPUT_BAG, OUTPUT_BAG,
                    CONVERT_MAP, keep_set, storage_out)
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
