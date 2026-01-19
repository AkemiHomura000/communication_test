#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


@dataclass
class TopicAccumulator:
    topic_name: str
    target_frames: int
    out_dir: str
    frame_count: int = 0
    points_xyz: List[Tuple[float, float, float]] = field(default_factory=list)
    points_xyzi: Optional[List[Tuple[float, float, float, float]]] = None  # if intensity exists

    def add_points(self, xyz_list: List[Tuple[float, float, float]],
                   xyzi_list: Optional[List[Tuple[float, float, float, float]]]):
        self.frame_count += 1
        if xyzi_list is not None:
            if self.points_xyzi is None:
                self.points_xyzi = []
            self.points_xyzi.extend(xyzi_list)
        else:
            self.points_xyz.extend(xyz_list)

    def is_done(self) -> bool:
        return self.frame_count >= self.target_frames

    def output_path(self) -> str:
        safe = self.topic_name.strip("/").replace("/", "_")
        return os.path.join(self.out_dir, f"{safe}_integrated_{self.target_frames}frames.pcd")


class IntegratePointCloudToPCD(Node):
    def __init__(self):
        super().__init__("integrate_pointcloud_to_pcd")

        # Parameters
        self.declare_parameter("topics", [
            "/livox/lidar_192_168_1_133",
            "/livox/lidar_192_168_1_183",
        ])
        self.declare_parameter("frames", 50)
        self.declare_parameter("out_dir", "./pcd_output")
        self.declare_parameter("skip_nan", True)
        self.declare_parameter("binary_pcd", True)

        topics = self.get_parameter("topics").get_parameter_value().string_array_value
        target_frames = int(self.get_parameter("frames").value)
        out_dir = str(self.get_parameter("out_dir").value)
        self.skip_nan = bool(self.get_parameter("skip_nan").value)
        self.binary_pcd = bool(self.get_parameter("binary_pcd").value)

        os.makedirs(out_dir, exist_ok=True)

        # QoS (match bag as close as possible)
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.accs: Dict[str, TopicAccumulator] = {}
        self.subs = []

        for t in topics:
            self.accs[t] = TopicAccumulator(topic_name=t, target_frames=target_frames, out_dir=out_dir)
            sub = self.create_subscription(PointCloud2, t, lambda msg, tn=t: self.cb(msg, tn), qos)
            self.subs.append(sub)
            self.get_logger().info(f"Subscribed: {t}")

        self.get_logger().info(
            f"Integrating {target_frames} frames per topic. Output dir: {os.path.abspath(out_dir)}"
        )

    def cb(self, msg: PointCloud2, topic_name: str):
        acc = self.accs[topic_name]
        if acc.is_done():
            return

        field_names = [f.name for f in msg.fields]
        has_intensity = ("intensity" in field_names)

        if has_intensity:
            pts = list(point_cloud2.read_points(
                msg,
                field_names=("x", "y", "z", "intensity"),
                skip_nans=self.skip_nan
            ))
            xyzi = [(float(x), float(y), float(z), float(i)) for (x, y, z, i) in pts]
            acc.add_points(xyz_list=[], xyzi_list=xyzi)
            total_pts = len(acc.points_xyzi) if acc.points_xyzi is not None else 0
        else:
            pts = list(point_cloud2.read_points(
                msg,
                field_names=("x", "y", "z"),
                skip_nans=self.skip_nan
            ))
            xyz = [(float(x), float(y), float(z)) for (x, y, z) in pts]
            acc.add_points(xyz_list=xyz, xyzi_list=None)
            total_pts = len(acc.points_xyz)

        self.get_logger().info(
            f"[{topic_name}] frame {acc.frame_count}/{acc.target_frames}, accumulated points: {total_pts}"
        )

        if acc.is_done():
            out_path = acc.output_path()
            if has_intensity:
                self.write_pcd_xyzi(out_path, acc.points_xyzi or [])
            else:
                self.write_pcd_xyz(out_path, acc.points_xyz)

            self.get_logger().info(f"[{topic_name}] DONE. Saved: {out_path}")

            if all(a.is_done() for a in self.accs.values()):
                self.get_logger().info("All topics finished. Shutting down.")
                rclpy.shutdown()

    def write_pcd_xyz(self, path: str, points: List[Tuple[float, float, float]]):
        if self.binary_pcd:
            self._write_pcd_binary_xyz(path, points)
        else:
            self._write_pcd_ascii_xyz(path, points)

    def write_pcd_xyzi(self, path: str, points: List[Tuple[float, float, float, float]]):
        if self.binary_pcd:
            self._write_pcd_binary_xyzi(path, points)
        else:
            self._write_pcd_ascii_xyzi(path, points)

    @staticmethod
    def _pcd_header(fields: str, size: str, ptype: str, count: str, width: int, points: int, data: str) -> str:
        return (
            "# .PCD v0.7 - Point Cloud Data file format\n"
            "VERSION 0.7\n"
            f"FIELDS {fields}\n"
            f"SIZE {size}\n"
            f"TYPE {ptype}\n"
            f"COUNT {count}\n"
            f"WIDTH {width}\n"
            "HEIGHT 1\n"
            "VIEWPOINT 0 0 0 1 0 0 0\n"
            f"POINTS {points}\n"
            f"DATA {data}\n"
        )

    def _write_pcd_ascii_xyz(self, path: str, points: List[Tuple[float, float, float]]):
        header = self._pcd_header("x y z", "4 4 4", "F F F", "1 1 1", len(points), len(points), "ascii")
        with open(path, "w", encoding="utf-8") as f:
            f.write(header)
            for x, y, z in points:
                f.write(f"{x} {y} {z}\n")

    def _write_pcd_ascii_xyzi(self, path: str, points: List[Tuple[float, float, float, float]]):
        header = self._pcd_header("x y z intensity", "4 4 4 4", "F F F F", "1 1 1 1", len(points), len(points), "ascii")
        with open(path, "w", encoding="utf-8") as f:
            f.write(header)
            for x, y, z, i in points:
                f.write(f"{x} {y} {z} {i}\n")

    def _write_pcd_binary_xyz(self, path: str, points: List[Tuple[float, float, float]]):
        header = self._pcd_header("x y z", "4 4 4", "F F F", "1 1 1", len(points), len(points), "binary")
        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            for x, y, z in points:
                f.write(struct.pack("<fff", x, y, z))

    def _write_pcd_binary_xyzi(self, path: str, points: List[Tuple[float, float, float, float]]):
        header = self._pcd_header("x y z intensity", "4 4 4 4", "F F F F", "1 1 1 1", len(points), len(points), "binary")
        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            for x, y, z, i in points:
                f.write(struct.pack("<ffff", x, y, z, i))


def main():
    rclpy.init()
    node = IntegratePointCloudToPCD()
    rclpy.spin(node)
    node.destroy_node()


if __name__ == "__main__":
    main()
