#!/usr/bin/env python3
"""Set bag record times to sensor header times without modifying sensor data."""

import argparse
import heapq
from pathlib import Path

import rosbag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    if partial.exists():
        parser.error("partial output already exists; choose a new output path")

    pending = []
    latest_ns = 0
    last_written_ns = -1
    counts = {}

    def write_one(output):
        nonlocal last_written_ns
        stamp_ns, _, topic, msg, header = heapq.heappop(pending)
        if stamp_ns < last_written_ns:
            raise ValueError("sensor time goes backwards beyond the 0.5 s reorder window")
        output.write(topic, msg, msg.header.stamp, connection_header=header)
        last_written_ns = stamp_ns
        counts[topic] = counts.get(topic, 0) + 1

    # Scan packets can arrive after newer IMU samples. Sort within a bounded window
    # so the adapted bag and its playback clock both advance in sensor time.
    try:
        with rosbag.Bag(str(args.input), "r") as source, rosbag.Bag(str(partial), "w") as output:
            for index, (topic, msg, _, header) in enumerate(source.read_messages(return_connection_header=True)):
                if not hasattr(msg, "header") or msg.header.stamp.to_nsec() <= 0:
                    raise ValueError("missing sensor header stamp on " + topic)
                stamp_ns = msg.header.stamp.to_nsec()
                latest_ns = max(latest_ns, stamp_ns)
                heapq.heappush(pending, (stamp_ns, index, topic, msg, header))
                while pending and pending[0][0] <= latest_ns - 500_000_000:
                    write_one(output)
                if index and index % 10000 == 0:
                    print("Processed", index, "messages", flush=True)
            while pending:
                write_one(output)
        partial.rename(args.output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    print("Saved", args.output)
    for topic, count in sorted(counts.items()):
        print(topic, count)


if __name__ == "__main__":
    main()
