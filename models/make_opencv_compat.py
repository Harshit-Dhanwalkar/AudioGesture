#!/usr/bin/env python3
"""Make MediaPipe hand-landmark ONNX loadable by OpenCV < 4.7

usage: make_opencv_compat.py in.onnx out.onnx        (needs: pip install onnx)
"""
import sys
from collections import Counter

import onnx
from onnx import helper, numpy_helper


def _attr(node, name, default):
    for a in node.attribute:
        if a.name == name:
            return a.i
    return default


def _const_map(graph):
    """name -> numpy array, for every initializer and Constant node output."""
    consts = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant":
            for a in node.attribute:
                if a.name == "value":
                    consts[node.output[0]] = numpy_helper.to_array(a.t)
    return consts


def main(src, dst):
    model = onnx.load(src)
    graph = model.graph
    consts = _const_map(graph)
    applied = Counter()

    for node in graph.node:
        if node.op_type == "Clip" and len(node.input) > 1:
            lo = hi = None
            if node.input[1]:
                lo = float(consts[node.input[1]])
            if len(node.input) > 2 and node.input[2]:
                hi = float(consts[node.input[2]])
            del node.input[1:]
            if lo is not None:
                node.attribute.append(helper.make_attribute("min", lo))
            if hi is not None:
                node.attribute.append(helper.make_attribute("max", hi))
            applied["Clip"] += 1

        elif (node.op_type == "Gemm"
              and _attr(node, "transB", 0) == 0
              and node.input[1] in consts):
            # OpenCV < 4.7 mis-sizes FC when B is K x N (transB=0).
            # Y = A.B == A.(B^T)^T
            name_t = node.input[1] + "_T"
            graph.initializer.append(
                numpy_helper.from_array(consts[node.input[1]].T.copy(), name_t))
            node.input[1] = name_t
            for a in node.attribute:
                if a.name == "transB":
                    a.i = 1
                    break
            else:
                node.attribute.append(helper.make_attribute("transB", 1))
            applied["Gemm"] += 1

        elif node.op_type in ("Squeeze", "Unsqueeze") and len(node.input) == 2:
            # opset 13 moved `axes` from attribute to second input.
            axes = [int(a) for a in consts[node.input[1]].reshape(-1)]
            del node.input[1:]
            node.attribute.append(helper.make_attribute("axes", axes))
            applied[node.op_type] += 1

    onnx.save(model, dst)
    summary = ", ".join(f"{k}={v}" for k, v in applied.items()) or "no rewrites"
    print(f"{summary}: {src} -> {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
