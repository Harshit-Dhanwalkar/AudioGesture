#!/usr/bin/env python3
"""Make MediaPipe hand-landmark ONNX loadable by OpenCV < 4.7

opencv_zoo model exports ReLU6 as `Clip(x, min, max)` with min/max passed as
extra *inputs* (ONNX opset >= 11).  OpenCV before 4.7 only understands old
form where min/max are *attributes*, and refuses to load file:

    parseClip ... (expected: 'node_proto.input_size() == 1')

This rewrites every such Clip node into attribute form. Maths is identical; result is not a strictly valid opset-13 ONNX file (so don't feed it to onnxruntime) but OpenCV >= 4.5 reads it, and newer OpenCV still accepts it

usage: make_opencv_compat.py in.onnx out.onnx        (needs: pip install onnx)
"""
import sys

import onnx
from onnx import helper, numpy_helper


def _attr(node, name, default):
    for a in node.attribute:
        if a.name == name:
            return a.i

    return default

def init_value(model, name):
    for i in model.graph.initializer:
        if i.name == name:
            return numpy_helper.to_array(i).item()

    for n in model.graph.node:
        if n.op_type == 'Constant' and n.output[0] == name:
            for a in n.attribute:
                if a.name == 'value':
                    return numpy_helper.to_array(a.t).item()

    return None

# def main(src: str, dst: str) -> int:
#     model = onnx.load(src)
#     graph = model.graph
#
#     consts = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
#     for node in graph.node:
#         if node.op_type == "Constant":
#             for attr in node.attribute:
#                 if attr.name == "value":
#                     consts[node.output[0]] = numpy_helper.to_array(attr.t)
#
#     fixed = {}
#     for node in graph.node:
#         if node.op_type == "Clip" and len(node.input) > 1:
#             lo = hi = None
#             if node.input[1]:
#                 lo = float(consts[node.input[1]])   # KeyError => min/max is not a constant
#             if len(node.input) > 2 and node.input[2]:
#                 hi = float(consts[node.input[2]])
#             del node.input[1:]
#             if lo is not None:
#                 node.attribute.append(helper.make_attribute("min", lo))
#             if hi is not None:
#                 node.attribute.append(helper.make_attribute("max", hi))
#         elif node.op_type == "Gemm" and _attr(node, "transB", 0) == 0 and node.input[1] in consts:
#             # OpenCV < 4.7 mis-sizes FC layer when B is stored K x N (transB=0)
#             # Pre-transpose weights and flip transB: Y = A.B == A.(B^T)^T
#             name_t = node.input[1] + "_T"
#             graph.initializer.append(numpy_helper.from_array(consts[node.input[1]].T.copy(), name_t))
#             node.input[1] = name_t
#             for a in node.attribute:
#                 if a.name == "transB":
#                     a.i = 1
#                     break
#             else:
#                 node.attribute.append(helper.make_attribute("transB", 1))
#         elif node.op_type in ("Squeeze", "Unsqueeze") and len(node.input) == 2:
#             # opset 13 moved `axes` from an attribute to a second input.
#             axes = [int(a) for a in consts[node.input[1]].reshape(-1)]
#             del node.input[1:]
#             node.attribute.append(helper.make_attribute("axes", axes))
#         else:
#             continue
#         fixed[node.op_type] = fixed.get(node.op_type, 0) + 1
#
#     onnx.save(model, dst)
#     print(f"rewrote {fixed}: {src} -> {dst}")
#     return 0
#
# if __name__ == "__main__":
#     if len(sys.argv) != 3:
#         sys.exit(__doc__)
#     sys.exit(main(sys.argv[1], sys.argv[2]))

def main(src, dst):
    model = onnx.load(src)
    for node in model.graph.node:
        if node.op_type != 'Clip' or len(node.input) < 2:
            continue
        mn = init_value(model, node.input[1]) if node.input[1] else None
        mx = (init_value(model, node.input[2])
              if len(node.input) >= 3 and node.input[2] else None)
        del node.input[1:]
        for name, val in (('min', mn), ('max', mx)):
            if val is None: continue
            if any(a.name == name for a in node.attribute): continue
            node.attribute.append(helper.make_attribute(name, float(val)))
    for imp in model.opset_import:
        if imp.domain in ('', 'ai.onnx') and imp.version > 10:
            imp.version = 10
    onnx.checker.check_model(model)   # this should now pass
    onnx.save(model, dst)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
