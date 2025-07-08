import struct

def pack_lane_result(result):
    # uuid: 4바이트 정수 (big endian)
    b = struct.pack('>I', result["uuid"])  # '>I': big endian, unsigned int (4 bytes)

    # offset(float), steering_angle(float)
    b += struct.pack('ff', result["offset"], result["steering_angle"])

    # skeleton points
    points = result["skeleton_points"]
    b += struct.pack('H', len(points))  # number of points (unsigned short)

    for x, y in points:
        b += struct.pack('HH', x, y)  # each point: (x, y) as unsigned shorts
    return b
