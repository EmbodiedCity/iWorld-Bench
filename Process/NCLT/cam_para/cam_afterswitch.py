import numpy as np

def generate_rotated_calibration(original_intrinsic, original_size, rotated_size):
    orig_width, orig_height = original_size
    
    rot_width, rot_height = rotated_size
    
    fx = original_intrinsic[0, 0]
    fy = original_intrinsic[1, 1]
    cx = original_intrinsic[0, 2]
    cy = original_intrinsic[1, 2]
    
    
    new_cx = orig_height - 1 - cy
    new_cy = cx
    
    new_intrinsic = np.array([
        [fx, 0, new_cx],
        [0, fy, new_cy],
        [0, 0, 1]
    ])
    
    fx_norm = fx / rot_width
    fy_norm = fy / rot_height
    cx_norm = new_cx / rot_width
    cy_norm = new_cy / rot_height
    
    result = []
    result.append("# Intrinsic Matrix (3x3)")
    result.append(f"{fx:.6f} 0.000000 {new_cx:.6f} 0.000000 {fy:.6f} {new_cy:.6f} 0.000000 0.000000 1.000000")
    
    result.append("# Image Size (width=x-axis, height=y-axis)")
    result.append(f"{rot_width} {rot_height}")
    
    result.append("# Standardized Intrinsics (normalized by image size)")
    result.append(f"{fx_norm:.6f} {fy_norm:.6f} {cx_norm:.6f} {cy_norm:.6f}")
    
    return result, new_intrinsic

if __name__ == "__main__":
    K_original = np.array([
        [399.433184, 0, 826.361952],
        [0, 399.433184, 621.668624],
        [0, 0, 1]
    ])
    
    original_size = (1616, 1232)
    
    rotated_size = (1232, 1616)
    
    print("rotated K after CW270 (CCW90)")
    print(f"orig={original_size[0]}x{original_size[1]} rot={rotated_size[0]}x{rotated_size[1]}")
    
    calibration_lines, K_new = generate_rotated_calibration(
        K_original, original_size, rotated_size
    )
    
    for line in calibration_lines:
        print(line)
    
    with open("camera_calibration_rotated.txt", "w") as f:
        f.write("\n".join(calibration_lines))
    print("wrote camera_calibration_rotated.txt")