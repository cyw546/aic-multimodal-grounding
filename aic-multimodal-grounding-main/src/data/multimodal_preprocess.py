import cv2
import numpy as np


def read_rgb(img_path):
    """读取RGB图像，返回HWC RGB格式 0~255 uint8"""
    bgr = cv2.imread(img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def read_infrared(img_path, target_hw):
    """
    读取红外单通道图像
    :param img_path: 红外图片路径
    :param target_hw: (H,W) 和rgb对齐的尺寸
    :return: shape [H,W,1] float32 0~1
    """
    ir = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    ir = cv2.resize(ir, (target_hw[1], target_hw[0]))
    ir = ir.astype(np.float32) / 255.0
    return ir[..., np.newaxis]


def read_depth_16bit(img_path, target_hw):
    """
    读取16位uint16深度图，过滤无效值，归一化，resize对齐rgb
    :param img_path: 16bit深度图路径
    :param target_hw: (H,W) 和rgb对齐的尺寸
    :return: shape [H,W,1] float32 0~1
    """
    depth_raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    assert depth_raw.dtype == np.uint16, "深度图必须是16位"

    # 过滤无效0
    depth_valid = depth_raw[depth_raw > 0]
    if len(depth_valid) == 0:
        depth_norm = np.zeros_like(depth_raw, dtype=np.float32)
    else:
        d_min, d_max = depth_valid.min(), depth_valid.max()
        depth_norm = (depth_raw.astype(np.float32) - d_min) / (d_max - d_min + 1e-6)

    depth_norm = cv2.resize(depth_norm, (target_hw[1], target_hw[0]))
    return depth_norm[..., np.newaxis]


def multimodal_early_fusion(rgb, ir, depth):
    """
    早期融合：RGB(3)+红外(1)+深度(1) → 5通道
    rgb: H,W,3  0~255 uint8
    ir: H,W,1 0~1 float32
    depth: H,W,1 0~1 float32
    return: H,W,5 float32
    """
    rgb_norm = rgb.astype(np.float32) / 255.0
    fuse = np.concatenate([rgb_norm, ir, depth], axis=-1)
    return fuse


if __name__ == "__main__":
    # 测试示例，改成你本地样例图片路径运行测试
    rgb_img = read_rgb("./sample/rgb.png")
    H, W = rgb_img.shape[:2]
    ir_img = read_infrared("./sample/ir.png", target_hw=(H, W))
    depth_img = read_depth_16bit("./sample/depth.png", target_hw=(H, W))

    fuse_img = multimodal_early_fusion(rgb_img, ir_img, depth_img)
    print(f"融合后shape: {fuse_img.shape}")