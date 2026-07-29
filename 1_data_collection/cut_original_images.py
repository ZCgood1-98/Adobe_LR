import cv2
import numpy as np
from pathlib import Path

def detect_side_black_borders(
    image_bgr: np.ndarray,
    black_thresh: int = 50,
    edge_scan_ratio: float = 0.18,
    min_height_coverage: float = 0.92,
    max_allowed_gap_ratio: float = 0.02,
):
    """
    返回: (left_crop, right_crop)
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1) 近黑区域
    black = (gray <= black_thresh).astype(np.uint8) * 255

    # 2) 轻微闭运算，减少噪点/小孔
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    black = cv2.morphologyEx(black, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 3) 连通域分析
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((black > 0).astype(np.uint8), connectivity=8)

    # 标记哪些连通域触达左边/右边
    left_labels = set(np.unique(labels[:, 0]))
    right_labels = set(np.unique(labels[:, -1]))

    # 构建左右边缘黑区mask
    left_mask = np.isin(labels, list(left_labels)).astype(np.uint8)
    right_mask = np.isin(labels, list(right_labels)).astype(np.uint8)

    # 只关心靠边的一段宽度范围
    scan_w = max(1, int(w * edge_scan_ratio))
    left_region = left_mask[:, :scan_w]
    right_region = right_mask[:, w - scan_w :]

    def estimate_crop_width(region_mask, side="left"):
        """
        region_mask: h x scan_w (0/1)
        """
        col_coverage = region_mask.mean(axis=0)  # 每列黑像素占比
        valid_cols = col_coverage >= min_height_coverage

        # 允许少量间断：做一维“闭运算”近似（卷积）
        if valid_cols.any():
            arr = valid_cols.astype(np.uint8)
            k = 5
            padded = np.pad(arr, (k//2, k//2), mode='edge')
            smooth = np.convolve(padded, np.ones(k, dtype=np.uint8), mode='valid')
            valid_cols = smooth >= (k - int(np.ceil(k * max_allowed_gap_ratio)))

        if side == "left":
            # 从左向右找到连续有效列
            idx = np.where(~valid_cols)[0]
            crop = idx[0] if len(idx) else len(valid_cols)
        else:
            # 对右侧区域，按从右向左计算
            rev = valid_cols[::-1]
            idx = np.where(~rev)[0]
            crop = idx[0] if len(idx) else len(rev)

        return int(crop)

    left_crop = estimate_crop_width(left_region, side="left")
    right_crop = estimate_crop_width(right_region, side="right")

    return left_crop, right_crop

def refine_trim_no_black_edge(img, black_thresh=50, edge_black_ratio=0.005,
                               max_trim=150, min_keep_ratio=0.45):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    min_keep_w = max(20, int(w * min_keep_ratio))

    # 每列黑像素占比
    col_black = np.mean(gray <= black_thresh, axis=0)  # shape: (w,)

    # 从左找到第一列"不黑"的位置
    left_cut = 0
    for i in range(min(max_trim, w - min_keep_w)):
        if col_black[i] > edge_black_ratio:
            left_cut = i + 1
        else:
            break  # 遇到干净列就停

    # 从右找
    right_cut = 0
    for i in range(min(max_trim, w - min_keep_w)):
        if col_black[w - 1 - i] > edge_black_ratio:
            right_cut = i + 1
        else:
            break

    if left_cut + right_cut >= w:
        return img, 0, 0

    return img[:, left_cut: w - right_cut], left_cut, right_cut


def crop_black_borders_v2(img: np.ndarray, uniformity_thresh: float = 2.0, max_trim_ratio: float = 0.4):
    """
    用颜色均匀性判断黑边，而非亮度阈值。
    真正的UI黑边：每列像素标准差极小（颜色完全一致）。
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 每列像素的标准差（真正黑边标准差接近0，照片内容标准差较大）
    col_std = gray.std(axis=0)  # shape: (w,)
    # 每行像素的标准差（用于判断是否整行都是黑边）
    row_std = gray.std(axis=1)  # shape: (h,)
    
    max_trim_w = int(w * max_trim_ratio)
    max_trim_h = int(h * max_trim_ratio)
    
    # 从左找到第一列"不均匀"的列（即内容开始）
    left_cut = 0
    for i in range(max_trim_w):
        if col_std[i] < uniformity_thresh:
            left_cut = i + 1
        else:
            break
    
    # 从右找
    right_cut = 0
    for i in range(max_trim_w):
        if col_std[w - 1 - i] < uniformity_thresh:
            right_cut = i + 1
        else:
            break
    
    # 从上找（有些UI黑边可能是上下都有的）    
    top_cut = 0
    for i in range(max_trim_h):
        if row_std[i] < uniformity_thresh:
            top_cut = i + 1
        else:
            break
        
    # 从下找
    bottom_cut = 0
    for i in range(max_trim_h):
        if row_std[h - 1 - i] < uniformity_thresh:
            bottom_cut = i + 1
        else:
            break
            
    if left_cut + right_cut >= w or top_cut + bottom_cut >= h:
        return img, 0, 0, 0, 0
    
    trimmed = img[
        top_cut: h - bottom_cut if bottom_cut > 0 else h,
        left_cut: w - right_cut if right_cut > 0 else w
    ]
    return trimmed, left_cut, right_cut, top_cut, bottom_cut


def process_folder(root: str):
    root_path = Path(root)
    files = list(root_path.rglob("original_captured.png"))
    print(f"Found {len(files)} images")

    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            print(f"[WARN] failed to read: {p}")
            continue

        cropped, l, r, t, b = crop_black_borders_v2(img)
        out_path = p.with_name("original_captured_cropped.png")
        cv2.imwrite(str(out_path), cropped)
        print(f"[OK] {p} -> {out_path} (left={l}, right={r}, top={t}, bottom={b})")


if __name__ == "__main__":
    process_folder("scraped_data_v4")