import os
import json

# ================= 配置区域 =================
# 数据根目录路径 (方便后续调整)
BASE_DIR = r"/home/a5000/Work/Chen/adobe_lr/scraped_data_v4"
OUTPUT_FILE = r"/home/a5000/Work/Chen/adobe_lr/all_combined_details.json"
# ===========================================

def merge_details():
    """
    遍历 BASE_DIR 下的所有图片目录，将所有版本下的 details.json 合并为一个总的 json 文件。
    """
    if not os.path.exists(BASE_DIR):
        print(f"错误: 目录不存在 - {BASE_DIR}")
        return

    all_images_data = []

    # 遍历根目录下的所有子目录 (每个子目录对应一个 Image ID)
    for image_id in os.listdir(BASE_DIR):
        image_dir = os.path.join(BASE_DIR, image_id)
        
        if not os.path.isdir(image_dir):
            continue
            
        print(f"正在处理图片 ID: {image_id}")
        
        combined_data = {
            "image_id": image_id,
            "versions": {}
        }
        
        # 查找该图片目录下的所有 version_x 文件夹
        version_dirs = [d for d in os.listdir(image_dir) if d.startswith("version_") and os.path.isdir(os.path.join(image_dir, d))]
        
        # 按版本号排序 (version_0, version_1, ...)
        try:
            version_dirs.sort(key=lambda x: int(x.split("_")[1]))
        except:
            version_dirs.sort()
            
        found_versions = False
        
        for v_dir in version_dirs:
            v_path = os.path.join(image_dir, v_dir)
            details_path = os.path.join(v_path, "details.json")
            
            if os.path.exists(details_path):
                try:
                    with open(details_path, "r", encoding="utf-8") as f:
                        details = json.load(f)
                        combined_data["versions"][v_dir] = details
                        found_versions = True
                        print(f"  - 已合并: {v_dir}")
                except Exception as e:
                    print(f"  - 读取失败 {v_dir}: {e}")
            else:
                print(f"  - 跳过 {v_dir} (无 details.json)")
        
        if found_versions:
            all_images_data.append(combined_data)
            # 保存合并后的 json 到图片根目录
            output_path = os.path.join(image_dir, "combined_details.json")
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(combined_data, f, ensure_ascii=False, indent=2)
                print(f"  >>> 合并完成，已保存至: {output_path}\n")
            except Exception as e:
                print(f"  >>> 保存合并文件失败: {e}\n")
        else:
            print(f"  >>> 未找到任何版本的 details.json，跳过。\n")

    # 保存包含所有图片版本的最终 JSON 文件
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(all_images_data, f, ensure_ascii=False, indent=2)
        print(f"===========================================")
        print(f"所有的 JSON 数据已成功合并，已保存至: {OUTPUT_FILE}")
    except Exception as e:
        print(f"保存总合并文件失败: {e}")

if __name__ == "__main__":
    print(f"开始合并 JSON 数据，根目录: {BASE_DIR}\n")
    merge_details()
    print("所有任务完成。")
