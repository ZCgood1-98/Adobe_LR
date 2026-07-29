import json
import time
import os
import sys
import re
import shutil
from PIL import Image, ImageOps
from playwright.sync_api import sync_playwright

# --- 配置 / Configuration ---
# 限制最大抓取数量 (设置为 None 表示不限制，或者具体的数字)
MAX_SCRAPE_COUNT = 100

# 每张图片最多抓取多少个修改版本。
# 这是“最多检查多少个候选版本”的上限，不是最终保存数量。
# 设为 None 表示不限制，遍历该图片下的所有版本候选。
MAX_VERSIONS_PER_IMAGE = None

# 每组图片最多保留多少个“无 Mask 的版本图”。
# 主图不计入这里；最终结果是 1 张主图 + 最多 3 张无 Mask 版本图。
MAX_VALID_VERSIONS_TO_KEEP = 3

# 需要抓取的图片索引（从0开始计数，0=第1张，23=第24张，29=第30张）
# 如果为空列表 []，则自动抓取前 MAX_SCRAPE_COUNT 张
TARGET_INDICES = []

# 逻辑处理：
# 1. 如果 TARGET_INDICES 不为空，则优先使用它，并根据 MAX_SCRAPE_COUNT 截取
# 2. 如果 TARGET_INDICES 为空，则生成 0 到 MAX_SCRAPE_COUNT-1 的序列
if TARGET_INDICES:
    if MAX_SCRAPE_COUNT is not None:
        TARGET_INDICES = TARGET_INDICES[:MAX_SCRAPE_COUNT]
else:
    if MAX_SCRAPE_COUNT is not None:
        TARGET_INDICES = list(range(MAX_SCRAPE_COUNT))
    else:
        # 如果既没有指定索引，也没有指定最大数量，默认抓取前10张以防无限抓取
        TARGET_INDICES = list(range(10))

def auto_rotate_image(image_path):
    """
    根据 EXIF 信息自动旋转图片，修正方向。
    """
    try:
        # 1. 打开图片
        image = Image.open(image_path)
        
        # 2. 获取旋转后的图片
        # exif_transpose 会返回一个新的 Image 对象（如果需要旋转），或者原对象
        transposed_image = ImageOps.exif_transpose(image)
        
        # 3. 如果对象不同，说明进行了旋转
        if transposed_image is not image:
            # 必须先关闭原图句柄，才能覆盖写入
            image.close()
            
            # 保存旋转后的图片
            transposed_image.save(image_path)
            print(f"    [自动旋转] 已根据 EXIF 修正图片方向: {os.path.basename(image_path)}")
        else:
            # 没有旋转，关闭文件
            image.close()
            
    except Exception as e:
        print(f"    [自动旋转] 处理失败 (可能无 EXIF 或非 JPG): {e}")

def download_high_res_image(page, save_path, specific_image_id=None, direct_url=None):
    """
    通过构造 API URL 下载高清原图 (PNG 格式)
    specific_image_id: 指定要下载的 Image ID (UUID)
    direct_url: 如果提供了直接 URL，则直接下载该 URL
    """
    try:
        # [新增] 支持直接 URL 下载
        if direct_url:
            print(f"    [下载] 使用直接 URL: {direct_url[:80]}...")
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            try:
                 response = page.request.get(direct_url, headers=headers, timeout=30000)
                 if response.status == 200:
                     # 确保目录存在
                     os.makedirs(os.path.dirname(save_path), exist_ok=True)
                     with open(save_path, "wb") as f:
                         f.write(response.body())
                     return True
            except Exception as e:
                print(f"    [错误] 直接下载失败: {e}")
                return False

        image_id = specific_image_id
        
        # 如果没有提供 specific_image_id，尝试从 URL 提取
        if not image_id:
            # 1. 从当前 URL 提取 UUID
            # URL 格式示例: https://lightroom.adobe.com/learn/discover/25b04011-4719-4a25-998f-a83a222a21e1
            current_url = page.url
            match = re.search(r"discover/([a-f0-9-]+)", current_url)
            if match:
                image_id = match.group(1)
        
        if not image_id:
            print("    无法从参数或 URL 提取图片 ID，跳过高清下载")
            return False
            
        # 2. 构造高清 PNG URL (尝试多种尺寸)
        # 优先尝试 4096，失败则降级
        target_sizes = ['4096', '2048', '1280']
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        # 尝试获取 Authorization header
        try:
            auth = page.evaluate("() => window.localStorage.getItem('access_token')")
            if auth: headers["Authorization"] = f"Bearer {auth}"
        except: pass

        for size in target_sizes:
            high_res_url = f"https://cdn.cp.adobe.io/content/2/rendition/{image_id}/version/1/format/png/dimension/height/size/{size}"
            print(f"    尝试下载高清原图 (Size: {size}): {high_res_url}")
            
            try:
                # [修改] 增加网络请求超时时间，从 15 秒增加到 60 秒，防止大图下载超时
                response = page.request.get(high_res_url, headers=headers, timeout=60000) 
                
                if response.status == 200:
                    body = response.body()
                    file_size = len(body)
                    
                    # [修改] 用户要求取消大小限制，只要下载到图片就算成功
                    # 检查文件大小 (不再限制 300KB)
                    min_size = 10 # 象征性限制，只要不是空文件就行
                    
                    if file_size < min_size:
                        print(f"    [警告] 下载的文件为空，尝试下一尺寸")
                        continue
                    
                    # 确保目录存在 (防止路径不存在报错)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    with open(save_path, "wb") as f:
                        f.write(body)
                    
                    auto_rotate_image(save_path)
        
                    print(f"    高清原图下载成功! (Size: {size}) 大小: {file_size/1024/1024:.2f} MB")
                    return True
                else:
                    print(f"    下载失败 (Size: {size})，状态码: {response.status}")
                    
                    # [关键] 遇到 503/404/429 主动中断，避免无意义重试
                    if response.status in [404, 503, 429]:
                        print(f"    [严重错误] 服务器返回 {response.status}，脚本主动中断以保护环境/避免无效请求。")
                        sys.exit(1)
                    
            except Exception as e:
                print(f"    下载请求出错 (Size: {size}): {e}")
                
        print("    所有尺寸尝试均失败")
        return False

    except Exception as e:
        print(f"    高清下载过程发生异常: {e}")
        return False

def capture_original_image(page, save_dir):
    """
    通过点击'原照设置'并直接从 Canvas/Image 中截取或提取原图。
    因为 Adobe 更改了加载策略，不再使用 Blob，而是切片渲染，所以这里采用强截取策略。
    """
    print("  [原图捕获] 开始尝试捕获原照...")
    
    # 1. 查找 '原照设置' (Original)
    target_texts = ["原照设置", "Original", "原照"]
    target_element = None
    for text in target_texts:
        el = page.locator(f"text='{text}'").first
        if el.is_visible():
            target_element = el
            break
            
    if not target_element:
        print("  [原图捕获] 未找到 '原照设置' 按钮，跳过")
        return False

    # 2. 点击原照，给足时间渲染
    print("    点击 '原照' 按钮，等待渲染...")
    try:
        target_element.click()
        time.sleep(5) # 给足够长的时间让切片加载并渲染到 Canvas 上
    except Exception as e:
        print(f"    点击原照失败: {e}")
        return False

    # 3. 寻找主画布并尝试截取
    print("    正在尝试从画布提取原图...")
    try:
        # 寻找主要的图片容器
        canvas_container = page.locator("canvas, .Loupe-image, img[class*='Loupe-image']").first
        
        if canvas_container.count() == 0:
            print("    [原图捕获] 找不到画布或图片容器")
            return False

        # 尝试通过 toDataURL 提取 (最清晰，无损)
        save_as_path = os.path.join(save_dir, "original_captured.png")
        
        # 判断是 img 还是 canvas
        tag_name = canvas_container.evaluate("el => el.tagName.toLowerCase()")
        
        if tag_name == "img":
            # 如果是普通的 img 标签，尝试获取 src
            src = canvas_container.get_attribute("src")
            if src and not src.startswith("blob:"):
                print("    发现 img src，尝试直接下载...")
                if download_high_res_image(page, save_as_path, specific_image_id=None, direct_url=src):
                    return True

        # 对于 Canvas 或者 Blob img，尝试提取 DataURL
        data_url = canvas_container.evaluate("""(el) => {
            try {
                if (el.tagName.toLowerCase() === 'canvas') {
                    return el.toDataURL('image/png', 1.0);
                }
            } catch(e) {
                return null;
            }
            return null;
        }""")
        
        if data_url and data_url.startswith("data:image"):
            import base64
            header, encoded = data_url.split(",", 1)
            data = base64.b64decode(encoded)
            with open(save_as_path, "wb") as f:
                f.write(data)
            print(f"    [成功] 通过 Canvas 提取保存原图: {len(data)/1024/1024:.2f} MB")
            return True
            
        # 如果 toDataURL 失败 (比如被 Tainted 跨域污染)，直接截图
        print("    DataURL 提取失败或不可用，回退到元素高质量截图...")
        # 确保截取的是没有其他 UI 遮挡的部分
        canvas_container.screenshot(path=save_as_path)
        print(f"    [成功] 已通过元素截图保存原图: original_captured.png")
        return True

    except Exception as e:
        print(f"    [原图捕获] 提取过程异常: {e}")
        return False

def scan_for_mask(page):
    """
    快速扫描是否存在 Mask 步骤
    返回 True 如果发现 Mask
    """
    print("    [Mask检测] 正在扫描是否存在 Mask...")

    # 1. 查找滚动容器
    scroll_container = page.locator("div[class*='Scrollbar'], div[class*='DiscoverSideBar']").first
    if scroll_container.count() == 0:
        # 尝试等待一下
        try:
             page.wait_for_selector("div[class*='Scrollbar'], div[class*='DiscoverSideBar']", timeout=3000)
        except: pass
        if scroll_container.count() == 0:
            return False

    # 2. 也是滚动扫描，但只要发现立即返回
    try:
        # 先检查当前视口（无需滚动）
        if page.locator("div[class*='DiscoverStepMasking']").count() > 0:
            print("    [Mask检测] (首屏) 发现 Mask！")
            return True

        scroll_container.evaluate("(el) => el.scrollTop = 0")
        time.sleep(0.3)

        last_scroll_top = -1
        no_change_count = 0

        # 稍微快速滚几次
        for _ in range(30):
            # 检查当前视口内的 Mask
            if page.locator("div[class*='DiscoverStepMasking']").count() > 0:
                print("    [Mask检测] 发现 Mask！")
                return True

            # 滚动
            scroll_container.evaluate("(el) => el.scrollBy(0, 800)")
            time.sleep(0.3)

            # 检查到底
            curr = scroll_container.evaluate("(el) => el.scrollTop")
            if curr == last_scroll_top:
                no_change_count += 1
                if no_change_count >= 2: break
            else:
                no_change_count = 0
                last_scroll_top = curr

        return False

    except Exception as e:
        print(f"    [Mask检测] 出错: {e}")
        return False

def wait_for_edits_loaded(page):
    """
    等待 '正在加载编辑...' 消失并且内容稳定
    轮询检查，一旦有数据立即返回，不再硬等待
    """
    print("  等待编辑完成加载 (轮询检测)...")
    
    # 1. 快速检查侧边栏容器
    try:
        # 使用更通用的选择器等待侧边栏容器
        page.wait_for_selector("div[class*='DiscoverStepsRoot'], div[class*='DiscoverSideBar']", timeout=60000)
    except:
        print("  侧边栏容器等待超时")
    
    # 2. 轮询检查：加载文字消失 AND 出现编辑步骤
    max_wait_time = 6000 # 增加到 6000秒 (100分钟)，实际上相当于无限等待，直到加载完成
    poll_interval = 2 # 2秒轮询一次
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        try:
            # 检查加载文字
            loading_cn = page.locator("text='正在加载编辑...'").count()
            loading_en = page.locator("text='Loading edits...'").count()
            
            # 检查是否有具体步骤内容 (表明数据已加载)
            steps_count = page.locator("div[class*='DiscoverStep-']").count()
            
            # 如果没有加载文字 并且 已经有步骤显示
            if loading_cn == 0 and loading_en == 0 and steps_count > 0:
                # 稍微缓冲一下，确保渲染稳定
                time.sleep(1)
                loading_cn_2 = page.locator("text='正在加载编辑...'").count()
                loading_en_2 = page.locator("text='Loading edits...'").count()
                
                if loading_cn_2 == 0 and loading_en_2 == 0:
                    print(f"  加载完成 (发现 {steps_count} 个步骤)")
                    return
            
            # 如果没有加载文字但也没步骤，可能是刚开始加载或确实为空，继续等待
            
        except:
            pass
            
        time.sleep(poll_interval)
        elapsed = int(time.time() - start_time)
        if elapsed > 0 and elapsed % 5 == 0:
             print(f"    已等待 {elapsed}s...")
             
    print("  等待超时 (继续尝试抓取)")

def parse_transform_y(style_attr):
    """
    从 style 属性中解析 Y 轴位移值
    """
    if not style_attr: return -1
    # 支持 translate(0px, 100px) 和 translate3d(0px, 100px, 0px)
    match = re.search(r"translate(?:3d)?\(\s*0px\s*,\s*([\d.]+)px", style_attr)
    if match:
        return float(match.group(1))
    return -1

def close_blue_tooltip(page):
    """
    如果出现蓝色提示框或特定的遮挡按钮，将其关闭
    """
    try:
        # [新增] 处理 Toast/新手引导弹窗拦截 (根据用户报错日志新增)
        # 匹配包含 "了解摄影师" 等文本的 Toast 容器并尝试点击关闭或点掉它
        toast_container = page.locator("div[class*='Toast'], div[class*='ToastContainer']").filter(has_text="了解摄影师").first
        if toast_container.count() > 0 and toast_container.is_visible():
            print("  尝试关闭新手引导 Toast 弹窗...")
            # 尝试找关闭按钮 (X)，如果没有，点击页面中心来取消焦点
            close_btn = toast_container.locator("button, svg").first
            if close_btn.count() > 0:
                close_btn.click(force=True)
            else:
                page.mouse.click(100, 100) # 点击左上角空白处关掉气泡
            time.sleep(1)

        # 0. 优先级最高：强力暴力点击所有可能的关闭按钮
        # 查找页面上所有可能是“关闭”图标的按钮
        # 策略：查找所有包含 SVG 的按钮，且该 SVG 的 path 看起来像个 "X"
        potential_close_btns = page.locator("button:has(svg)").all()
        for btn in potential_close_btns:
            try:
                if btn.is_visible():
                    # 检查 SVG path (更宽泛的匹配)
                    svg_html = btn.inner_html()
                    # 匹配常见的 "X" 形状路径特征 (通常很短，或者包含特定的 M/L 指令)
                    # 这里直接尝试点击所有类名包含 Button 且尺寸较小的按钮
                    class_attr = btn.get_attribute("class") or ""
                    if "Button" in class_attr and btn.bounding_box()['width'] < 50:
                         # 再次确认不是什么重要的功能按钮，而是悬浮层上的
                         # 通常关闭按钮没有文字内容
                         if not btn.inner_text().strip():
                             print(f"  尝试点击潜在的关闭按钮: {class_attr[:20]}...")
                             btn.click()
                             time.sleep(0.5)
            except: pass

        # 1. 检查用户指定的特定关闭按钮 (精确匹配)
        # Class: Button-qi839s62Ae0OZBo6
        close_btn_specific = page.locator("button[class*='Button-qi839s62Ae0OZBo6']").first
        if close_btn_specific.count() > 0 and close_btn_specific.is_visible():
            print("  关闭特定的遮挡按钮 (Button-qi839s62Ae0OZBo6)...")
            close_btn_specific.click()
            time.sleep(1)
        
        # 备用：通过 SVG path 查找 (精确匹配用户提供的 path)
        # d="M7.8 6.7L5.1 4l2.7-2.7..."
        close_btn_svg = page.locator("xpath=//button[descendant::path[starts-with(@d, 'M7.8 6.7')]]").first
        if close_btn_svg.count() > 0 and close_btn_svg.is_visible():
             print("  关闭特定的遮挡按钮 (SVG Path)...")
             close_btn_svg.click()
             time.sleep(1)

        # 2. 通过文本或关闭按钮查找提示框 (CoachMark)
        # 文本: "了解摄影师如何..."
        tooltip = page.locator("div[role='tooltip'], div[class*='CoachMark']").filter(has_text="了解摄影师").first
        if tooltip.count() > 0 and tooltip.is_visible():
            print("  关闭蓝色提示框...")
            # 尝试找到内部的关闭按钮
            close_btn = tooltip.locator("button, div[class*='Close']").first
            if close_btn.count() > 0:
                close_btn.click()
            else:
                # 点击提示框本身或其他地方？
                tooltip.click()
            time.sleep(1)
            
        # 3. 通用的 CoachMark 关闭按钮
        close_btn_gen = page.locator(".CoachMark-close, button[aria-label='Close'], button[aria-label='关闭']").filter(has=page.locator("xpath=ancestor::div[contains(@class, 'CoachMark')]")).first
        if close_btn_gen.count() > 0 and close_btn_gen.is_visible():
             close_btn_gen.click()
             time.sleep(1)
             
    except: pass

def scrape_current_view(page, save_dir, image_id=None, skip_mask_details=False):
    """
    抓取当前视图的参数和 Mask 详情。
    当 skip_mask_details=True 时，只保留基础步骤信息，不展开 Mask 详情。
    主图下载逻辑已移至外部处理，此处仅负责记录主图路径（如果存在）。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 首先关闭任何提示框
    close_blue_tooltip(page)
        
    extracted_data = {
        "url": page.url,
        "image_id": image_id,
        "details": [],
        "masks": [],
        "main_image": None
    }
    
    # --- 1. 记录主图路径 (已在外部下载) ---
    main_img_path = os.path.join(save_dir, "main_image.png")
    if os.path.exists(main_img_path):
        extracted_data["main_image"] = main_img_path
    else:
        print("  注意: 未找到主图 main_image.png")

    # --- 2. 抓取编辑步骤 (两阶段策略) ---
    print("  抓取编辑步骤 (两阶段: 扫描 + 详情)...")
    
    # 查找可滚动容器 - 更健壮的选择器
    # 注意：如果下载图片时还没有完全加载，这里可能需要再次确认加载状态
    # 但由于 process_image_page 中已经调用了 wait_for_edits_loaded，这里通常是安全的
    scroll_container = page.locator("div[class*='Scrollbar'], div[class*='DiscoverSideBar']").first
    
    if scroll_container.count() == 0:
        print("    未找到滚动容器，尝试等待...")
        try:
             page.wait_for_selector("div[class*='Scrollbar'], div[class*='DiscoverSideBar']", timeout=10000)
             scroll_container = page.locator("div[class*='Scrollbar'], div[class*='DiscoverSideBar']").first
        except: pass

    mask_dir = os.path.join(save_dir, "masks")
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
        
    collected_steps = {} # Key: Y位置 (int), Value: 步骤信息 dict
    
    # (scroll_container 已经在前面定义并尝试查找)
    
    if scroll_container.count() > 0:
        print("    找到滚动容器")
        
        # 重置滚动到顶部
        try:
            scroll_container.evaluate("(el) => el.scrollTop = 0")
        except: pass
        time.sleep(1)
        
        # --- 阶段 1: 扫描 (滚动并收集基本信息，不点击) ---
        print("    阶段 1: 扫描所有步骤 (不展开)...")
        
        step_height = 200 # 较小的步长以获得更好的覆盖
        last_scroll_top = -1
        no_change_count = 0
        max_iterations = 100 
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # 获取可见步骤
            steps = page.locator("div[class*='DiscoverStep-']").all()
            
            for step in steps:
                try:
                    style = step.get_attribute("style")
                    y_pos = parse_transform_y(style)
                    
                    # 如果 y_pos 为 -1，尝试生成合成键
                    if y_pos == -1: 
                        content_hash = hash(step.inner_text())
                        y_pos = 999000 + abs(content_hash) % 1000
                    
                    if y_pos in collected_steps:
                        continue # 已收集
                        
                    # 基本解析 (仅识别类型)
                    step_info = {}
                    inner_text = step.inner_text()
                    inner_html = step.inner_html()
                    
                    # 识别类型
                    if step.locator("div[class*='DiscoverStepMasking']").count() > 0:
                        step_info["type"] = "masking"
                        step_info["needs_expansion"] = True
                        step_info["label"] = "Mask" # 占位符
                    elif "DiscoverStepColorMixer" in inner_html or "混色器" in inner_text or "Color Mixer" in inner_text:
                        step_info["type"] = "color_mixer"
                        try:
                            color_dot = step.locator("div[style*='border-color']").first
                            if color_dot.count() > 0:
                                step_info["color_style"] = color_dot.get_attribute("style")
                            labels = step.locator("div[class*='DiscoverStepColorMixer-Ag7eowQG6Hk6wswa']").all_inner_texts()
                            values = step.locator("div[class*='DiscoverStepColorMixer-A3ovjnV1y6QNRGfa']").all_inner_texts()
                            mix_vals = {}
                            min_len = min(len(labels), len(values))
                            for k in range(min_len):
                                mix_vals[labels[k].strip()] = values[k].strip()
                            step_info["values"] = mix_vals
                        except: step_info["raw_text"] = inner_text
                    elif "DiscoverStepCurve" in inner_html or "点曲线" in inner_text or "Point Curve" in inner_text:
                        step_info["type"] = "point_curve"
                        # 立即截取曲线截图 (通常可见无需展开)
                        try:
                            c_path = os.path.join(mask_dir, f"curve_{int(y_pos)}.png")
                            step.screenshot(path=c_path)
                            step_info["curve_screenshot"] = c_path
                        except: pass
                    elif "DiscoverStepCrop" in inner_html or "裁剪" in inner_text:
                        step_info["type"] = "crop"
                        try:
                            val = step.locator("div[class*='DiscoverStepCrop']").first.inner_text()
                            step_info["value"] = val.strip()
                            step_info["label"] = "裁剪"
                        except: pass
                    else:
                        step_info["type"] = "standard"
                        # 健壮地查找标签/值
                        rows_text_divs = step.locator("div[class*='DiscoverStepStringRow'], div[class*='DiscoverStepParameterRow']").all()
                        if len(rows_text_divs) >= 2:
                            step_info["label"] = rows_text_divs[0].inner_text().strip()
                            step_info["value"] = rows_text_divs[-1].inner_text().strip()
                        elif len(rows_text_divs) == 1:
                            step_info["label"] = rows_text_divs[0].inner_text().strip()
                        
                        if "label" not in step_info and "value" not in step_info:
                             text_content = inner_text.strip()
                             if text_content:
                                 # 尝试通过换行符解析 Label/Value
                                 if "\n" in text_content:
                                     parts = text_content.split("\n")
                                     if len(parts) >= 2:
                                         step_info["label"] = parts[0].strip()
                                         step_info["value"] = parts[-1].strip()
                                     else:
                                         step_info["raw_text"] = text_content
                                 else:
                                     step_info["raw_text"] = text_content
                    
                    # 只有当包含有效信息时才收集
                    if "label" in step_info or "value" in step_info or "raw_text" in step_info or "color_style" in step_info or "curve_screenshot" in step_info:
                        collected_steps[y_pos] = step_info
                        
                except Exception as e:
                    print(f"      扫描步骤错误: {e}")
            
            # 向下滚动
            try:
                scroll_container.evaluate(f"(el) => el.scrollBy(0, {step_height})")
                time.sleep(1.5) # 等待渲染 (延长以防止漏抓)
                
                # 检查是否到达底部
                current_scroll_top = scroll_container.evaluate("(el) => el.scrollTop")
                if current_scroll_top == last_scroll_top:
                    no_change_count += 1
                    if no_change_count >= 3:
                        print("    滚动到底部 (无变化)")
                        break
                else:
                    no_change_count = 0
                    last_scroll_top = current_scroll_top
            except:
                break
        
        if skip_mask_details:
            print("    阶段 2: 检测到 Mask，跳过展开，仅保留基础步骤数据...")
        else:
            # --- 阶段 2: 展开 Mask (针对性处理) ---
            print("    阶段 2: 展开并抓取 Mask 详情...")
            
            # 重置滚动
            scroll_container.evaluate("(el) => el.scrollTop = 0")
            time.sleep(1)
            
            # 再次向下滚动，专门寻找 Mask
            mask_iteration = 0
            processed_masks_y = set()
            
            scroll_container.evaluate("(el) => el.scrollTop = 0")
            last_scroll_top = -1
            no_change_count = 0
            
            while mask_iteration < 100: # 安全限制
                mask_iteration += 1
                
                # 查找可见 Mask
                visible_masks = page.locator("div[class*='DiscoverStep-']").filter(has=page.locator("div[class*='DiscoverStepMasking']")).all()
                
                for m_step in visible_masks:
                    try:
                        style = m_step.get_attribute("style")
                        y_pos = parse_transform_y(style)
                        if y_pos == -1: continue
                        
                        if y_pos in processed_masks_y:
                            continue
                            
                        # 发现新 Mask!
                        print(f"      处理 Mask (Y={y_pos})")
                        
                        # 1. 点击展开
                        mask_btn = m_step.locator("div[class*='DiscoverStepMasking']").first
                        mask_btn.click()
                        time.sleep(2) # 等待展开动画
                        
                        # 2. 抓取详情
                        mask_info = {}
                        mask_info["type"] = "masking"
                        
                        # 截取面板截图
                        try:
                            m_panel_path = os.path.join(mask_dir, f"mask_panel_{int(y_pos)}.png")
                            m_step.screenshot(path=m_panel_path)
                            mask_info["mask_panel_screenshot"] = m_panel_path
                        except: pass
                        
                        # 截取缩略图
                        canvas = m_step.locator("canvas").first
                        if canvas.count() > 0:
                            c_path = os.path.join(mask_dir, f"mask_thumb_{int(y_pos)}.png")
                            try:
                                canvas.screenshot(path=c_path)
                                mask_info["mask_thumbnail"] = c_path
                            except: pass
                        
                        # 提取参数
                        sub_params = m_step.locator("div[class*='DiscoverStepMaskingParameter']").all()
                        mask_params = []
                        for sp in sub_params:
                            mp = {}
                            rows_text_divs = sp.locator("div[class*='DiscoverStepParameterRow']").all()
                            if len(rows_text_divs) >= 2:
                                mp["label"] = rows_text_divs[0].inner_text().strip()
                                mp["value"] = rows_text_divs[-1].inner_text().strip()
                            elif len(rows_text_divs) == 1:
                                mp["label"] = rows_text_divs[0].inner_text().strip()
                            if mp: mask_params.append(mp)
                        mask_info["masking_parameters"] = mask_params
                        
                        # 更新收集的数据
                        collected_steps[y_pos] = mask_info
                        processed_masks_y.add(y_pos)
                        
                        # 3. 关闭
                        mask_btn.click()
                        time.sleep(1) # 等待折叠
                        
                    except Exception as e:
                        print(f"      Mask 处理失败: {e}")
                
                # 向下滚动
                try:
                    scroll_container.evaluate(f"(el) => el.scrollBy(0, 200)")
                    time.sleep(1.5) # 等待渲染 (延长以防止漏抓)
                    
                    curr = scroll_container.evaluate("(el) => el.scrollTop")
                    if curr == last_scroll_top:
                        no_change_count += 1
                        if no_change_count >= 3: break
                    else:
                        no_change_count = 0
                        last_scroll_top = curr
                except: break

    else:
        print("    未找到滚动容器")
        # 兜底简单的抓取...
        pass
    
    # 按 Y 位置排序
    sorted_y = sorted(collected_steps.keys())
    step_list = [collected_steps[y] for y in sorted_y]
    
    print(f"    共抓取到 {len(step_list)} 个步骤")
            
    extracted_data["details"] = step_list
    extracted_data["step_count"] = len(step_list)
    
    # 保存 JSON
    with open(os.path.join(save_dir, "details.json"), 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)
        
    return extracted_data

def extract_filmstrip_image_id(item):
    """
    从底片条项目中尽量提取当前版本对应的图片 ID。
    """
    try:
        href = item.get_attribute("href")
        if href:
            match = re.search(r"/learn/(?:remix|discover)/([a-f0-9-]+)", href)
            if match:
                return match.group(1)

        a_tag = item.locator("a[href*='/learn/']").first
        if a_tag.count() > 0:
            href = a_tag.get_attribute("href")
            if href:
                match = re.search(r"/learn/(?:remix|discover)/([a-f0-9-]+)", href)
                if match:
                    return match.group(1)

        thumb = item.locator("img").first
        if thumb.count() > 0:
            src = thumb.get_attribute("src")
            if src:
                match = re.search(r"rendition/([a-f0-9-]+)", src)
                if match:
                    return match.group(1)
    except Exception:
        pass

    return None

def download_current_visible_image(page, save_path, specific_image_id=None):
    """
    下载当前页面正在显示的图片。
    优先走 Adobe 高清接口；失败时退回当前画布/img 的直接提取。
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        try:
            os.remove(save_path)
        except Exception:
            pass

    image_id = specific_image_id or get_image_id_from_url(page.url)
    direct_download_success = False
    if image_id:
        print(f"    [图片下载] 尝试通过图片 ID 下载: {image_id}")
        direct_download_success = download_high_res_image(page, save_path, specific_image_id=image_id)

    temp_capture_path = os.path.join(os.path.dirname(save_path), "__temp_current_view.png")
    if os.path.exists(temp_capture_path):
        try:
            os.remove(temp_capture_path)
        except Exception:
            pass

    try:
        canvas = page.locator("canvas, .Loupe-image, img[class*='Loupe-image']").first
        if canvas.count() > 0:
            try:
                canvas.click()
                time.sleep(1.5)
            except Exception:
                pass

            tag_name = canvas.evaluate("el => el.tagName.toLowerCase()")

            if tag_name == "img":
                src = canvas.get_attribute("src")
                if src:
                    print("    [图片下载] 主接口不足时，尝试直接下载当前 img src")
                    download_high_res_image(page, temp_capture_path, direct_url=src)

            elif tag_name == "canvas":
                data_url = canvas.evaluate("""(canvas) => {
                    try {
                        return canvas.toDataURL('image/png', 1.0);
                    } catch(e) {
                        return null;
                    }
                }""")

                if data_url and data_url.startswith("data:image"):
                    import base64

                    header, encoded = data_url.split(",", 1)
                    data = base64.b64decode(encoded)
                    with open(temp_capture_path, "wb") as f:
                        f.write(data)
                    auto_rotate_image(temp_capture_path)
                    print(f"    [图片下载] 通过 Canvas 提取到回退图片: {len(data)/1024/1024:.2f} MB")
    except Exception as e:
        print(f"    [图片下载] 当前画面提取失败: {e}")

    main_exists = os.path.exists(save_path) and os.path.getsize(save_path) > 10
    temp_exists = os.path.exists(temp_capture_path) and os.path.getsize(temp_capture_path) > 10

    if temp_exists and not main_exists:
        os.replace(temp_capture_path, save_path)
        main_exists = True
        print("    [图片下载] 主接口失败，改用当前画面提取结果")
    elif temp_exists and main_exists:
        save_size = os.path.getsize(save_path)
        temp_size = os.path.getsize(temp_capture_path)
        if temp_size > save_size:
            os.replace(temp_capture_path, save_path)
            print("    [图片下载] 当前画面提取结果更大，已替换主下载文件")
        else:
            os.remove(temp_capture_path)
    elif os.path.exists(temp_capture_path):
        try:
            os.remove(temp_capture_path)
        except Exception:
            pass

    return os.path.exists(save_path) and os.path.getsize(save_path) > 10

def process_image_page(page, base_output_dir):
    """
    处理单个图片详情页
    """
    print(f"\n正在处理图片页面: {get_image_id_from_url(page.url)}")

    try:
        print("  正在检查主图 Mask 信息...")
        wait_for_edits_loaded(page)
        if scan_for_mask(page):
            print("  [跳过] 主图检测到 Mask，整组 URL 不再抓取")
            return None
        print("  [检测通过] 主图未发现 Mask，开始抓取主图与版本图")
    except Exception as e:
        print(f"  Mask 检测过程异常 (按无 Mask 继续): {e}")

    # 1. 创建图片目录
    current_image_id = get_image_id_from_url(page.url)
    if not current_image_id:
        current_image_id = f"unknown_{int(time.time())}"
        
    image_base_dir = os.path.join(base_output_dir, current_image_id)
    if not os.path.exists(image_base_dir):
        os.makedirs(image_base_dir)
        
    # 2. 尝试获取底片条 (Filmstrip)
    filmstrip_items = []

    # 策略 A: 官方/已知类名 (Filmstrip, Carousel)
    try:
        # 等待任意可能的底片条容器
        page.wait_for_selector(
            ".Filmstrip-filmstrip, .Filmstrip-container, div[class*='Filmstrip'], div[class*='Carousel'], #ze-feed-hscroller", 
            timeout=5000
        )
    except: pass

    # 定义可能的项目选择器列表 (优先级从高到低)
    item_selectors = [
        "#ze-feed-hscroller a[href*='/learn/']",
        "div[class*='FilmstripView'] a[href*='/learn/']",
        ".Filmstrip-filmstrip .Filmstrip-item", 
        ".Filmstrip-card",
        "div[class*='Filmstrip'] div[role='listitem']",
        "div[class*='Filmstrip'] img",
        "div[class*='Carousel'] div[class*='item']",
        "div[class*='filmstrip'] > div",
        # 通用结构：水平滚动容器下的直接子元素
        "div[style*='overflow-x: auto'] > div",
        "div[style*='overflow-x: scroll'] > div"
    ]

    for selector in item_selectors:
        try:
            items = page.locator(selector).all()
            if len(items) > 1:
                filmstrip_items = items
                print(f"      命中底片条选择器: {selector} (数量: {len(items)})")
                break
        except: pass

    if MAX_VERSIONS_PER_IMAGE is not None and len(filmstrip_items) > MAX_VERSIONS_PER_IMAGE:
        print(f"  [限制] 检测到 {len(filmstrip_items)} 个版本，将仅抓取前 {MAX_VERSIONS_PER_IMAGE} 个")
        filmstrip_items = filmstrip_items[:MAX_VERSIONS_PER_IMAGE]

    # [修改] 由于已切换到 Canvas 提取策略，不再需要清理 Blob
    # 注释掉或删除 Blob 清理代码

    # [新增] 拦截请求并注入 CORS Header，解决 Canvas Tainted 问题
    def add_cors_headers(route):
        try:
            response = route.fetch()
            headers = response.headers
            headers["Access-Control-Allow-Origin"] = "*"
            route.fulfill(response=response, headers=headers)
        except:
            route.continue_()

    # 仅对图片请求应用 CORS 修复 (宽泛匹配以防漏掉)
    try:
        page.route("**/*.jpg", add_cors_headers)
        page.route("**/*.png", add_cors_headers)
        page.route("**/*.webp", add_cors_headers)
        page.route("**/rendition/*", add_cors_headers) # Adobe API
    except: pass

    # 3. 先保存主图。version_0 固定表示主图，便于与后续 version_1~3 区分。
    main_dir = os.path.join(image_base_dir, "version_0")
    if os.path.exists(main_dir):
        try:
            shutil.rmtree(main_dir)
        except Exception:
            pass
    os.makedirs(main_dir, exist_ok=True)

    print("  开始保存主图 (version_0)...")
    main_img_path = os.path.join(main_dir, "main_image.png")
    download_current_visible_image(page, main_img_path, specific_image_id=current_image_id)
    scrape_parameters_only(page, main_dir, skip_mask_details=False)
    capture_original_image(page, image_base_dir)

    # 4. 再从底片条中收集最多 3 个无 Mask 版本
    if len(filmstrip_items) == 0:
        print("  未发现底片条，本组仅保存主图")
        return current_image_id

    print(f"  发现 {len(filmstrip_items)} 个底片条候选，开始筛选无 Mask 版本")
    filmstrip_ids = {}
    for i, item in enumerate(filmstrip_items):
        try:
            item.scroll_into_view_if_needed()
            time.sleep(0.1)
        except Exception:
            pass
        filmstrip_ids[i] = extract_filmstrip_image_id(item)

    accepted_versions = 0
    seen_image_ids = {current_image_id}

    for i, item in enumerate(filmstrip_items):
        if accepted_versions >= MAX_VALID_VERSIONS_TO_KEEP:
            break

        print(f"\n  --- 检查版本候选 {i+1}/{len(filmstrip_items)} ---")
        candidate_id = filmstrip_ids.get(i)
        if candidate_id and candidate_id in seen_image_ids:
            print(f"    [跳过] 候选与主图或已保存版本重复: {candidate_id}")
            continue

        try:
            close_blue_tooltip(page)
            item.scroll_into_view_if_needed()
            time.sleep(0.5)
            item.click(force=True)
            time.sleep(1)
        except Exception as e:
            print(f"    [跳过] 点击候选失败: {e}")
            continue

        try:
            wait_for_edits_loaded(page)
        except Exception as e:
            print(f"    [跳过] 候选编辑数据加载失败: {e}")
            continue

        current_version_id = candidate_id or get_image_id_from_url(page.url)
        if current_version_id and current_version_id in seen_image_ids:
            print(f"    [跳过] 当前候选实际与已保存版本重复: {current_version_id}")
            continue

        version_has_mask = scan_for_mask(page)
        if version_has_mask:
            print("    [跳过] 当前版本包含 Mask，不保存")
            continue

        accepted_versions += 1
        version_dir = os.path.join(image_base_dir, f"version_{accepted_versions}")
        if os.path.exists(version_dir):
            try:
                shutil.rmtree(version_dir)
            except Exception:
                pass
        os.makedirs(version_dir, exist_ok=True)

        version_image_path = os.path.join(version_dir, "main_image.png")
        print(f"    [保留] 保存无 Mask 版本到 version_{accepted_versions}")
        download_current_visible_image(page, version_image_path, specific_image_id=current_version_id)
        scrape_parameters_only(page, version_dir, skip_mask_details=False)

        if current_version_id:
            seen_image_ids.add(current_version_id)

    print(f"\n  本组完成: 主图 1 张，额外无 Mask 版本 {accepted_versions} 张")

    return current_image_id

def get_image_id_from_url(url):
    match = re.search(r"discover/([a-f0-9-]+)", url)
    if match:
        return match.group(1)
    return None

def scrape_parameters_only(page, save_dir, skip_mask_details=False):
    """
    仅抓取参数和Mask，不处理主图下载
    """
    # 这里复用原 scrape_current_view 的逻辑
    scrape_current_view(page, save_dir, skip_mask_details=skip_mask_details)


def return_to_feed(page):
    """
    返回主界面
    """
    print("  返回主界面...")
    try:
        # 尝试返回按钮
        back_btn = page.locator("button:has-text('返回'), .BackButton-s8cNGtLhVIkkRO8B, .BackButton-QrjojEcXOoL4yXx7").first
        if back_btn.count() > 0 and back_btn.is_visible():
            back_btn.click()
            time.sleep(5)
            if "/learn/discover" in page.url and "/learn/discover/" not in page.url:
                return True
                
        # 尝试社区按钮
        comm_btn = page.locator(".UnauthSidebarItem-hbDpXGDRpcGJz1h0", has_text="社区").first
        if comm_btn.count() > 0:
            comm_btn.click()
            time.sleep(5)
            if "/learn/discover" in page.url:
                return True
                
        # 直接跳转
        page.goto("https://lightroom.adobe.com/learn/discover", wait_until="domcontentloaded")
        time.sleep(5)
        return True
    except:
        page.goto("https://lightroom.adobe.com/learn/discover", wait_until="domcontentloaded")
        return True

def create_new_page(browser):
    """创建新的 context 和 page"""
    # [回调] 为了保证首页列表能正常滚动触发懒加载，视口恢复为 1920x1080
    # 但是保留 device_scale_factor=2 (Retina)，这样依然能获得接近 4K 清晰度的原图
    context = browser.new_context(
        viewport={"width": 1920, "height": 1080},
        device_scale_factor=2
    )
    page = context.new_page()
    
    return context, page

def run():
    with sync_playwright() as p:
        # 启动浏览器 (有头模式)
        browser = p.chromium.launch(headless=False, args=["--disable-gpu", "--no-sandbox"])
        
        context, page = create_new_page(browser)
        
        base_output_dir = "scraped_data_v4"
        if not os.path.exists(base_output_dir):
            os.makedirs(base_output_dir)
            
        # --- 阶段 1: 收集 URL (只在主页操作一次) ---
        feed_url = "https://lightroom.adobe.com/learn/discover"
        print(f"导航到主页: {feed_url}")
        page.goto(feed_url, wait_until="domcontentloaded")
        time.sleep(5) # 等待初始加载
        
        # [新增] 尝试处理 Cookie 弹窗或各种遮挡
        print("检查并关闭可能遮挡的弹窗...")
        try:
            # Adobe 常见的 Cookie 接受按钮
            cookie_btn = page.locator("button:has-text('Accept'), button:has-text('接受'), #onetrust-accept-btn-handler").first
            if cookie_btn.count() > 0 and cookie_btn.is_visible():
                cookie_btn.click()
                time.sleep(1)
        except: pass

        # 找出最大索引，确定需要滚动多远
        max_target_idx = max(TARGET_INDICES) if TARGET_INDICES else 0
        print(f"目标最大索引: {max_target_idx}")
        
        discovered_urls = []
        seen_urls = set()
        
        # 匹配包含 /learn/discover/ 的链接
        link_selector = "a[href*='/learn/discover/']"
        
        print(f"开始扫描 URL (目标最大索引: {max_target_idx})...")
        
        scroll_attempts = 0
        max_scrolls = 10000 # [修改] 实际上限由目标数量或页面底部决定
        consecutive_no_new = 0
        
        while len(discovered_urls) <= max_target_idx and scroll_attempts < max_scrolls:
            # 1. 提取当前可见的链接
            try:
                elements = page.locator(link_selector).all()
                new_count = 0
                
                for el in elements:
                    try:
                        href = el.get_attribute("href")
                        if href:
                            # [过滤] 排除分类标签链接 (包含 sm: 或非 UUID 格式)
                            if "sm:" in href: 
                                continue
                                
                            if not re.search(r"[0-9a-f]{8}-", href) and not re.search(r"[0-9a-f]{32}", href):
                                continue

                            # 补全 URL
                            if not href.startswith("http"):
                                full_url = "https://lightroom.adobe.com" + href
                            else:
                                full_url = href
                                
                            # 去重
                            if full_url not in seen_urls:
                                seen_urls.add(full_url)
                                discovered_urls.append(full_url)
                                new_count += 1
                    except: pass
                
                print(f"  [滚动 {scroll_attempts}] 新发现 {new_count} 个, 总计 {len(discovered_urls)} 个 (目标 > {max_target_idx})")
                
                if new_count == 0:
                    consecutive_no_new += 1
                else:
                    consecutive_no_new = 0
                    
                if len(discovered_urls) > max_target_idx:
                    break
                
                if consecutive_no_new > 20: # [修改] 增加容忍度
                    print("  [提示] 连续多次未发现新内容，判断已到底部")
                    break
            except Exception as e:
                print(f"  [扫描错误] {e}")

            # 2. 向下滚动
            try:
                # [回调] 恢复原版滚动逻辑，配合 1080P 视口使用
                # 尝试查找主要的滚动容器
                feed_container = page.locator("div[class*='DiscoverFeed'], div[class*='VirtualList']").first
                if feed_container.count() > 0:
                     feed_container.evaluate("el => el.scrollTop = el.scrollHeight")
                else:
                     page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                
                # 模拟鼠标滚轮，更真实
                page.mouse.move(500, 500)
                for _ in range(3):
                    page.mouse.wheel(0, 3000)
                    time.sleep(0.2)
            except: pass
            
            time.sleep(1.5) # 等待加载
            scroll_attempts += 1
            
        # 收集完成
        target_urls = []
        for idx in TARGET_INDICES:
             if idx < len(discovered_urls):
                 target_urls.append(discovered_urls[idx])
                 print(f"  [收集成功] 索引 {idx}: {discovered_urls[idx]}")
             else:
                 print(f"  [警告] 索引 {idx} 超出实际获取范围 ({len(discovered_urls)})")
        
        print(f"\n--- URL 收集完成，共 {len(target_urls)} 个任务 ---\n")
        
        # --- 阶段 2: 直连抓取 (无需返回主页) ---
        for i, url in enumerate(target_urls):
            # 每抓取 5 张图片，强制硬刷新一次 context 和 page，防止内存泄漏和状态污染
            if i > 0 and i % 5 == 0:
                print(f"\n[硬刷新] 已处理 {i} 张图片，正在重启浏览器 Context 清理缓存...")
                try:
                    page.close()
                    context.close()
                except: pass
                time.sleep(2)
                context, page = create_new_page(browser)
                
            print(f"\n>>> 正在处理第 {i+1}/{len(target_urls)} 个任务: {url}")
            try:
                # 直接跳转到详情页
                page.goto(url, wait_until="domcontentloaded")
                
                # 调用处理函数 (注意: process_image_page 内部会处理等待和抓取)
                process_image_page(page, base_output_dir)
                
            except Exception as e:
                print(f"  任务处理失败: {e}")
            
            # 任务间隔，防风控
            time.sleep(2)
                
        print("\n所有任务完成")
        browser.close()

if __name__ == "__main__":
    run()
