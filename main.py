"""
Fruit Ninja — Expression Mode (Optimized)
基于 Pygame + MediaPipe + OpenCV 的手势控制水果忍者游戏

优化内容:
- 旋转缓存: 预渲染水果旋转帧，避免每帧调用 rotozoom
- 帧转换抽离: camera_frame_to_surface() 消除重复代码
- 修复: 重复的 pygame.init() / boss 分数显示
- 存储图片索引: Fruit 类记录 image_index，避免 O(n) 查找
- 轨迹表面缓存: 复用透明表面，避免每帧创建
- 常量集中管理: 提取魔法数字到配置区
"""

import pygame
import cv2
import mediapipe as mp
import random
import sys
import numpy as np
import math
import json
import os
from datetime import datetime

# ============================================================
# 配置常量
# ============================================================
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720
FPS = 30
GRAVITY = 0.7
AIR_RESISTANCE = 0.05

# 游戏参数
TARGET_SCORE = 30          # 普通模式通关分数
STORM_TARGET_SCORE = 150    # 水果风暴通关分数
STORM_SPAWN_INTERVAL = 20   # 风暴模式生成间隔（帧）
STORM_MIN_FRUITS = 4
STORM_MAX_FRUITS = 8
NORMAL_SPAWN_INTERVAL = 50  # 普通模式生成间隔（帧）
NORMAL_MIN_FRUITS = 2
NORMAL_MAX_FRUITS = 4
BOMB_SPAWN_CHANCE = 0.2     # 生成炸弹概率

# Boss 战参数
BOSS_TOTAL_NEEDED = 60      # 击败 Boss 所需分数
BOSS_TIME_LIMIT = 60        # Boss 战时间限制（秒）
BOSS_CUT_COOLDOWN = 10      # 切割冷却帧数
BOSS_BOMB_SPAWN_MIN = 60    # 炸弹生成间隔最小值（帧）
BOSS_BOMB_SPAWN_MAX = 90  # 炸弹生成间隔最大值（帧）
BOSS_EXPLOSION_DURATION = 60  # 爆炸动画持续帧数
BOSS_PENALTY = 3            # 切到炸弹扣分

# 手部检测
MAX_TRAIL_LENGTH = 5        # 轨迹点保留数量
MIN_MOVE_DISTANCE = 15      # 最小移动距离（像素），低于此值不算滑动

# 排行榜 (两条路线分开排名：普通模式+Boss战 / 水果风暴+Boss战)
LEADERBOARD_FILE = 'leaderboard.json'

# 表情检测阈值
SMILE_THRESH = 0.02
SAD_CORNER_THRESH = -0.015
SAD_BROW_RATIO = 0.85

# 旋转缓存: 预渲染 N 个角度（每 360/N 度一个）
ROTATION_CACHE_ANGLES = 72  # 每 5° 一个，视觉平滑且内存可控

# 全局旋转缓存（模块级，同一图片在所有实例间共享）
_shared_rotation_cache = {}  # key: id(image), value: {angle_idx: rotated_surface}


def get_rotated_image(image, angle, num_angles=ROTATION_CACHE_ANGLES):
    """从共享缓存获取旋转后的图片。首次访问时自动构建缓存。"""
    img_id = id(image)
    if img_id not in _shared_rotation_cache:
        # 构建该图片的旋转缓存
        cache = {}
        for i in range(num_angles):
            ang = i * (360 / num_angles)
            cache[i] = pygame.transform.rotozoom(image, ang, 1)
        _shared_rotation_cache[img_id] = cache

    cache = _shared_rotation_cache[img_id]
    normalized = angle % 360
    index = round(normalized / (360 / num_angles)) % num_angles
    return cache[index]


# ============================================================
# 初始化 pygame（pre_init 必须在 init 之前）
# ============================================================
pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Fruit Ninja – Expression Mode")

# ============================================================
# 摄像头
# ============================================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    sys.exit()


# ============================================================
# 工具函数
# ============================================================

def camera_frame_to_surface(frame, target_width=SCREEN_WIDTH, target_height=SCREEN_HEIGHT):
    """将 OpenCV BGR 摄像头帧转换为 pygame Surface。

    统一处理: BGR→RGB → surfarray → rotate(-90°) → flip(水平) → scale
    所有游戏场景共用此函数，消除重复代码。
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_surface = pygame.surfarray.make_surface(frame_rgb)
    frame_surface = pygame.transform.rotate(frame_surface, -90)
    frame_surface = pygame.transform.flip(frame_surface, True, False)
    if frame_surface.get_size() != (target_width, target_height):
        frame_surface = pygame.transform.scale(frame_surface, (target_width, target_height))
    return frame_surface


def draw_smooth_line(screen, points, color, width, trail_surface):
    """在预分配的 trail_surface 上绘制平滑轨迹线，然后 blit 到 screen。"""
    if len(points) < 2:
        return
    trail_surface.fill((0, 0, 0, 0))  # 清空透明表面
    for i in range(len(points) - 1):
        pygame.draw.line(trail_surface, color, points[i], points[i + 1], width)
    screen.blit(trail_surface, (0, 0))


def resize_and_top_align_image(img, window_width, window_height):
    """将标题图片缩放并顶部居中放置。"""
    img_height, img_width = img.shape[:2]
    scale = 0.3
    new_width = int(img_width * scale)
    new_height = int(img_height * scale)
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    x_offset = (window_width - new_width) // 2 - 150
    y_offset = 0

    if x_offset < 0:
        x_offset = 0
    if x_offset + new_width > window_width:
        new_width = window_width - x_offset
        resized_img = resized_img[:, :new_width]
    if y_offset + new_height > window_height:
        new_height = window_height - y_offset
        resized_img = resized_img[:new_height, :]

    background = np.zeros((window_height, window_width, 4), dtype=np.uint8)
    background[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_img
    return background, (x_offset, y_offset, new_width, new_height)


def overlay_image_alpha(background, overlay, x, y):
    """将带透明通道的 overlay 叠加到 background 上。"""
    h, w = overlay.shape[:2]
    bg_h, bg_w = background.shape[:2]

    if y >= bg_h or x >= bg_w:
        return background
    if y + h > bg_h:
        h = bg_h - y
        overlay = overlay[:h, :]
    if x + w > bg_w:
        w = bg_w - x
        overlay = overlay[:, :w]

    roi = background[y:y + h, x:x + w]
    b, g, r, a = cv2.split(overlay)
    overlay_color = cv2.merge((b, g, r))
    mask = a / 255.0

    for c in range(3):
        roi[:, :, c] = roi[:, :, c] * (1 - mask) + overlay_color[:, :, c] * mask

    return background


def draw_text_button(screen, text, x, y, w, h, font,
                     color=(255, 255, 255), bg_color=(0, 100, 200),
                     hover_color=(50, 150, 250)):
    """绘制带 hover 效果的文字按钮，返回其 Rect。"""
    mouse = pygame.mouse.get_pos()
    if x <= mouse[0] <= x + w and y <= mouse[1] <= y + h:
        pygame.draw.rect(screen, hover_color, (x, y, w, h))
    else:
        pygame.draw.rect(screen, bg_color, (x, y, w, h))
    text_surf = font.render(text, True, color)
    text_rect = text_surf.get_rect(center=(x + w // 2, y + h // 2))
    screen.blit(text_surf, text_rect)
    return pygame.Rect(x, y, w, h)


def load_leaderboard():
    """从 JSON 文件加载排行榜数据。返回按时间升序排列的列表（不限制条目数）。"""
    if not os.path.exists(LEADERBOARD_FILE):
        return []
    try:
        with open(LEADERBOARD_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            # 按完成时间升序排列（越短越好）
            data.sort(key=lambda x: x.get('time', float('inf')))
            return data
        return []
    except (json.JSONDecodeError, IOError):
        return []


def save_leaderboard_entry(name, route_mode, time_seconds):
    """向排行榜添加一条新记录并保存。返回更新后的排行榜列表。

    route_mode: '普通模式+Boss战' 或 '水果风暴+Boss战'
    time_seconds: 两条路线各自模式用时累加的总时间

    同一玩家 + 同一路线只保留最佳（最短用时）成绩，新成绩更优则替换旧成绩。
    """
    leaderboard = load_leaderboard()

    # 查找同一玩家同一路线是否已有记录
    existing_idx = None
    for i, entry in enumerate(leaderboard):
        if entry.get('name') == name and entry.get('mode') == route_mode:
            existing_idx = i
            break

    if existing_idx is not None:
        # 只有新成绩更好（用时更短）才替换
        if time_seconds < leaderboard[existing_idx]['time']:
            leaderboard[existing_idx]['time'] = time_seconds
            leaderboard[existing_idx]['date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # 否则不保存，直接返回
    else:
        entry = {
            'name': name,
            'mode': route_mode,
            'time': time_seconds,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        leaderboard.append(entry)

    leaderboard.sort(key=lambda x: x['time'])
    try:
        with open(LEADERBOARD_FILE, 'w', encoding='utf-8') as f:
            json.dump(leaderboard, f, ensure_ascii=False, indent=2)
    except IOError:
        pass
    return leaderboard


# ============================================================
# 表情检测类（MediaPipe Face Mesh）
# ============================================================
class ExpressionDetector:
    def __init__(self, max_num_faces=1):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # 关键点索引
        self.LIP_LEFT = 61
        self.LIP_RIGHT = 291
        self.LIP_TOP = 13
        self.LIP_BOTTOM = 14
        self.LEFT_EYEBROW_IN = 107
        self.RIGHT_EYEBROW_IN = 336
        self.LEFT_EYEBROW_OUT = 105
        self.RIGHT_EYEBROW_OUT = 334

    def detect_expression(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        lm = results.multi_face_landmarks[0].landmark

        def pt(idx):
            return np.array([lm[idx].x, lm[idx].y])

        lip_left = pt(self.LIP_LEFT)
        lip_right = pt(self.LIP_RIGHT)
        lip_top = pt(self.LIP_TOP)
        lip_bottom = pt(self.LIP_BOTTOM)
        brow_left_in = pt(self.LEFT_EYEBROW_IN)
        brow_right_in = pt(self.RIGHT_EYEBROW_IN)
        brow_left_out = pt(self.LEFT_EYEBROW_OUT)
        brow_right_out = pt(self.RIGHT_EYEBROW_OUT)

        lip_center = (lip_top + lip_bottom) / 2.0
        mouth_width = np.linalg.norm(lip_right - lip_left)
        if mouth_width < 0.01:
            return 'neutral'
        left_offset_y = lip_center[1] - lip_left[1]
        right_offset_y = lip_center[1] - lip_right[1]
        avg_corner_offset = (left_offset_y + right_offset_y) / 2.0

        inner_dist = np.linalg.norm(brow_right_in - brow_left_in)
        outer_dist = np.linalg.norm(brow_right_out - brow_left_out)
        brow_ratio = inner_dist / (outer_dist + 1e-6)

        if avg_corner_offset > SMILE_THRESH:
            return 'smile'
        elif avg_corner_offset < SAD_CORNER_THRESH and brow_ratio < SAD_BROW_RATIO:
            return 'sad'
        else:
            return 'neutral'


# ============================================================
# 手部检测类（支持双手）
# ============================================================
class HandDetector:
    def __init__(self, max_num_hands=2):
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.hand_landmarks = None
        self.handedness = None

    def detect_hands(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        self.hand_landmarks = results.multi_hand_landmarks
        self.handedness = results.multi_handedness
        return self.hand_landmarks

    def draw_hands(self, frame):
        if self.hand_landmarks:
            for hand_landmarks in self.hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
                )

    def get_fingertips_by_hand(self, frame_width, frame_height):
        """返回列表，每项为 (hand_label, x, y)，hand_label 是 'Left' 或 'Right'"""
        tips = []
        if self.hand_landmarks and self.handedness:
            for idx, hand_landmarks in enumerate(self.hand_landmarks):
                label = self.handedness[idx].classification[0].label
                index_tip = hand_landmarks.landmark[
                    mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP
                ]
                x = int(index_tip.x * frame_width)
                y = int(index_tip.y * frame_height)
                tips.append((label, x, y))
        return tips


# ============================================================
# 水果类（带旋转缓存优化）
# ============================================================
class Fruit:
    def __init__(self, image, x, y, image_index=-1):
        self.image = image
        self.image_index = image_index  # 在 fruit_images 中的索引，避免 O(n) 查找
        self.rect = self.image.get_rect(center=(x, y))
        self.is_cut = False
        self.velocity_x = random.uniform(-30, 30)
        self.velocity_y = random.uniform(-60, -45)
        self.rotation_angle = 0.0
        self.rotation_speed = random.uniform(1, 3)

    def reset(self):
        self.rect.x = random.randint(100, SCREEN_WIDTH - 100)
        self.rect.y = SCREEN_HEIGHT + 100
        self.velocity_x = random.uniform(-30, 30)
        self.velocity_y = random.uniform(-60, -45)
        self.rotation_angle = 0.0
        self.rotation_speed = random.uniform(1, 3)
        self.is_cut = False

    def update(self):
        if not self.is_cut:
            self.velocity_y += GRAVITY
            self.velocity_x *= (1 - AIR_RESISTANCE)
            self.velocity_y *= (1 - AIR_RESISTANCE)
            self.rect.x += self.velocity_x
            self.rect.y += self.velocity_y
            self.rotation_angle += self.rotation_speed
            if self.rotation_angle > 360:
                self.rotation_angle -= 360
            if self.rect.y > SCREEN_HEIGHT:
                self.reset()

    def draw(self, screen):
        # 使用全局共享旋转缓存代替实时 rotozoom（大幅提升性能）
        rotated_image = get_rotated_image(self.image, self.rotation_angle)
        rotated_rect = rotated_image.get_rect(center=self.rect.center)
        screen.blit(rotated_image, rotated_rect.topleft)

    def check_cut(self, finger_positions):
        if len(finger_positions) < 2:
            return False
        big_rect = self.rect.inflate(8, 8)
        for i in range(len(finger_positions) - 1):
            if big_rect.clipline(finger_positions[i], finger_positions[i + 1]):
                return True
        return False


# ============================================================
# 炸弹类（继承水果）
# ============================================================
class Bomb(Fruit):
    def __init__(self, image, x, y):
        super().__init__(image, x, y, image_index=-1)
        self.exploded = False

    def check_cut(self, finger_positions):
        if self.exploded:
            return False
        if len(finger_positions) < 2:
            return False
        for i in range(len(finger_positions) - 1):
            if self.rect.clipline(finger_positions[i], finger_positions[i + 1]):
                self.exploded = True
                return True
        return False

    def reset(self):
        self.exploded = False
        super().reset()


# ============================================================
# 半水果（碎片）—— 带旋转缓存
# ============================================================
class HalfFruit:
    def __init__(self, image, x, y, vx, vy, angle):
        self.image = image
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.angle = float(angle)
        self._gravity = GRAVITY

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += self._gravity
        self.angle += 5

    def draw(self, screen):
        rotated_image = get_rotated_image(self.image, self.angle)
        screen.blit(rotated_image, (self.x, self.y))


# ============================================================
# Boss 榴莲类（静止，只计分）
# ============================================================
class DurianBoss:
    def __init__(self, image, x, y):
        self.image = image
        self.rect = self.image.get_rect(center=(x, y))
        self.score_per_cut = 1
        self.cut_count = 0

        self.speed = 10.0
        self.speed_factor = 1.0

        angle = random.uniform(0, 2 * math.pi)
        self.vx = self.speed * math.cos(angle)
        self.vy = self.speed * math.sin(angle)

        self.update_counter = 0
        self.direction_change_interval = 60

    def set_speed_factor(self, factor):
        """调整速度倍数（半血时加快）"""
        self.speed_factor = factor
        current_angle = math.atan2(self.vy, self.vx)
        actual_speed = self.speed * self.speed_factor
        self.vx = actual_speed * math.cos(current_angle)
        self.vy = actual_speed * math.sin(current_angle)

    def update(self, screen_width, screen_height):
        self.rect.x += self.vx
        self.rect.y += self.vy

        collided = False
        if self.rect.left <= 0:
            self.rect.left = 0
            self.vx = abs(self.vx)
            collided = True
        if self.rect.right >= screen_width:
            self.rect.right = screen_width
            self.vx = -abs(self.vx)
            collided = True
        if self.rect.top <= 0:
            self.rect.top = 0
            self.vy = abs(self.vy)
            collided = True
        if self.rect.bottom >= screen_height:
            self.rect.bottom = screen_height
            self.vy = -abs(self.vy)
            collided = True

        if collided:
            actual_speed = self.speed * self.speed_factor
            angle = math.atan2(self.vy, self.vx)
            self.vx = actual_speed * math.cos(angle)
            self.vy = actual_speed * math.sin(angle)

        self.update_counter += 1
        if self.update_counter >= self.direction_change_interval:
            self.update_counter = 0
            new_angle = random.uniform(0, 2 * math.pi)
            actual_speed = self.speed * self.speed_factor
            self.vx = actual_speed * math.cos(new_angle)
            self.vy = actual_speed * math.sin(new_angle)

    def draw(self, screen):
        screen.blit(self.image, self.rect.topleft)

    def check_cut(self, finger_positions):
        if len(finger_positions) < 2:
            return False
        for i in range(len(finger_positions) - 1):
            if self.rect.clipline(finger_positions[i], finger_positions[i + 1]):
                return True
        return False


# ============================================================
# 加载音效
# ============================================================
def load_sound(path, name="sound"):
    """安全加载音效，失败时打印提示并返回 None。"""
    try:
        return pygame.mixer.Sound(path)
    except (pygame.error, FileNotFoundError) as e:
        print(f"[警告] 无法加载音效 {name} ({path}): {e}")
        return None


def load_image(path, size=None, fallback_color=None, fallback_size=(120, 120)):
    """安全加载图片，支持缩放和降级回退。"""
    try:
        img = pygame.image.load(path).convert_alpha()
        if size:
            img = pygame.transform.scale(img, size)
        return img
    except (pygame.error, FileNotFoundError) as e:
        print(f"[警告] 无法加载图片 {path}: {e}")
        if fallback_color:
            surf = pygame.Surface(fallback_size, pygame.SRCALPHA)
            surf.fill(fallback_color)
            return surf
        return None


cut_sound = load_sound('./sound/cut.mp3', 'cut')
bomb_sound = load_sound('./sound/bomb.mp3', 'bomb')

# ============================================================
# 加载图片素材
# ============================================================
logo_image = load_image('./image/logo.png')
if logo_image is None:
    logo_image = pygame.Surface((100, 50), pygame.SRCALPHA)

# 水果图片
FRUIT_SIZE_MAP = [
    ('./image/apple.png', (150, 150)),
    ('./image/banana.png', (155, 75)),
    ('./image/peach.png', (150, 150)),
    ('./image/watermelon.png', (200, 200)),
    ('./image/strawberry.png', (160, 160)),
]
FRUIT_LEFT_SIZE_MAP = [
    ('./image/apple-1.png', (150, 150)),
    ('./image/banana-1.png', (155, 75)),
    ('./image/peach-1.png', (150, 150)),
    ('./image/watermelon-1.png', (200, 200)),
    ('./image/strawberry-1.png', (160, 160)),
]
FRUIT_RIGHT_SIZE_MAP = [
    ('./image/apple-2.png', (150, 150)),
    ('./image/banana-2.png', (155, 75)),
    ('./image/peach-2.png', (150, 150)),
    ('./image/watermelon-2.png', (200, 200)),
    ('./image/strawberry-2.png', (160, 160)),
]

fruit_images = []
fruit_left_images = []
fruit_right_images = []

for path, size in FRUIT_SIZE_MAP:
    img = load_image(path, size, fallback_color=(255, 0, 0), fallback_size=size)
    if img:
        fruit_images.append(img)

for path, size in FRUIT_LEFT_SIZE_MAP:
    img = load_image(path, size, fallback_color=(255, 100, 100), fallback_size=size)
    if img:
        fruit_left_images.append(img)

for path, size in FRUIT_RIGHT_SIZE_MAP:
    img = load_image(path, size, fallback_color=(200, 0, 0), fallback_size=size)
    if img:
        fruit_right_images.append(img)

if not fruit_images:
    print("Error: No fruit images loaded. Exiting.")
    sys.exit()

# 炸弹图片
bomb_image = load_image('./image/bomb.png', (120, 120),
                        fallback_color=(0, 0, 0, 255), fallback_size=(120, 120))
if bomb_image is None:
    bomb_image = pygame.Surface((120, 120), pygame.SRCALPHA)
    pygame.draw.circle(bomb_image, (0, 0, 0), (60, 60), 60)

# 爆炸特效
explosion_image = load_image('./image/explosion.png', (120, 120),
                             fallback_color=(255, 0, 0, 128), fallback_size=(120, 120))
if explosion_image is None:
    explosion_image = pygame.Surface((120, 120), pygame.SRCALPHA)
    pygame.draw.circle(explosion_image, (255, 0, 0, 128), (60, 60), 60)

# 榴莲图片
durian_img = load_image('./image/nailong.png', (300, 300),
                        fallback_color=(128, 0, 128, 255), fallback_size=(180, 180))
if durian_img is None:
    durian_img = pygame.Surface((180, 180), pygame.SRCALPHA)
    pygame.draw.circle(durian_img, (128, 0, 128), (90, 90), 80)

durian_half_img = load_image('./image/nailong2.png', (300, 300))
if durian_half_img is None:
    durian_half_img = durian_img.copy()
    pygame.draw.circle(durian_half_img, (255, 255, 0), (90, 90), 40, 5)

# 榴莲爆炸图
durian_explosion_img = load_image('./image/nailong3.png', (450, 450))
if durian_explosion_img is None:
    durian_explosion_img = pygame.transform.scale(explosion_image, (450, 450))

# ============================================================
# 游戏状态变量
# ============================================================
fruits = []
half_fruits = []
fruit_spawn_timer = 0

hand_detector = HandDetector()
expression_detector = ExpressionDetector()

# 双手独立轨迹列表
left_hand_trail = []
right_hand_trail = []
# 轨迹用的透明表面（缓存复用）
trail_surface_cache = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)

smile_detected = False
game_mode = None
running = True
score = 0
clock = pygame.time.Clock()

target_score = TARGET_SCORE
game_over = False
start_time = pygame.time.get_ticks()
showText = False
message_display_time = 0
message_text = None
message_rect = None
start_bgm_loaded = False
game_bgm_loaded = False
final_elapsed_time = 0
bg_music_playing = False
start_bgm_playing = False

# Boss 战相关变量
boss_fight = False
boss_score = 0
boss_total_needed = BOSS_TOTAL_NEEDED
boss_exploded = False
boss_explosion_timer = 0
boss_durian = None
boss_completed = False
boss_cut_cooldown = 0
half_health_triggered = False
boss_bombs = []
boss_bomb_spawn_timer = 0
boss_time_limit = BOSS_TIME_LIMIT
boss_time_remaining = BOSS_TIME_LIMIT
boss_start_ticks = 0
boss_failed = False
boss_completion_time = 0  # Boss 战完成用时（秒）
leaderboard_data = load_leaderboard()  # 启动时加载排行榜

# 路线追踪：记录进入 Boss 战之前完成的前置模式和时间
previous_mode_name = None   # '普通模式' 或 '水果风暴'
previous_mode_time = 0      # 前置模式用时（秒）

# 排行榜滚动偏移（鼠标滚轮控制）
lb_scroll_offset = 0
MAX_VISIBLE_ROWS = 8        # 每条路线表格最多同时显示的行数

# 用户名输入
player_name = ""
username_phase = True  # 用户名输入阶段
input_text = ""

# 手指移动检测
prev_fingertips = {}


# ============================================================
# 主循环
# ============================================================
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and username_phase:
            if event.key == pygame.K_RETURN and input_text.strip():
                player_name = input_text.strip()
                username_phase = False
            elif event.key == pygame.K_BACKSPACE:
                input_text = input_text[:-1]
            elif event.key != pygame.K_RETURN:
                # 限制用户名长度
                if len(input_text) < 12:
                    input_text += event.unicode

    # ========================================================
    # 游戏结束界面
    # ========================================================
    if game_over:
        elapsed_time = final_elapsed_time
        if bg_music_playing:
            pygame.mixer.music.pause()
            bg_music_playing = False

        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame_surface = camera_frame_to_surface(frame)
            screen.blit(frame_surface, (0, 0))

        # 半透明遮罩
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        screen.blit(overlay, (0, 0))

        font = pygame.font.Font('xiangjao.ttf', 60)
        if boss_completed:
            final_score_text = font.render(
                f'Boss战通关！ 击败奶龙！ 路线总用时: {final_elapsed_time}s',
                True, (255, 215, 0)
            )
        elif boss_failed:
            final_score_text = font.render(
                f'Boss战失败！ 未能在{BOSS_TIME_LIMIT}秒内击败奶龙！', True, (255, 100, 100)
            )
        else:
            mode_name = "普通模式" if game_mode == 'normal' else "水果风暴"
            final_score_text = font.render(
                f'{mode_name} 通关！得分: {score} 用时: {elapsed_time}s',
                True, (255, 255, 255)
            )
        final_score_rect = final_score_text.get_rect(
            center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 160)
        )
        screen.blit(final_score_text, final_score_rect)

        btn_width, btn_height = 200, 70
        gap = 40
        total_w = btn_width * 2 + gap
        start_x = (SCREEN_WIDTH - total_w) // 2
        btn_y = SCREEN_HEIGHT // 2 - 90
        btn_font = pygame.font.Font('xiangjao.ttf', 36)

        # 判断是否显示挑战Boss按钮
        show_boss_btn = False
        if not boss_fight and not boss_completed:
            if (game_mode == 'normal' and elapsed_time <= 60) or \
                    (game_mode == 'storm' and elapsed_time <= 60):
                show_boss_btn = True

        restart_rect = draw_text_button(
            screen, '再来一局', start_x, btn_y, btn_width, btn_height, btn_font,
            color=(255, 255, 255), bg_color=(0, 128, 0), hover_color=(0, 180, 0)
        )
        quit_rect = draw_text_button(
            screen, '退出游戏', start_x + btn_width + gap, btn_y, btn_width, btn_height,
            btn_font, color=(255, 255, 255), bg_color=(180, 0, 0), hover_color=(220, 40, 40)
        )

        boss_btn_rect = None
        if show_boss_btn:
            boss_btn_rect = draw_text_button(
                screen, '挑战Boss', start_x - btn_width - gap, btn_y,
                btn_width, btn_height, btn_font,
                color=(255, 255, 255), bg_color=(255, 140, 0), hover_color=(255, 100, 0)
            )

        # --- 两条路线分开的排行榜显示 ---
        lb_title_font = pygame.font.Font('xiangjao.ttf', 32)
        lb_header_font = pygame.font.Font('xiangjao.ttf', 20)
        lb_font = pygame.font.Font('xiangjao.ttf', 18)
        lb_header_y = btn_y + btn_height + 20

        # 排行榜标题
        lb_title = lb_title_font.render('🏆 排行榜', True, (255, 215, 0))
        lb_title_rect = lb_title.get_rect(center=(SCREEN_WIDTH // 2, lb_header_y))
        screen.blit(lb_title, lb_title_rect)

        # 将排行榜数据按路线分组
        route_normal = [e for e in leaderboard_data if e.get('mode') == '普通模式+Boss战']
        route_storm = [e for e in leaderboard_data if e.get('mode') == '水果风暴+Boss战']

        # 左右两栏布局
        left_col_x = 40
        right_col_x = SCREEN_WIDTH // 2 + 20
        table_width = SCREEN_WIDTH // 2 - 60
        table_start_y = lb_header_y + 35

        # 计算两条路线中较长的长度，用于限制滚动范围
        max_data_len = max(len(route_normal), len(route_storm))
        max_scroll = max(0, max_data_len - MAX_VISIBLE_ROWS)
        if lb_scroll_offset > max_scroll:
            lb_scroll_offset = max_scroll
        if lb_scroll_offset < 0:
            lb_scroll_offset = 0

        def draw_route_table(title, data, col_start_x, start_y, table_w):
            """绘制单条路线的排行榜表格（支持滚动，只渲染可见行）。"""
            # 路线标题
            route_title = lb_header_font.render(title, True, (255, 200, 100))
            screen.blit(route_title, (col_start_x, start_y))

            if not data:
                empty = lb_font.render('暂无记录', True, (120, 120, 120))
                screen.blit(empty, (col_start_x, start_y + 28))
                return

            # 表头
            header_color = (180, 180, 180)
            screen.blit(lb_font.render('排名', True, header_color), (col_start_x, start_y + 28))
            screen.blit(lb_font.render('玩家', True, header_color), (col_start_x + 40, start_y + 28))
            screen.blit(lb_font.render('用时', True, header_color), (col_start_x + 115, start_y + 28))
            screen.blit(lb_font.render('日期', True, header_color), (col_start_x + 170, start_y + 28))

            # 分隔线
            pygame.draw.line(screen, (60, 60, 60),
                             (col_start_x, start_y + 48),
                             (col_start_x + table_w, start_y + 48), 1)

            row_h = 26
            # 只渲染可见范围内的行
            visible_start = lb_scroll_offset
            visible_end = min(len(data), lb_scroll_offset + MAX_VISIBLE_ROWS)

            for i in range(visible_start, visible_end):
                entry = data[i]
                row_y = start_y + 52 + (i - visible_start) * row_h
                # 高亮前三名（使用全局排名）
                if i == 0:
                    row_color = (255, 215, 0)
                elif i == 1:
                    row_color = (192, 192, 192)
                elif i == 2:
                    row_color = (205, 133, 63)
                else:
                    row_color = (200, 200, 200)

                name = entry.get('name', '???')
                if len(name) > 6:
                    name = name[:5] + '…'

                screen.blit(lb_font.render(f'#{i + 1}', True, row_color),
                            (col_start_x, row_y))
                screen.blit(lb_font.render(name, True, (255, 255, 255)),
                            (col_start_x + 40, row_y))
                screen.blit(lb_font.render(f'{entry["time"]}s', True, (255, 255, 255)),
                            (col_start_x + 115, row_y))
                screen.blit(lb_font.render(entry.get('date', '----'), True, (140, 140, 140)),
                            (col_start_x + 170, row_y))

            # 滚动提示箭头
            arrow_color = (180, 180, 180)
            arrow_font = pygame.font.Font('xiangjao.ttf', 18)
            if lb_scroll_offset > 0:
                up_arrow = arrow_font.render('▲ 更多', True, arrow_color)
                screen.blit(up_arrow, (col_start_x + table_w - 80, start_y + 28))
            if visible_end < len(data):
                down_y = start_y + 52 + MAX_VISIBLE_ROWS * row_h
                down_arrow = arrow_font.render(f'▼ 还有{len(data) - visible_end}条', True, arrow_color)
                screen.blit(down_arrow, (col_start_x + table_w - 110, down_y))

        draw_route_table('▸ 普通模式→Boss战', route_normal,
                         left_col_x, table_start_y, table_width)
        draw_route_table('▸ 水果风暴→Boss战', route_storm,
                         right_col_x, table_start_y, table_width)

        if not route_normal and not route_storm:
            lb_empty = lb_header_font.render('暂无记录，快来挑战吧！', True, (150, 150, 150))
            lb_empty_rect = lb_empty.get_rect(
                center=(SCREEN_WIDTH // 2, table_start_y + 40)
            )
            screen.blit(lb_empty, lb_empty_rect)

        pygame.display.flip()

        # 游戏结束事件处理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEWHEEL:
                # 鼠标滚轮滚动排行榜
                lb_scroll_offset -= event.y  # event.y: 上滚=1, 下滚=-1
                # 限制滚动范围（不能小于0）
                if lb_scroll_offset < 0:
                    lb_scroll_offset = 0
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if restart_rect.collidepoint(event.pos):
                    # 重置所有状态
                    start_time = pygame.time.get_ticks()
                    game_over = False
                    score = 0
                    fruits.clear()
                    half_fruits.clear()
                    left_hand_trail.clear()
                    right_hand_trail.clear()
                    smile_detected = False
                    game_mode = None
                    boss_fight = False
                    boss_completed = False
                    boss_exploded = False
                    boss_score = 0
                    boss_durian = None
                    half_health_triggered = False
                    boss_failed = False
                    boss_completion_time = 0
                    previous_mode_name = None
                    previous_mode_time = 0
                    lb_scroll_offset = 0
                    leaderboard_data = load_leaderboard()
                    username_phase = False  # 重启不重新输入用户名
                    input_text = ""
                    pygame.mixer.music.stop()
                elif quit_rect.collidepoint(event.pos):
                    running = False
                elif show_boss_btn and boss_btn_rect and boss_btn_rect.collidepoint(event.pos):
                    # 进入 Boss 战（previous_mode_name/time 已由前置模式设置）
                    lb_scroll_offset = 0
                    game_over = False
                    boss_fight = True
                    boss_score = 0
                    boss_exploded = False
                    boss_completed = False
                    half_health_triggered = False
                    boss_failed = False
                    boss_completion_time = 0
                    boss_time_remaining = boss_time_limit
                    boss_start_ticks = pygame.time.get_ticks()
                    leaderboard_data = load_leaderboard()
                    boss_durian = DurianBoss(durian_img, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
                    fruits.clear()
                    half_fruits.clear()
                    left_hand_trail.clear()
                    right_hand_trail.clear()
                    boss_bombs.clear()
                    boss_bomb_spawn_timer = 30
                    pygame.mixer.music.stop()
                    try:
                        pygame.mixer.music.load('./sound/boss_bgm.ogg')
                        pygame.mixer.music.play(loops=-1)
                    except (pygame.error, FileNotFoundError):
                        print("[警告] Boss BGM 加载失败")
        continue

    # ========================================================
    # 读取摄像头帧
    # ========================================================
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    frame = cv2.flip(frame, 1)

    # ========================================================
    # 用户名输入界面
    # ========================================================
    if username_phase:
        frame_surface = camera_frame_to_surface(frame)
        screen.blit(frame_surface, (0, 0))

        # 半透明遮罩
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        screen.blit(overlay, (0, 0))

        # 标题
        title_font = pygame.font.Font('xiangjao.ttf', 55)
        title_text = title_font.render('请输入你的用户名', True, (255, 255, 255))
        title_rect = title_text.get_rect(
            center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 100)
        )
        screen.blit(title_text, title_rect)

        # 输入框背景
        input_box_w, input_box_h = 400, 60
        input_box_x = (SCREEN_WIDTH - input_box_w) // 2
        input_box_y = SCREEN_HEIGHT // 2 - 20
        pygame.draw.rect(screen, (50, 50, 50),
                         (input_box_x, input_box_y, input_box_w, input_box_h))
        pygame.draw.rect(screen, (255, 215, 0),
                         (input_box_x, input_box_y, input_box_w, input_box_h), 3)

        # 输入文字
        input_font = pygame.font.Font('xiangjao.ttf', 40)
        display_text = input_text + ('|' if pygame.time.get_ticks() % 800 < 400 else '')
        input_surf = input_font.render(display_text, True, (255, 255, 255))
        input_rect = input_surf.get_rect(
            center=(SCREEN_WIDTH // 2, input_box_y + input_box_h // 2)
        )
        screen.blit(input_surf, input_rect)

        # 提示文字
        tip_font = pygame.font.Font('xiangjao.ttf', 28)
        tip_text = tip_font.render('按回车键确认', True, (180, 180, 180))
        tip_rect = tip_text.get_rect(
            center=(SCREEN_WIDTH // 2, input_box_y + input_box_h + 50)
        )
        screen.blit(tip_text, tip_rect)

        pygame.display.flip()
        clock.tick(FPS)
        continue

    # ========================================================
    # 等待表情检测界面
    # ========================================================
    if not smile_detected:
        if not start_bgm_playing:
            pygame.mixer.stop()
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
            try:
                pygame.mixer.music.load('./sound/startbgm.ogg')
                pygame.mixer.music.set_volume(0.7)
                pygame.mixer.music.play(loops=-1)
            except (pygame.error, FileNotFoundError):
                print("[警告] 启动 BGM 加载失败")
            start_bgm_playing = True

        title_image = cv2.imread('./image/title.png', cv2.IMREAD_UNCHANGED)
        if title_image is not None:
            top_aligned_image, (x_offset, y_offset, new_width, new_height) = \
                resize_and_top_align_image(title_image, SCREEN_WIDTH, SCREEN_HEIGHT)
            frame_resized = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
            overlay_image_alpha(frame_resized, top_aligned_image, x_offset, y_offset)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_surface = pygame.surfarray.make_surface(frame_rgb)
            frame_surface = pygame.transform.rotate(frame_surface, -90)
            frame_surface = pygame.transform.flip(frame_surface, True, False)
            screen.blit(frame_surface, (0, 0))
        else:
            # 降级：直接显示摄像头画面
            frame_surface = camera_frame_to_surface(frame)
            screen.blit(frame_surface, (0, 0))

        font_tip = pygame.font.Font('xiangjao.ttf', 32)
        tip1 = font_tip.render("笑脸进入普通模式", True, (0, 0, 0))
        tip2 = font_tip.render("哭脸进入水果风暴", True, (0, 0, 0))
        screen.blit(tip1, (40, 450))
        screen.blit(tip2, (40, 500))

        expression = expression_detector.detect_expression(frame)
        if expression is not None:
            chinese_font = pygame.font.Font('xiangjao.ttf', 50)
            if expression == 'smile':
                smile_detected = True
                game_mode = 'normal'
                mode_text = '普通模式(60秒内完成，可解锁BOSS关卡)'
                text_color = (255, 255, 255)
                print("Smile detected! Mode: Normal")
            elif expression == 'sad':
                smile_detected = True
                game_mode = 'storm'
                mode_text = '水果风暴！(60秒内完成，可解锁BOSS关卡)'
                text_color = (255, 255, 0)
                print("Sad face detected! Mode: Fruit Storm")

            if smile_detected:
                message_text = chinese_font.render(
                    f'{mode_text} 5秒后开始', True, text_color
                )
                message_rect = message_text.get_rect(
                    center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
                )
                showText = True
                message_display_time = pygame.time.get_ticks()

    # ========================================================
    # 5秒倒计时提示
    # ========================================================
    elif showText:
        current_time = pygame.time.get_ticks()
        hand_landmarks = hand_detector.detect_hands(frame)
        hand_detector.draw_hands(frame)
        frame_surface = camera_frame_to_surface(frame)
        screen.blit(frame_surface, (0, 0))

        if current_time - message_display_time < 5000:
            if message_text:
                screen.blit(message_text, message_rect)
        else:
            showText = False
            start_time = pygame.time.get_ticks()

    # ========================================================
    # Boss 战逻辑
    # ========================================================
    elif boss_fight:
        # 检测手部并绘制
        hand_landmarks = hand_detector.detect_hands(frame)
        hand_detector.draw_hands(frame)

        # 获取指尖坐标
        fingertips = hand_detector.get_fingertips_by_hand(SCREEN_WIDTH, SCREEN_HEIGHT)
        for label, x, y in fingertips:
            if x is not None and y is not None:
                if label == 'Left':
                    left_hand_trail.append((x, y))
                else:
                    right_hand_trail.append((x, y))

        # 限制轨迹长度
        while len(left_hand_trail) > MAX_TRAIL_LENGTH:
            left_hand_trail.pop(0)
        while len(right_hand_trail) > MAX_TRAIL_LENGTH:
            right_hand_trail.pop(0)

        # --- 榴莲切割检测 ---
        if boss_durian and not boss_exploded and not boss_completed:
            if boss_cut_cooldown > 0:
                boss_cut_cooldown -= 1
            else:
                cut_occurred = False
                # 左手检测
                if len(left_hand_trail) >= 2:
                    p1, p2 = left_hand_trail[-2], left_hand_trail[-1]
                    if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) >= 5:
                        if boss_durian.check_cut(left_hand_trail):
                            cut_occurred = True
                            left_hand_trail.clear()
                # 右手检测
                if not cut_occurred and len(right_hand_trail) >= 2:
                    p1, p2 = right_hand_trail[-2], right_hand_trail[-1]
                    if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) >= 5:
                        if boss_durian.check_cut(right_hand_trail):
                            cut_occurred = True
                            right_hand_trail.clear()

                if cut_occurred:
                    boss_score += 1
                    if cut_sound:
                        cut_sound.play()
                    boss_cut_cooldown = BOSS_CUT_COOLDOWN

                    # 半血加速
                    if not half_health_triggered and boss_score >= boss_total_needed // 2:
                        half_health_triggered = True
                        boss_durian.image = durian_half_img
                        boss_durian.set_speed_factor(2.0)

                    # 击败 Boss
                    if boss_score >= boss_total_needed:
                        boss_exploded = True
                        boss_explosion_timer = BOSS_EXPLOSION_DURATION
                        if bomb_sound:
                            bomb_sound.play()
                        pygame.mixer.music.stop()

        # --- 炸弹生成与更新 ---
        if boss_durian and not boss_exploded and not boss_completed:
            if boss_bomb_spawn_timer <= 0:
                boss_bomb_spawn_timer = random.randint(BOSS_BOMB_SPAWN_MIN, BOSS_BOMB_SPAWN_MAX)
                bx, by = boss_durian.rect.center
                angle_deg = random.uniform(-60, 60)
                angle_rad = math.radians(angle_deg)
                speed = random.uniform(35, 55)
                vx = speed * math.sin(angle_rad)
                vy = -speed * math.cos(angle_rad)
                bomb = Bomb(bomb_image, bx, by)
                bomb.velocity_x = vx
                bomb.velocity_y = vy
                bomb.rect.center = (bx, by)
                boss_bombs.append(bomb)
            else:
                boss_bomb_spawn_timer -= 1

        # 更新炸弹位置
        for bomb in boss_bombs[:]:
            bomb.update()
            if (bomb.rect.y > SCREEN_HEIGHT or
                    bomb.rect.x < -100 or bomb.rect.x > SCREEN_WIDTH + 100):
                boss_bombs.remove(bomb)

        # --- 炸弹切割检测 ---
        bomb_cut_occurred = False
        for bomb in boss_bombs[:]:
            if not bomb_cut_occurred and len(left_hand_trail) >= 2:
                if bomb.check_cut(left_hand_trail):
                    bomb_cut_occurred = True
                    boss_bombs.remove(bomb)
                    boss_score = max(0, boss_score - BOSS_PENALTY)
                    if bomb_sound:
                        bomb_sound.play()
                    half_fruits.append(HalfFruit(
                        explosion_image, bomb.rect.x, bomb.rect.y, 0, -5, 0
                    ))
                    left_hand_trail.clear()
                    break

            if not bomb_cut_occurred and len(right_hand_trail) >= 2:
                if bomb.check_cut(right_hand_trail):
                    bomb_cut_occurred = True
                    boss_bombs.remove(bomb)
                    boss_score = max(0, boss_score - BOSS_PENALTY)
                    if bomb_sound:
                        bomb_sound.play()
                    half_fruits.append(HalfFruit(
                        explosion_image, bomb.rect.x, bomb.rect.y, 0, -5, 0
                    ))
                    right_hand_trail.clear()
                    break

        # --- 倒计时更新 ---
        if not boss_exploded and not boss_completed and not boss_failed:
            elapsed_seconds = (pygame.time.get_ticks() - boss_start_ticks) // 1000
            boss_time_remaining = max(0, boss_time_limit - elapsed_seconds)
            if boss_time_remaining <= 0 and boss_score < boss_total_needed:
                boss_failed = True
                game_over = True
                final_elapsed_time = 0
                pygame.mixer.music.stop()

        # --- Boss 位置更新 ---
        if boss_durian and not boss_exploded and not boss_completed:
            boss_durian.update(SCREEN_WIDTH, SCREEN_HEIGHT)

        # --- 绘制画面 ---
        frame_surface = camera_frame_to_surface(frame)
        screen.blit(frame_surface, (0, 0))

        # 绘制炸弹
        for bomb in boss_bombs:
            bomb.draw(screen)

        # 绘制碎片
        for half_fruit in half_fruits[:]:
            half_fruit.draw(screen)
            half_fruit.update()
            if half_fruit.y > SCREEN_HEIGHT:
                half_fruits.remove(half_fruit)

        # 绘制榴莲或爆炸
        if boss_durian:
            if boss_exploded:
                explosion_rect = durian_explosion_img.get_rect(
                    center=boss_durian.rect.center
                )
                screen.blit(durian_explosion_img, explosion_rect)
                boss_explosion_timer -= 1
                if boss_explosion_timer <= 0:
                    boss_fight = False
                    boss_completed = True
                    game_over = True
                    boss_completion_time = (pygame.time.get_ticks() - boss_start_ticks) // 1000
                    # 累加前置模式用时 + Boss战用时 = 路线总用时
                    route_total_time = previous_mode_time + boss_completion_time
                    route_mode_name = f'{previous_mode_name}+Boss战'
                    final_elapsed_time = route_total_time
                    leaderboard_data = save_leaderboard_entry(
                        player_name, route_mode_name, route_total_time
                    )
            else:
                boss_durian.draw(screen)

        # 绘制手指轨迹
        if len(left_hand_trail) > 1:
            draw_smooth_line(screen, left_hand_trail, (0, 255, 0, 200), 6,
                             trail_surface_cache)
        if len(right_hand_trail) > 1:
            draw_smooth_line(screen, right_hand_trail, (255, 255, 255, 200), 6,
                             trail_surface_cache)

        # 显示分数和倒计时
        font = pygame.font.Font(None, 55)
        score_text = font.render(
            f'Boss Score: {boss_score}/{boss_total_needed}', True, (255, 255, 255)
        )
        screen.blit(score_text, (10, 10))
        time_text = font.render(f'Time: {boss_time_remaining}s', True, (255, 255, 255))
        screen.blit(time_text, (SCREEN_WIDTH - 150, 10))

        # 提示文字
        tip_font = pygame.font.Font('xiangjao.ttf', 30)
        tip = tip_font.render('切奶龙！每刀+1分  小心炸弹！', True, (255, 255, 0))
        screen.blit(tip, (SCREEN_WIDTH // 2 - tip.get_width() // 2, SCREEN_HEIGHT - 50))

    # ========================================================
    # 正常游戏进行中（普通模式 / 水果风暴）
    # ========================================================
    else:
        # 切换音乐：停止启动音乐 → 播放游戏背景音乐
        if start_bgm_playing:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
            start_bgm_playing = False

        if not bg_music_playing:
            try:
                pygame.mixer.music.load('./sound/bg.ogg')
                pygame.mixer.music.set_volume(0.7)
                pygame.mixer.music.play(loops=-1)
            except (pygame.error, FileNotFoundError):
                print("[警告] 游戏背景音乐加载失败")
            bg_music_playing = True

        hand_landmarks = hand_detector.detect_hands(frame)
        hand_detector.draw_hands(frame)
        frame_surface = camera_frame_to_surface(frame)

        # --- 双手轨迹记录 ---
        fingertips = hand_detector.get_fingertips_by_hand(SCREEN_WIDTH, SCREEN_HEIGHT)
        current_fingertips = {}

        for label, x, y in fingertips:
            if x is not None and y is not None:
                current_fingertips[label] = (x, y)

                # 移动检测：手指有明显移动时才记录轨迹
                is_moving = False
                if label in prev_fingertips:
                    prev_x, prev_y = prev_fingertips[label]
                    distance = math.hypot(x - prev_x, y - prev_y)
                    if distance >= MIN_MOVE_DISTANCE:
                        is_moving = True
                else:
                    is_moving = True  # 第一帧默认允许

                if is_moving:
                    if label == 'Left':
                        left_hand_trail.append((x, y))
                    else:
                        right_hand_trail.append((x, y))

        # 限制轨迹长度
        while len(left_hand_trail) > MAX_TRAIL_LENGTH:
            left_hand_trail.pop(0)
        while len(right_hand_trail) > MAX_TRAIL_LENGTH:
            right_hand_trail.pop(0)

        prev_fingertips = current_fingertips
        all_finger_positions = left_hand_trail + right_hand_trail

        # --- 切割检测 ---
        if len(all_finger_positions) > 1:
            for fruit in fruits[:]:
                if fruit.check_cut(all_finger_positions):
                    if isinstance(fruit, Bomb):
                        score = max(0, score - BOSS_PENALTY)
                        if bomb_sound:
                            bomb_sound.play()
                        half_fruits.append(HalfFruit(
                            explosion_image, fruit.rect.x, fruit.rect.y, 0, -5, 0
                        ))
                        fruit.reset()
                    else:
                        fruit.is_cut = True
                        score += 1
                        if cut_sound:
                            cut_sound.play()
                        # 使用预存的 image_index 避免 O(n) 查找
                        idx = fruit.image_index
                        if 0 <= idx < len(fruit_left_images):
                            half_fruits.append(HalfFruit(
                                fruit_left_images[idx],
                                fruit.rect.x, fruit.rect.y,
                                fruit.velocity_x - 1, fruit.velocity_y, -45
                            ))
                            half_fruits.append(HalfFruit(
                                fruit_right_images[idx],
                                fruit.rect.x + 50, fruit.rect.y,
                                fruit.velocity_x + 1, fruit.velocity_y, 45
                            ))
                        fruit.reset()
                    # 切割后清空轨迹，防止重复切割
                    left_hand_trail.clear()
                    right_hand_trail.clear()
                    break

        # --- 更新和绘制水果 ---
        for fruit in fruits:
            fruit.update()
        for half_fruit in half_fruits[:]:
            half_fruit.update()
            if half_fruit.y > SCREEN_HEIGHT:
                half_fruits.remove(half_fruit)

        # --- 模式参数 ---
        if game_mode == 'storm':
            spawn_interval = STORM_SPAWN_INTERVAL
            min_fruits = STORM_MIN_FRUITS
            max_fruits = STORM_MAX_FRUITS
            include_bomb = False
        else:
            spawn_interval = NORMAL_SPAWN_INTERVAL
            min_fruits = NORMAL_MIN_FRUITS
            max_fruits = NORMAL_MAX_FRUITS
            include_bomb = True

        fruit_spawn_timer += 1
        if fruit_spawn_timer >= spawn_interval:
            num_fruits = random.randint(min_fruits, max_fruits)
            for _ in range(num_fruits):
                if include_bomb and random.random() < BOMB_SPAWN_CHANCE:
                    fruits.append(Bomb(
                        bomb_image,
                        random.randint(100, SCREEN_WIDTH - 100),
                        SCREEN_HEIGHT
                    ))
                else:
                    idx = random.randrange(len(fruit_images))
                    fruits.append(Fruit(
                        fruit_images[idx],
                        random.randint(100, SCREEN_WIDTH - 100),
                        SCREEN_HEIGHT,
                        image_index=idx  # 存储索引，避免后续 O(n) 查找
                    ))
            fruit_spawn_timer = 0

        # --- 绘制场景 ---
        screen.fill((255, 255, 255))
        screen.blit(frame_surface, (0, 0))
        for fruit in fruits:
            fruit.draw(screen)
        for half_fruit in half_fruits:
            half_fruit.draw(screen)

        # 左右手轨迹（不同颜色）
        if len(left_hand_trail) > 1:
            draw_smooth_line(screen, left_hand_trail, (0, 255, 0, 200), 6,
                             trail_surface_cache)
        if len(right_hand_trail) > 1:
            draw_smooth_line(screen, right_hand_trail, (255, 255, 255, 200), 6,
                             trail_surface_cache)

        # --- 显示时间、分数 ---
        elapsed_time = (pygame.time.get_ticks() - start_time) // 1000
        font = pygame.font.Font(None, 55)
        time_text = font.render(f'Time: {elapsed_time}s', True, (0, 0, 0))
        screen.blit(time_text, (SCREEN_WIDTH - 180, 10))

        # 通关判定
        if game_mode == 'storm':
            if score >= STORM_TARGET_SCORE:
                final_elapsed_time = (pygame.time.get_ticks() - start_time) // 1000
                game_over = True
                # 暂存路线信息，等 Boss 战完成后累加保存
                previous_mode_name = '水果风暴'
                previous_mode_time = final_elapsed_time
            display_target = STORM_TARGET_SCORE
        else:
            if score >= target_score:
                final_elapsed_time = (pygame.time.get_ticks() - start_time) // 1000
                game_over = True
                # 暂存路线信息，等 Boss 战完成后累加保存
                previous_mode_name = '普通模式'
                previous_mode_time = final_elapsed_time
            display_target = target_score

        score_text = font.render(f'Score: {score}/{display_target}', True, (0, 0, 0))
        screen.blit(score_text, (10, 10))
        screen.blit(logo_image,
                     (SCREEN_WIDTH // 2 - logo_image.get_width() // 2, 10))

        if game_mode == 'storm':
            storm_font = pygame.font.Font(None, 40)
            storm_text = storm_font.render('水果风暴!', True, (255, 215, 0))
            screen.blit(storm_text,
                         (SCREEN_WIDTH // 2 - storm_text.get_width() // 2, 80))

    pygame.display.flip()
    clock.tick(FPS)

# ============================================================
# 清理
# ============================================================
cap.release()
cv2.destroyAllWindows()
pygame.quit()
