# 颠覆了传统的水果忍者游戏

这是一个**基于 Pygame、Mediapipe 和 OpenCV**实现的**PC 端体感游戏**。玩家通过**手势控制**切割屏幕上的水果获取分数。在传统切水果玩法的基础上，创新地引入了**表情识别（笑脸/哭脸）**切换游戏模式、**Boss 战**以及**排行榜系统**，使交互更加有趣和好玩。

## 目录

- [Fruit Ninja 游戏](#fruit-ninja-游戏)
  - [目录](#目录)
  - [安装](#安装)
  - [运行游戏](#运行游戏)
  - [游戏说明](#游戏说明)
  - [文件目录结构](#文件目录结构)
  - [运行环境说明](#运行环境说明)
  - [作者](#作者)

## 安装

1. 创建虚拟环境并激活:

   ```
   conda create -n FruitNinja
   conda activate FruitNinja
   ```

2. 安装环境依赖

   ```
   pip install -r requirements.txt
   ```

3. 使用 VScode 打开本项目文件夹

4. 选择创建好的虚拟环境 FruitNinja

## 运行游戏

在运行代码之前请确保你已经连接了一个网络摄像头并正确安装了所有依赖项。

```
python main.py
```

## 游戏说明

**整体流程：** 输入用户名 → 表情识别选择模式 → 倒计时开始 → 切割水果 → 通关后挑战 Boss → 查看排行榜。

- **目标：** 通过手势在屏幕上切割水果获取分数，尽快通关。
- **操作方式：** 将手放在摄像头前，移动**食指**进行切割（支持双手，左右手轨迹颜色不同）。

- **进入游戏（表情识别）：**
  - 保持**微笑** → 进入**普通模式**（通关分数 30，会出现炸弹）。
  - 摆出**哭脸** → 进入**水果风暴**（通关分数 150，水果生成更快更多，无炸弹）。

- **游戏模式：**
  - **普通模式：** 每次生成 2~4 个水果，随机出现炸弹，切到炸弹扣 3 分。
  - **水果风暴：** 每次生成 4~8 个水果，节奏更快，不出现炸弹。

- **Boss 战：**
  - 普通模式或水果风暴在 **60 秒内通关**后，可在结算界面点击「挑战Boss」进入。
  - Boss 为「**奶龙**」，需要在 **60 秒内切割 60 次**才能击败；半血后 Boss 会加速移动。
  - Boss 会不断抛出炸弹，切到炸弹扣 3 分，需要及时躲避。

- **排行榜：**
  - 游戏启动后需先输入**用户名**，成绩会记录到本地 `leaderboard.json`。
  - 排行榜按两条路线分别排名：「**普通模式→Boss战**」和「**水果风暴→Boss战**」，以**路线总用时**（前置模式用时 + Boss 战用时）排序，同一玩家同一路线只保留最佳成绩。

- **结束与重开：** 结算界面点击「再来一局」重新开始，点击「退出游戏」关闭程序。

## 文件目录结构

```
水果忍者/
│
├── image/                          
│   ├── apple.png                   
│   ├── apple-1.png / apple-2.png   
│   ├── banana.png                  
│   ├── banana-1.png / banana-2.png 
│   ├── peach.png                   
│   ├── peach-1.png / peach-2.png   
│   ├── strawberry.png              
│   ├── strawberry-1.png            
│   ├── strawberry-2.png            
│   ├── watermelon.png             
│   ├── watermelon-1.png           
│   ├── watermelon-2.png            
│   ├── bomb.png                  
│   ├── explosion.png               
│   ├── logo.png                
│   ├── title.png                 
│   ├── nailong.png                
│   ├── nailong2.png               
│   └── nailong3.png             
│
├── sound/                       
│   ├── bg.ogg                   
│   ├── startbgm.ogg                
│   ├── boss_bgm.ogg               
│   ├── cut.mp3                   
│   └── bomb.mp3                   
│
├── main.py
├── requirements.txt
├── xiangjao.ttf
├── .gitignore
├── .gitattributes
├── leaderboard.json               # 首次运行后自动生成，不入库
└── README.md
```

## 运行环境说明

- 实测通过的组合：**Python 3.13** + `requirements.txt` 中锁定的依赖版本（Windows 10/11）。
- `mediapipe 0.10.11` 没有 Python 3.13 的安装包，**不要把依赖降级回旧版本号**，否则 `pip install` 会直接失败。
- `leaderboard.json` 是运行时自动生成的本地成绩文件，已加入 `.gitignore`，不会提交到仓库；首次运行后自动创建。
- 依赖使用 `opencv-contrib-python`，**不要同时安装 `opencv-python`**，两个包共存可能导致 `cv2` 导入异常。
- 需要可用的网络摄像头，否则无法进行手势与表情识别。

「素材仅供学习交流，版权归原作者」
