# 夹爪渐进式张开/闭合功能说明

## 📋 功能概述

现在夹爪控制支持**渐进式移动**，不再是瞬间跳变，而是以可控的速度平滑张开或闭合。

---

## 🆕 新增功能

### 1. 渐进式张开（Gradual Open）

```json
{
  "type": "gripper",
  "side": "right",
  "target": "open",
  "duration": 0.5
}
```

**效果**：
- 夹爪在 **0.5秒** 内平滑张开
- 不是瞬间跳变，而是分多步完成
- 视觉上更自然、更真实

---

### 2. 渐进式闭合（Gradual Close）

```json
{
  "type": "gripper",
  "side": "right",
  "target": "close",
  "duration": 1.0,
  "delta": 0.02
}
```

**效果**：
- 夹爪在 **1.0秒** 内闭合 **2cm**
- 平滑的闭合动作
- 适合精确控制

---

## 🎛️ 参数说明

### 通用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `type` | string | - | 固定为 `"gripper"` |
| `side` | string | `"right"` | 机械臂侧：`"left"` 或 `"right"` |
| `target` | string | - | 目标动作：`"open"`, `"close"`, `"hold"` |

### 渐进控制参数

| 参数 | 类型 | 默认值 | 说明 | 优先级 |
|------|------|--------|------|--------|
| `duration` | float | `0.0` | 总时长（秒） | 高 |
| `step_size` | float | `0.005` | 每步步长（米） | 中 |
| `step_sleep` | float | `0.1` | 每步间隔（秒） | 低 |

### "open" 专用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `duration` | float | `0.0` | 张开总时长，设为 > 0 启用渐进模式 |
| `step_sleep` | float | `0.1` | 每步间隔时间 |

### "close" 专用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `duration` | float | `0.0` | 闭合总时长，设为 > 0 启用渐进模式 |
| `delta` | float | `0.001` | 闭合距离（米） |
| `step_sleep` | float | `0.1` | 每步间隔时间 |

---

## 🔀 模式切换

### 触发渐进模式的条件

**对于 "open"**:
```python
if duration > 0 or step_size < 0.03:
    # 渐进模式
else:
    # 瞬间模式（向后兼容）
```

**对于 "close"**:
```python
if duration > 0 or step_size < abs(delta):
    # 渐进模式
else:
    # 瞬间模式
```

### 向后兼容

**旧的指令格式仍然有效**：

```json
{"type": "gripper", "side": "right", "target": "open"}
```

**行为**：
- 如果不提供 `duration` 参数（或 `duration=0`）
- 并且 `step_size >= 0.03`（或使用默认值）
- 则使用**瞬间模式**（保持原有行为）

---

## 📊 使用示例

### 示例1：快速张开（0.5秒）

```json
{
  "type": "gripper",
  "side": "right",
  "target": "open",
  "duration": 0.5
}
```

**实际执行**：
- 读取当前位置：例如 `0.01m`
- 目标位置：`0.01 + 0.04 = 0.05m`（+4cm）
- 步数：`0.5s / 0.1s = 5步`
- 每步位置：`[0.018, 0.026, 0.034, 0.042, 0.05]`
- 每步间隔：`0.1秒`
- 总时长：约 `0.5秒`

### 示例2：缓慢张开（1.2秒）

```json
{
  "type": "gripper",
  "side": "left",
  "target": "open",
  "duration": 1.2,
  "step_sleep": 0.15
}
```

**实际执行**：
- 步数：`1.2s / 0.15s = 8步`
- 每步间隔：`0.15秒`
- 总时长：约 `1.2秒`
- 更加平滑

### 示例3：精确控制步长

```json
{
  "type": "gripper",
  "side": "right",
  "target": "open",
  "step_size": 0.008,
  "step_sleep": 0.12
}
```

**实际执行**：
- 张开距离：`4cm`
- 步数：`0.04m / 0.008m = 5步`
- 每步间隔：`0.12秒`
- 总时长：约 `0.6秒`

### 示例4：瞬间张开（向后兼容）

```json
{
  "type": "gripper",
  "side": "right",
  "target": "open"
}
```

**实际执行**：
- 调用 `set_gripper(side, +0.03, controller)`
- 单帧完成（瞬间）
- 保持旧行为

### 示例5：渐进式闭合

```json
{
  "type": "gripper",
  "side": "right",
  "target": "close",
  "duration": 0.8,
  "delta": 0.025
}
```

**实际执行**：
- 闭合距离：`2.5cm`
- 步数：`0.8s / 0.1s = 8步`
- 每步间隔：`0.1秒`
- 总时长：约 `0.8秒`

---

## 🎬 在 run_pick_place_sequence.py 中的应用

### 修改前（瞬间张开）

```python
sequence.append({"type": "gripper", "side": "right", "target": "open"})
sequence.append({"type": "sleep", "seconds": 1.0})
```

**效果**：夹爪瞬间弹开，然后等待1秒

### 修改后（渐进张开）

```python
sequence.append({"type": "gripper", "side": "right", "target": "open", "duration": 0.5})
sequence.append({"type": "sleep", "seconds": 0.5})
```

**效果**：夹爪在0.5秒内平滑张开，然后等待0.5秒

---

## 🧮 速度计算

### 张开速度计算

```python
# 总张开距离
distance = 0.04  # 4cm

# 方式1：基于 duration
duration = 0.5  # 秒
step_sleep = 0.1  # 秒/步
steps = duration / step_sleep  # 5步
speed = distance / duration  # 0.08 m/s = 8 cm/s

# 方式2：基于 step_size
step_size = 0.008  # 米/步
steps = distance / step_size  # 5步
total_time = steps * step_sleep  # 0.5秒
speed = distance / total_time  # 0.08 m/s
```

### 推荐速度

| 场景 | duration | 速度 | 说明 |
|------|----------|------|------|
| **快速** | 0.3s | 13.3 cm/s | 适合演示，较快 |
| **正常** | 0.5s | 8 cm/s | 推荐，自然 |
| **缓慢** | 1.0s | 4 cm/s | 适合精确操作 |
| **极慢** | 2.0s | 2 cm/s | 适合展示细节 |

---

## 🔧 实现原理

### 代码逻辑（curi1_control.py）

```python
# 第2249-2296行
if t_lower == "open":
    duration = float(cmd.get("duration", 0.0))
    step_sleep = float(cmd.get("step_sleep", 0.1))

    if duration > 0:
        # 1. 读取当前夹爪位置
        current = read_gripper_position(side)

        # 2. 计算目标位置
        target_pos = current + 0.04  # 张开4cm

        # 3. 计算步数
        steps = int(duration / step_sleep)

        # 4. 生成中间位置序列
        intermediate = np.linspace(current, target_pos, steps + 1)[1:]

        # 5. 为每个位置生成指令
        for val in intermediate:
            new_cmds.append({
                "type": "gripper",
                "side": side,
                "target": float(val),  # 数值类型！
                "split": True,
            })
            new_cmds.append({"type": "sleep", "seconds": step_sleep})

        # 6. 插入到指令队列前面
        automation_queue = new_cmds + automation_queue
```

### 关键点

1. **读取当前位置**：从 MuJoCo `data.qpos` 读取实际关节位置
2. **生成中间点**：使用 `np.linspace()` 在当前和目标位置间均匀插值
3. **递归处理**：生成的中间指令仍然是 `"type": "gripper"`，但 `target` 是数值
4. **队列插入**：新指令插入到队列前面，立即执行

---

## ⚙️ 调试技巧

### 查看执行过程

在终端A（运行 `curi1_control.py`）中会看到：

```bash
# 渐进模式
[automation] gripper right -> open (gradual, 5 steps)

# 瞬间模式
[automation] gripper right -> open (instant)
```

### 调整参数

如果运动不够平滑：
```json
{
  "duration": 1.0,      // 增加总时长
  "step_sleep": 0.05    // 减小每步间隔（更多步）
}
```

如果运动太慢：
```json
{
  "duration": 0.3,      // 减小总时长
  "step_sleep": 0.1     // 保持间隔不变
}
```

---

## 🆚 对比总结

| 特性 | 瞬间模式 | 渐进模式 |
|------|----------|----------|
| **触发条件** | `duration=0` 且 `step_size>=0.03` | `duration>0` 或 `step_size<0.03` |
| **执行方式** | 直接修改 `data.qpos` | 生成多个中间指令 |
| **帧数** | 1帧 | N帧（取决于步数） |
| **视觉效果** | 瞬移、跳变 | 平滑、连续 |
| **总时长** | 0秒 | 可控（duration参数） |
| **真实性** | 低（物理上不可能） | 高（接近真实机器人） |
| **适用场景** | 快速演示、调试 | 录制训练数据、展示 |
| **向后兼容** | ✅ 默认行为 | ✅ 可选启用 |

---

## 📚 完整示例：抓取序列

```python
# 在 run_pick_place_sequence.py 中
sequence = [
    # 1. 移动到准备位置
    move("right", [x, y, 0.62], [90, 0, 0], duration=2.0),
    {"type": "sleep", "seconds": 1.0},

    # 2. 调整姿态
    move("right", [x, y, 0.62], [105, 0, 0], duration=2.0),
    {"type": "sleep", "seconds": 1.0},

    # 3. 渐进式张开夹爪（0.5秒）
    {"type": "gripper", "side": "right", "target": "open", "duration": 0.5},
    {"type": "sleep", "seconds": 0.5},

    # 4. 下降靠近物体
    move("right", [x, y-0.05, 0.59], [105, 0, 0], duration=2.0),
    {"type": "sleep", "seconds": 1.0},

    # 5. 渐进式闭合夹爪（多步精确控制）
    {"type": "gripper", "side": "right", "target": 0.0325},
    {"type": "sleep", "seconds": 0.2},
    {"type": "gripper", "side": "right", "target": 0.028},
    {"type": "sleep", "seconds": 0.2},
    {"type": "gripper", "side": "right", "target": 0.019},
    {"type": "sleep", "seconds": 0.2},

    # 6. 提升物体
    move("right", [x, y-0.05, 0.70], [105, 0, 0], duration=2.0),
]
```

---

## 🎯 推荐配置

### 典型场景配置

#### 场景1：演示抓取（追求真实感）

```json
{
  "type": "gripper",
  "side": "right",
  "target": "open",
  "duration": 0.8,
  "step_sleep": 0.1
}
```

#### 场景2：快速调试

```json
{
  "type": "gripper",
  "side": "right",
  "target": "open"
}
```
（使用默认瞬间模式）

#### 场景3：录制训练数据

```json
{
  "type": "gripper",
  "side": "right",
  "target": "open",
  "duration": 0.5,
  "step_sleep": 0.08
}
```
（稍快但足够平滑）

---

## 🐛 常见问题

### Q1: 为什么设置了 duration 但还是瞬间张开？

**A**: 检查 `step_size` 参数，如果 `step_size >= 0.03` 且 `duration=0`，会触发瞬间模式。

**解决**：明确设置 `duration > 0`

### Q2: 运动太慢/太快怎么办？

**A**: 调整 `duration` 参数：
- 太慢：减小 `duration`（例如从 1.0 改为 0.5）
- 太快：增大 `duration`（例如从 0.3 改为 0.8）

### Q3: 运动不够平滑，看起来卡顿？

**A**: 减小 `step_sleep` 增加步数：
```json
{
  "duration": 0.5,
  "step_sleep": 0.05  // 从 0.1 减小到 0.05
}
```
这会将步数从5步增加到10步

### Q4: 渐进模式会影响性能吗？

**A**: 轻微影响。渐进模式会生成多个中间指令，每步需要一次物理模拟。但对于典型的5-10步，性能影响可忽略。

---

## 📖 参考

- 修改文件1: [curi1_control.py:2249-2353](../curi1_control.py#L2249-L2353)
- 修改文件2: [run_pick_place_sequence.py:80,89](./run_pick_place_sequence.py#L80)
- 相关函数: `set_gripper()` (第908行)
- MuJoCo文档: https://mujoco.readthedocs.io/

---

## 版本信息

- **版本**: v1.0
- **日期**: 2025-01-04
- **修改内容**: 添加渐进式夹爪张开/闭合功能
- **向后兼容**: ✅ 是
