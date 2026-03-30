# Posture Feedback Message from lower computer
# 当前姿态：0=移动姿态, 1=进攻姿态, 2=防御姿态
uint8 current_posture

# 距离上次姿态切换的时间（秒）
float32 time_since_last_switch

# 当前姿态持续时间（秒）
float32 current_posture_duration

# 是否可以切换姿态（冷却时间已过）
bool can_switch

# 是否被打:0=未被打, 1=被打
bool be_attacked

# 17mm枪管热量
uint8 shooter_17mm_1_barrel_heat
