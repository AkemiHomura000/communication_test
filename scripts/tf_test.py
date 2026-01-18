import numpy as np

def Rz(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])

def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.cross(a, b)

def main():
    # ----------------------------
    # 1) 设定“具体数值”
    # ----------------------------
    # 轮速计：在 base_link 表达的机体速度
    Vx, Vy, w = 1.0, 0.0, 0.00  # m/s, m/s, rad/s
    v_b = np.array([Vx, Vy, 0.0])
    omega_b = np.array([0.0, 0.0, w])

    # 云台 yaw 及其角速度
    psi = np.pi     # rad
    psi_dot = 0.0    # rad/s
    omega_g = np.array([0.0, 0.0, psi_dot])  # gimbal relative to base

    # 外参：base->gimbal 的平移（在 base 表达）
    p_bg = np.array([0.30, 0.0, 0.0])  # m

    # 外参：gimbal->livox 的静态外参（在 gimbal 表达）
    p_gl = np.array([0.20, -0.0, 0.0])  # m
    R_gl = Rz(0.0)  # 给一个小的静态安装偏角（可改为单位阵）

    # ----------------------------
    # 2) 计算 base->livox 当前的 r_bl（在 base 表达）
    # ----------------------------
    R_bg = Rz(psi)
    R_bl = R_bg @ R_gl
    r_bl = p_bg + R_bg @ p_gl   # base 到 livox 原点向量（在 base 表达）

    # ----------------------------
    # 3) 用“第4点公式”计算 livox 原点速度（先在 base 表达）
    # ----------------------------
    omega_total = omega_b + omega_g
    v_l_formula_in_b = v_b + cross(omega_total, r_bl)
    omega_l_formula_in_b = omega_total

    # 再旋转到 livox_frame 表达
    R_lb = R_bl.T
    v_l_formula_in_l = R_lb @ v_l_formula_in_b
    omega_l_formula_in_l = R_lb @ omega_l_formula_in_b

    # ----------------------------
    # 4) 数值差分验证（ground truth）
    # ----------------------------
    # 思路：在一个很小 dt 内，
    # - base 坐标系自身在“世界”中旋转 omega_b*dt
    # - gimbal yaw 变化 psi_dot*dt
    # 我们直接在 base 系里计算 livox 原点位置 r_bl(t) 的变化：
    # r_bl(t) = p_bg + R_bg(psi(t)) * p_gl
    # dr/dt = d/dt(R_bg) * p_gl = (omega_g × (R_bg*p_gl)) （omega_g 在 base 表达）
    #
    # 同时 livox 点因为 base 在动：
    # v_l = v_b + omega_b × r + (dr/dt)   —— 这里 dr/dt 就是 omega_g × (R_bg*p_gl)
    # 注意：omega_g × r 里还包含对 p_bg 的项，但 p_bg 不随 psi 转，所以正确的是：
    # omega_g × (R_bg*p_gl) 而不是 omega_g × (p_bg + R_bg*p_gl)
    #
    # 但是我们第4点写的是 (omega_b + omega_g) × r_bl。
    # 它在严格意义上成立的前提是：livox_frame 作为一个“随云台转的刚体点”，
    # omega_g 的旋转中心在 base 原点（或 p_bg=0）。
    #
    # 因为你实际云台转轴通常在 p_bg 处（而非 base 原点），
    # 更严格的写法应是：v_l = v_b + omega_b×r_bl + omega_g×(r_bl - p_bg)
    #
    # 这里我们用两种方式都算一遍，并用差分验证哪一个“真正确”。

    dt = 1e-6  # 足够小，差分误差很小

    def r_bl_of(psi_val: float) -> np.ndarray:
        return p_bg + Rz(psi_val) @ p_gl

    r0 = r_bl_of(psi)
    r1 = r_bl_of(psi + psi_dot * dt)
    dr_num = (r1 - r0) / dt  # 数值求导：r_bl 在 base 系的变化率

    # ground truth：点速度 = base 线速度 + base 自转项 + r 自身变化项
    v_l_truth_in_b = v_b + cross(omega_b, r0) + dr_num
    omega_l_truth_in_b = omega_b + omega_g

    # 旋转到 livox 表达（用 t=psi 的 R_bl）
    v_l_truth_in_l = R_lb @ v_l_truth_in_b
    omega_l_truth_in_l = R_lb @ omega_l_truth_in_b

    # ----------------------------
    # 5) 再计算“严格版本”公式（考虑云台转轴在 p_bg 处）
    # ----------------------------
    v_l_strict_in_b = v_b + cross(omega_b, r_bl) + cross(omega_g, (r_bl - p_bg))
    omega_l_strict_in_b = omega_b + omega_g
    v_l_strict_in_l = R_lb @ v_l_strict_in_b
    omega_l_strict_in_l = R_lb @ omega_l_strict_in_b

    # ----------------------------
    # 6) 输出对比
    # ----------------------------
    np.set_printoptions(precision=10, suppress=True)

    print("=== Given numbers ===")
    print("v_b (base):", v_b)
    print("omega_b (base):", omega_b)
    print("psi, psi_dot:", psi, psi_dot)
    print("p_bg (base):", p_bg)
    print("p_gl (gimbal):", p_gl)
    print("r_bl (base):", r_bl)
    print()

    print("=== Formula in section 4 (naive: (omega_b+omega_g) x r_bl) ===")
    print("v_l formula (base):", v_l_formula_in_b)
    print("v_l formula (livox):", v_l_formula_in_l)
    print("omega_l formula (livox):", omega_l_formula_in_l)
    print()

    print("=== Numerical truth (finite difference) ===")
    print("dr_num (base):", dr_num)
    print("v_l truth (base):", v_l_truth_in_b)
    print("v_l truth (livox):", v_l_truth_in_l)
    print("omega_l truth (livox):", omega_l_truth_in_l)
    print()

    print("=== Strict formula (gimbal rotates about axis at p_bg) ===")
    print("v_l strict (base):", v_l_strict_in_b)
    print("v_l strict (livox):", v_l_strict_in_l)
    print()

    err_naive_b = v_l_formula_in_b - v_l_truth_in_b
    err_strict_b = v_l_strict_in_b - v_l_truth_in_b

    print("=== Errors vs numerical truth (base frame) ===")
    print("naive error (base):", err_naive_b, "  ||err|| =", np.linalg.norm(err_naive_b))
    print("strict error (base):", err_strict_b, "  ||err|| =", np.linalg.norm(err_strict_b))

if __name__ == "__main__":
    main()
