import numpy as np
import matplotlib.pyplot as plt

hbar = 1.0545718e-34     # Дж·с
m    = 9.10938356e-31    # кг
eV   = 1.602176634e-19   # 1 эВ = 1.602...e-19 Дж

U0_eV    = 50.0          # глубина прямоугольной ямы U0 > 0 (внутри U = -U0), эВ
a_ang    = 1.0           # половина ширины ямы a, ангстрем
L_factor = 5.0           # область моделирования [-L, L], L = L_factor * a
N        = 1000          # число сегментов (узлов будет N + 1)
num_show = 5             # сколько низших уровней / ψ отобразить

# Режим задания формы ямы: "rect" (прямоугольная), "analytic" (аналитическая функция), "table" (таблица CSV)
potential_mode = "analytic"

# --- Параметры для режима "analytic" ---
k_par = 5.0e2
def U_analytic_J(x: np.ndarray) -> np.ndarray:
    return 0.5 * k_par * x**2 # Дж

# --- Параметры для режима "table" ---
csv_path   = "potential.csv"  # путь к CSV
x_col      = 0                # индекс столбца x (м)
v_col      = 1                # индекс столбца V
csv_units  = "eV"             # "eV" или "J"


a   = a_ang * 1e-10       # м
L   = L_factor * a        # м
x   = np.linspace(-L, L, N+1)
dx  = x[1] - x[0]

V = np.zeros_like(x)

if potential_mode == "rect":
    U0_J = U0_eV * eV
    V[np.abs(x) < a] = -U0_J  # Дж

elif potential_mode == "analytic":
    V = np.asarray(U_analytic_J(x), dtype=float)

elif potential_mode == "table":
    if csv_path is None:
        raise RuntimeError("Не задан csv_path для режима 'table'.")
    tab = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    x_tab = tab[:, x_col].astype(float)
    v_tab = tab[:, v_col].astype(float)
    if csv_units.lower() == "ev":
        v_tab = v_tab * eV
    V = np.interp(x, x_tab, v_tab, left=v_tab[0], right=v_tab[-1])
else:
    raise RuntimeError(f"Неизвестный potential_mode: {potential_mode}")

# Гамильтониан на внутренних узлах (ψ(-L) = ψ(+L) = 0)
Nint = N - 1                               # узлов внутри: 1 .. N - 1
t = (hbar ** 2) / (2 * m * dx ** 2)        # коэффициент кинетической энергии
main = np.full(Nint, 2 * t) + V[1 : -1]    # диагональ
off  = np.full(Nint - 1, -t)               # под / над диагональю

H = np.zeros((Nint, Nint), dtype=float)
np.fill_diagonal(H, main)
i = np.arange(Nint - 1)
H[i, i + 1] = off
H[i + 1, i] = off

# Собственные вектора
E, vecs = np.linalg.eigh(H)

# Волновые функции и нормировка
psis = []
for k_idx in range(min(num_show, len(E))):
    psi_full = np.zeros_like(x)
    psi_full[1 : -1] = vecs[:, k_idx]
    norm = np.sqrt(np.trapz(np.abs(psi_full) ** 2, x))
    psi_full /= norm
    psis.append(psi_full)
psis = np.array(psis)
E_eV = E / eV

# «Связанные» состояния
# Для прямоугольной ямы снаружи U=0 это E<0.
# Для общего случая используем E < min(U на краях области):
V_edge_eV = min(V[0]/eV, V[-1]/eV)
bound_mask = E_eV < V_edge_eV

print("Первые собственные значения (эВ):")
for i_idx in range(min(num_show, len(E_eV))):
    print(f"  n ={i_idx + 1:2d}  E = {E_eV[i_idx]: .6f} эВ")

# Волновые числа (только для прямоугольной ямы)
if potential_mode == "rect":
    print("\nВолновые числа для связанных состояний (1 / нм):")
    any_bound = False
    for i_idx in range(min(num_show, len(E))):
        if E_eV[i_idx] < 0:
            any_bound = True
            E_J = E[i_idx]
            arg_inside = max(E_J + U0_J, 0.0)
            k_inside = np.sqrt(2 * m * arg_inside) / hbar  # 1/м
            print(f"  n ={i_idx + 1:2d}  k ={k_inside * 1e-9: .4f}")
    if not any_bound:
        print("  Нет связанных состояний среди первых вычисленных уровней.")

plt.figure(figsize=(10, 6))
plt.plot(x * 1e10, V / eV, "k--", label="Потенциал $U(x)$")
vmin, vmax = np.min(V / eV), np.max(V / eV)
scale = 0.5 * (vmax - vmin) if vmax > vmin else 1.0

palette = plt.cm.tab10.colors

for i_idx, psi in enumerate(psis):
    y = psi * scale + E_eV[i_idx]
    clr = palette[i_idx % len(palette)]
    ls = "-" if bound_mask[i_idx] else "--"
    plt.plot(x * 1e10, y, color=clr, linestyle=ls,
             label=f"n={i_idx + 1}, E={E_eV[i_idx]:.3f} эВ")

plt.title("Уравнение Шредингера: собственные функции и уровни")
plt.xlabel("x, Å")
plt.ylabel("Энергия / волновые функции (эВ)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
