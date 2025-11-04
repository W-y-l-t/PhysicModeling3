import numpy as np
import matplotlib.pyplot as plt

hbar = 1.0545718e-34     # Дж·с
m    = 9.10938356e-31    # кг
eV   = 1.602176634e-19   # 1 эВ = 1.602...e-19 Дж

U0_eV    = 50.0          # глубина ямы U0 > 0 (внутри ямы U = -U0), эВ
a_ang    = 1.0           # половина ширины ямы a, ангстрем
L_factor = 5.0           # область моделирования [-L, L], L = L_factor * a
N        = 1000          # число сегментов (узлов будет N + 1)
num_show = 3             # сколько низших уровней / ψ отобразить

a = a_ang * 1e-10        # м
L = L_factor * a         # м
U0_J = U0_eV * eV        # Дж

x  = np.linspace(-L, L, N+1)
dx = x[1] - x[0]

# Потенциал прямоугольной конечной ямы
V = np.zeros_like(x)
V[np.abs(x) < a] = -U0_J    # Дж

# Матрица Гамильтониана на внутренних узлах (ψ(L) = ψ(-L) = 0)
Nint = N - 1                               # узлов внутри: 1..N-1
t = (hbar ** 2) / (2 * m * dx ** 2)        # коэффициент кинетической энергии
main = np.full(Nint, 2 * t) + V[1 : -1]    # диагональ
off  = np.full(Nint - 1, -t)               # под / над диагональю

H = np.zeros((Nint, Nint), dtype=float)
np.fill_diagonal(H, main)
i = np.arange(Nint - 1)
H[i, i + 1] = off
H[i + 1, i] = off

# Собственные значения
E, vecs = np.linalg.eigh(H)

# Восстановление полных волновых функций и нормировка
psis = []
for k_idx in range(min(num_show, len(E))):
    psi_full = np.zeros_like(x)
    psi_full[1 : -1] = vecs[:, k_idx]
    norm = np.sqrt(np.trapz(np.abs(psi_full) ** 2, x))
    psi_full /= norm
    psis.append(psi_full)
psis = np.array(psis)
E_eV = E / eV

bound_mask = E_eV < 0

print("Первые собственные значения (эВ):")
for i_idx in range(min(num_show, len(E_eV))):
    print(f"  n ={i_idx + 1:2d}  E = {E_eV[i_idx]: .6f} эВ")

print("\nВолновые числа для связанных состояний (1 / нм):")
any_bound = False
for i_idx in range(min(num_show, len(E))):
    if bound_mask[i_idx]:
        any_bound = True
        E_J = E[i_idx]
        arg_inside = max(E_J + U0_J, 0.0)
        k_inside = np.sqrt(2 * m * arg_inside) / hbar
        print(f"  n ={i_idx + 1:2d}  k ={k_inside * 1e-9: .4f}")
if not any_bound:
    print("Нет связанных состояний среди первых вычисленных уровней.")

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

plt.title("Прямоугольная конечная яма: собственные функции и уровни")
plt.xlabel("x, Å")
plt.ylabel("Энергия / волновые функции (эВ)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
