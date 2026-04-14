
# Treliça Plana - Análise pelo Método dos Elementos Finitos (MEF)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys, os

#cores para terminal
C = {
    "title" : "\033[1;36m", "header": "\033[1;34m",
    "ok"    : "\033[1;32m", "warn"  : "\033[1;33m",
    "err"   : "\033[1;31m", "reset" : "\033[0m",
    "dim"   : "\033[2m",    "bold"  : "\033[1m",
    "BLU"   : "\033[94m",   "ORG"   : "\033[33m",   "TXT"   : "\033[37m"
}
def clr(t, c): return f"{C[c]}{t}{C['reset']}"
def linha(ch="─", n=60): print(clr(ch*n, "dim"))

def banner():
    os.system("cls" if os.name == "nt" else "clear")
    print(clr("╔══════════════════════════════════════════════════════════╗", "title"))
    print(clr("║    ANÁLISE DE TRELIÇA PLANA — MEF                        ║", "title"))
    print(clr("║    DOFs por nó: u · v   |   Apenas Forças Axiais         ║", "title"))
    print(clr("╚══════════════════════════════════════════════════════════╝", "title"))
    print()

#Dados de entrada - com opção de uso interativo
def exemplo_padrao():
    """Retorna dados pré-configurados de exemplo."""
    n_nos = 4
    n_el = 5
    
    # Coordenadas (X, Y)
    coord = np.array([
        [0.0, 0.0],  # Nó 1 (Apoio esquerdo)
        [4.0, 0.0],  # Nó 2 (Apoio central)
        [4.0, 3.0],  # Nó 3 (Topo do mastro)
        [7.0, 0.0]   # Nó 4 (Ponta direita)
    ])
    
    # Conectividade [nó i, nó j] (índices base zero)
    conect = np.array([
        [0, 1],  # El 1: Nó 1 a 2
        [1, 2],  # El 2: Nó 2 a 3
        [1, 3],  # El 3: Nó 2 a 4
        [2, 3],  # El 4: Nó 3 a 4
        [0, 2]   # El 5: Nó 1 a 3 (Cabo estabilizador)
    ])
    
    # Propriedades [Área, Módulo de Elasticidade]
    prop = np.array([
        [0.01, 210000.0],
        [0.01, 210000.0],
        [0.01, 210000.0],
        [0.01, 210000.0],
        [0.01, 210000.0]
    ])
    
    # Vínculos de Apoio (0 = livre, 1 = restrito)
    ndof = 2 * n_nos
    restr = np.zeros(ndof, dtype=int)
    restr[0] = 1  # Nó 1, DOF 1 (u - horiz)
    restr[1] = 1  # Nó 1, DOF 2 (v - vert)
    restr[3] = 1  # Nó 2, DOF 2 (v - vert)
    
    # Cargas Nodais Concentradas (Positivo: →, ↑)
    forca = np.zeros(ndof)
    forca[4] = 5.0    # Nó 3, Fx = 5 kN
    forca[7] = -10.0  # Nó 4, Fy = -10 kN
    
    return n_nos, n_el, coord, conect, prop, restr, forca

def input_valor(msg, tipo=float):
    """Lê e valida entrada do usuário."""
    while True:
        try:
            val = input(msg)
            if val == "":
                return 0
            return tipo(val)
        except ValueError:
            print(clr("  ✗ Entrada inválida. Tente novamente.", "err"))

def setup_dados():
    """Interface interativa para entrada de dados."""
    print(clr("\n  ENTRADA INTERATIVA DE DADOS", "header")); linha()
    
    opcao = input(clr("  Usar dados de exemplo? (s/n): ", "warn")).lower()
    if opcao == 's':
        return exemplo_padrao()
    
    print()
    n_nos = input_valor(clr("  Número de nós: ", "warn"), int)
    n_el  = input_valor(clr("  Número de elementos: ", "warn"), int)
    
    # Coordenadas dos nós
    print(clr("\n  COORDENADAS DOS NÓS", "header")); linha()
    coord = np.zeros((n_nos, 2))
    for i in range(n_nos):
        print(f"\n  Nó {i+1}:")
        coord[i, 0] = input_valor(f"    X (m): ", float)
        coord[i, 1] = input_valor(f"    Y (m): ", float)
    
    # Conectividade
    print(clr("\n  CONECTIVIDADE DOS ELEMENTOS", "header")); linha()
    conect = np.zeros((n_el, 2), dtype=int)
    for i in range(n_el):
        print(f"\n  Elemento {i+1}:")
        ini = input_valor(f"    Nó inicial (1-{n_nos}): ", int) - 1
        fim = input_valor(f"    Nó final (1-{n_nos}): ", int) - 1
        
        if ini < 0 or ini >= n_nos or fim < 0 or fim >= n_nos or ini == fim:
            print(clr("  ✗ Nós inválidos!", "err"))
            i -= 1
            continue
        
        conect[i] = [ini, fim]
    
    # Propriedades
    print(clr("\n  PROPRIEDADES DOS ELEMENTOS", "header")); linha()
    prop = np.zeros((n_el, 2))
    area_padrao = input_valor(clr("  Área comum para todos (m²): ", "warn"), float)
    modulo_padrao = input_valor(clr("  Módulo de elasticidade (Pa): ", "warn"), float)
    
    resp = input(clr("  Mesmo para todos os elementos? (s/n): ", "warn")).lower()
    if resp == 's':
        prop[:] = [area_padrao, modulo_padrao]
    else:
        for i in range(n_el):
            print(f"\n  Elemento {i+1}:")
            prop[i, 0] = input_valor(f"    Área (m²): ", float) or area_padrao
            prop[i, 1] = input_valor(f"    Módulo (Pa): ", float) or modulo_padrao
    
    # Restrições
    print(clr("\n  RESTRIÇÕES (APOIOS)", "header")); linha()
    print(clr("  (0=livre, 1=fixo)", "dim"))
    ndof = 2 * n_nos
    restr = np.zeros(ndof, dtype=int)
    for i in range(n_nos):
        print(f"\n  Nó {i+1}:")
        restr[2*i] = input_valor(f"    Restrição X (0/1): ", int)
        restr[2*i+1] = input_valor(f"    Restrição Y (0/1): ", int)
    
    # Cargas
    print(clr("\n  CARGAS NODAIS", "header")); linha()
    forca = np.zeros(ndof)
    resp = input(clr("  Há cargas aplicadas? (s/n): ", "warn")).lower()
    if resp == 's':
        for i in range(n_nos):
            fx = input_valor(clr(f"  Nó {i+1} - Força X (kN): ", "warn"), float)
            fy = input_valor(clr(f"  Nó {i+1} - Força Y (kN): ", "warn"), float)
            forca[2*i] = fx
            forca[2*i+1] = fy
    
    print(clr("\n  ✓ Dados carregados com sucesso!", "ok"))
    return n_nos, n_el, coord, conect, prop, restr, forca

#Matriz de rigidez global, solução do sistema linear, cálculo dos esforços normais e reações de apoio
def solver(n_nos, n_el, coord, conect, prop, restr, forca):
    ndof = 2 * n_nos
    KG   = np.zeros((ndof, ndof))
    L_list, c_list, s_list = [], [], []

    for i in range(n_el):
        ii, jj = conect[i]
        A, E   = prop[i]
        
        dx = coord[jj,0] - coord[ii,0]
        dy = coord[jj,1] - coord[ii,1]
        L  = np.sqrt(dx**2 + dy**2)
        c  = dx / L
        s  = dy / L
        
        L_list.append(L); c_list.append(c); s_list.append(s)

        # Matriz de Rigidez Local -> Global 4x4
        k = (E * A) / L
        Kg = k * np.array([
            [ c*c,  c*s, -c*c, -c*s],
            [ c*s,  s*s, -c*s, -s*s],
            [-c*c, -c*s,  c*c,  c*s],
            [-c*s, -s*s,  c*s,  s*s]
        ])

        dofs = [2*ii, 2*ii+1, 2*jj, 2*jj+1]
        for a in range(4):
            for b in range(4):
                KG[dofs[a], dofs[b]] += Kg[a, b]

    KG_orig    = KG.copy()
    forca_orig = forca.copy()

    # Penalização para condições de contorno
    for i in range(ndof):
        if restr[i] == 1:
            KG[i, :] = 0.0; KG[:, i] = 0.0
            KG[i, i] = 1.0; forca[i] = 0.0

    cond = np.linalg.cond(KG)
    if cond > 1e14:
        print(clr("\n  ✗ ERRO: Estrutura instável (Hipostática).", "err"))
        sys.exit(1)

    U = np.linalg.solve(KG, forca)
    R = KG_orig @ U - forca_orig

    # Cálculo dos esforços normais
    ESF = np.zeros(n_el)
    for i in range(n_el):
        ii, jj = conect[i]
        A, E   = prop[i]
        L, c, s = L_list[i], c_list[i], s_list[i]
        
        dofs = [2*ii, 2*ii+1, 2*jj, 2*jj+1]
        Ug = np.array([U[d] for d in dofs])
        
        B = np.array([-c, -s, c, s])
        N = (E * A / L) * np.dot(B, Ug)
        ESF[i] = N

    return U, R, ESF

# Impressão dos resultados no terminal e geração dos gráficos
def imprimir(n_nos, n_el, U, R, ESF):
    print(); linha("═")
    print(clr("  RESULTADOS DA TRELIÇA", "title")); linha("═")

    print(clr("\n  DESLOCAMENTOS NODAIS", "header")); linha()
    print(f"  {'Nó':>4}  {'u (m)':>14}  {'v (m)':>14}")
    linha()
    for i in range(n_nos):
        print(f"  {i+1:>4}  {U[2*i]:>14.6e}  {U[2*i+1]:>14.6e}")

    print(clr("\n  ESFORÇOS NORMAIS NAS BARRAS", "header")); linha()
    print(f"  {'El':>3}  {'Esforço (kN)':>14}  {'Estado':>10}")
    linha()
    for i, n_val in enumerate(ESF):
        estado = "Tração" if n_val > 1e-6 else ("Compressão" if n_val < -1e-6 else "Nulo")
        cor = C["BLU"] if n_val > 0 else (C["ORG"] if n_val < 0 else C["TXT"])
        print(f"  {i+1:>3}  {n_val:>14.4e}  {cor}{estado}{C['reset']}")

    print(clr("\n  REAÇÕES DE APOIO", "header")); linha()
    print(f"  {'Nó':>4}  {'Rx (kN)':>14}  {'Ry (kN)':>14}")
    linha()
    soma_rx = soma_ry = 0.0
    for i in range(n_nos):
        rx, ry = R[2*i], R[2*i+1]
        if abs(rx) > 1e-6 or abs(ry) > 1e-6:
            print(f"  {i+1:>4}  {rx:>14.5e}  {ry:>14.5e}")
            soma_rx += rx; soma_ry += ry
    print(clr(f"\n  Σ Rx = {soma_rx:.4f} kN    Σ Ry = {soma_ry:.4f} kN", "ok"))
    linha("═")

def gerar_graficos(n_nos, n_el, coord, conect, U, ESF):
    BG = "#0f1117"; AX = "#161b22"; GRD = "#2d333b"
    TXT = "#8b949e"; BLU = "#58a6ff"; ORG = "#f0883e"

    max_desl = max(abs(U))
    span = max(np.ptp(coord[:,0]), np.ptp(coord[:,1])) or 1
    scale = (0.12 * span / max_desl) if max_desl > 1e-15 else 1.0

    fig = plt.figure(figsize=(12, 6), facecolor=BG)
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)

    def style(ax, title):
        ax.set_facecolor(AX)
        for s in ax.spines.values(): s.set_edgecolor(GRD)
        ax.tick_params(colors=TXT, labelsize=8)
        ax.set_title(title, color="#e8eaf6", fontsize=10, fontweight="bold")
        ax.grid(color=GRD, linestyle="--", lw=0.5, alpha=0.6)
        ax.set_aspect("equal", adjustable="datalim")

    # Painel 1: Deformada
    ax1 = fig.add_subplot(gs[0, 0])
    style(ax1, f"Forma Deformada (amplificada ×{scale:.0f})")
    
    for i in range(n_el):
        ii, jj = conect[i]
        x_orig = [coord[ii,0], coord[jj,0]]
        y_orig = [coord[ii,1], coord[jj,1]]
        ax1.plot(x_orig, y_orig, color=GRD, lw=1.5, alpha=0.4) 
        
        x_def = [coord[ii,0] + U[2*ii]*scale, coord[jj,0] + U[2*jj]*scale]
        y_def = [coord[ii,1] + U[2*ii+1]*scale, coord[jj,1] + U[2*jj+1]*scale]
        ax1.plot(x_def, y_def, color=BLU, lw=2, marker='o', ms=5)

    # Painel 2: Esforço Normal
    ax2 = fig.add_subplot(gs[0, 1])
    style(ax2, "Esforços Normais (Tração=Azul, Compressão=Laranja)")
    
    max_N = max(abs(ESF)) or 1
    for i in range(n_el):
        ii, jj = conect[i]
        x_orig = [coord[ii,0], coord[jj,0]]
        y_orig = [coord[ii,1], coord[jj,1]]
        
        N = ESF[i]
        cor = BLU if N >= 0 else ORG
        lw = 2 + 4 * (abs(N) / max_N) 
        
        ax2.plot(x_orig, y_orig, color=cor, lw=lw)
        
        xm, ym = np.mean(x_orig), np.mean(y_orig)
        ax2.text(xm, ym, f"{N:.1f} kN", color="#ffffff", fontsize=8, ha="center", va="center",
                 bbox=dict(facecolor=cor, edgecolor='none', pad=2, alpha=0.8))

    plt.show()

def main():
    banner()
    print(clr("  Executando análise com os dados pré-configurados...", "warn"))
    
    n_nos, n_el, coord, conect, prop, restr, forca = setup_dados()
    U, R, ESF = solver(n_nos, n_el, coord, conect, prop, restr, forca)
    
    imprimir(n_nos, n_el, U, R, ESF)
    gerar_graficos(n_nos, n_el, coord, conect, U, ESF)

if __name__ == "__main__":
    main()

#o codigo nao tem objetos ou classes, é estruturado em funções para facilitar a leitura e manutenção. Ele é projetado para ser simples e direto, focando na análise de treliças planas usando o método dos elementos finitos (MEF). So que ele pode ser feito atraves de objetos e classes, so que isso deixaria mais complexo, de certa foorma