# -*- coding: utf-8 -*-
"""
Treliça Plana - Análise pelo Método dos Elementos Finitos (MEF)
Estrutura Orientada a Objetos com Exemplos Integrados
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys, os

# Cores para terminal
C = {
    "title" : "\033[1;36m", "header": "\033[1;34m",
    "ok"    : "\033[1;32m", "warn"  : "\033[1;33m",
    "err"   : "\033[1;31m", "reset" : "\033[0m",
    "dim"   : "\033[2m",    "bold"  : "\033[1m",
    "BLU"   : "\033[94m",   "ORG"   : "\033[33m",   "TXT"   : "\033[37m"
}
def clr(t, c): return f"{C[c]}{t}{C['reset']}"
def linha(ch="─", n=60): print(clr(ch*n, "dim"))

def input_valor(msg, tipo=float):
    """Lê e valida entrada do utilizador."""
    while True:
        try:
            val = input(msg)
            if val == "":
                return 0 if tipo == int else 0.0
            return tipo(val)
        except ValueError:
            print(clr("  ✗ Entrada inválida. Tenta novamente.", "err"))

class TrelicaPlana:
    """Encapsula o estado e o comportamento de uma treliça plana resolvida pelo MEF."""
    def __init__(self):
        self.coords = []
        self.conects = []
        self.props = []
        self.restrs = []
        self.forcas = []
        self.U = None
        self.R = None
        self.ESF = None
        self.resolvido = False

    def adicionar_no(self, x, y, restr_x=0, restr_y=0, fx=0.0, fy=0.0):
        self.coords.append([x, y])
        self.restrs.extend([restr_x, restr_y])
        self.forcas.extend([fx, fy])

    def adicionar_elemento(self, no_ini, no_fim, area, modulo_elasticidade):
        self.conects.append([no_ini, no_fim])
        self.props.append([area, modulo_elasticidade])

    def resolver(self):
        """Monta a matriz de rigidez global e resolve o sistema."""
        coords, conects = np.array(self.coords), np.array(self.conects)
        props, restrs, forcas = np.array(self.props), np.array(self.restrs), np.array(self.forcas)

        n_nos, n_el = len(coords), len(conects)
        ndof = 2 * n_nos
        KG = np.zeros((ndof, ndof))
        L_list, c_list, s_list = [], [], []

        for i in range(n_el):
            ii, jj = conects[i]
            A, E = props[i]
            dx, dy = coords[jj,0] - coords[ii,0], coords[jj,1] - coords[ii,1]
            L = np.sqrt(dx**2 + dy**2)
            c, s = dx / L, dy / L
            L_list.append(L); c_list.append(c); s_list.append(s)

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

        KG_orig, forca_orig = KG.copy(), forcas.copy()

        # Penalização para restrições
        for i in range(ndof):
            if restrs[i] == 1:
                KG[i, :] = 0.0; KG[:, i] = 0.0
                KG[i, i] = 1.0; forcas[i] = 0.0

        if np.linalg.cond(KG) > 1e14:
            print(clr("\n  ✗ ERRO: Estrutura instável (Hipostática).", "err"))
            sys.exit(1)

        self.U = np.linalg.solve(KG, forcas)
        self.R = KG_orig @ self.U - forca_orig
        self.ESF = np.zeros(n_el)

        for i in range(n_el):
            ii, jj = conects[i]
            A, E = props[i]
            L, c, s = L_list[i], c_list[i], s_list[i]
            dofs = [2*ii, 2*ii+1, 2*jj, 2*jj+1]
            Ug = np.array([self.U[d] for d in dofs])
            self.ESF[i] = (E * A / L) * np.dot(np.array([-c, -s, c, s]), Ug)

        self.resolvido = True

    def imprimir_relatorio(self):
        if not self.resolvido: return
        n_nos = len(self.coords)
        print(); linha("═")
        print(clr("  RESULTADOS DA TRELIÇA", "title")); linha("═")

        print(clr("\n  DESLOCAMENTOS NODAIS", "header")); linha()
        print(f"  {'Nó':>4}  {'u (m)':>14}  {'v (m)':>14}")
        linha()
        for i in range(n_nos):
            print(f"  {i+1:>4}  {self.U[2*i]:>14.6e}  {self.U[2*i+1]:>14.6e}")

        print(clr("\n  ESFORÇOS NORMAIS NAS BARRAS", "header")); linha()
        print(f"  {'El':>3}  {'Esforço (kN)':>14}  {'Estado':>10}")
        linha()
        for i, n_val in enumerate(self.ESF):
            estado = "Tração" if n_val > 1e-6 else ("Compressão" if n_val < -1e-6 else "Nulo")
            cor = C["BLU"] if n_val > 0 else (C["ORG"] if n_val < 0 else C["TXT"])
            print(f"  {i+1:>3}  {n_val:>14.4e}  {cor}{estado}{C['reset']}")

        print(clr("\n  REAÇÕES DE APOIO", "header")); linha()
        print(f"  {'Nó':>4}  {'Rx (kN)':>14}  {'Ry (kN)':>14}")
        linha()
        soma_rx = soma_ry = 0.0
        for i in range(n_nos):
            rx, ry = self.R[2*i], self.R[2*i+1]
            if abs(rx) > 1e-6 or abs(ry) > 1e-6:
                print(f"  {i+1:>4}  {rx:>14.5e}  {ry:>14.5e}")
                soma_rx += rx; soma_ry += ry
        print(clr(f"\n  Σ Rx = {soma_rx:.4f} kN    Σ Ry = {soma_ry:.4f} kN", "ok"))
        linha("═")

    def plotar(self):
        if not self.resolvido: return
        coords, conects = np.array(self.coords), np.array(self.conects)
        BG, AX, GRD = "#0f1117", "#161b22", "#2d333b"
        TXT, BLU, ORG = "#8b949e", "#58a6ff", "#f0883e"

        max_desl = max(abs(self.U))
        span = max(np.ptp(coords[:,0]), np.ptp(coords[:,1])) or 1
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

        ax1 = fig.add_subplot(gs[0, 0])
        style(ax1, f"Forma Deformada (amplificada ×{scale:.0f})")
        for i, (ii, jj) in enumerate(conects):
            x_orig, y_orig = [coords[ii,0], coords[jj,0]], [coords[ii,1], coords[jj,1]]
            ax1.plot(x_orig, y_orig, color=GRD, lw=1.5, alpha=0.4)
            x_def = [coords[ii,0] + self.U[2*ii]*scale, coords[jj,0] + self.U[2*jj]*scale]
            y_def = [coords[ii,1] + self.U[2*ii+1]*scale, coords[jj,1] + self.U[2*jj+1]*scale]
            ax1.plot(x_def, y_def, color=BLU, lw=2, marker='o', ms=5)

        ax2 = fig.add_subplot(gs[0, 1])
        style(ax2, "Esforços Normais (Tração=Azul, Compressão=Laranja)")
        max_N = max(abs(self.ESF)) or 1
        for i, (ii, jj) in enumerate(conects):
            x_orig, y_orig = [coords[ii,0], coords[jj,0]], [coords[ii,1], coords[jj,1]]
            N = self.ESF[i]
            cor = BLU if N >= 0 else ORG
            lw = 2 + 4 * (abs(N) / max_N)
            ax2.plot(x_orig, y_orig, color=cor, lw=lw)
            xm, ym = np.mean(x_orig), np.mean(y_orig)
            ax2.text(xm, ym, f"{N:.1f}", color="#ffffff", fontsize=8, ha="center", va="center",
                     bbox=dict(facecolor=cor, edgecolor='none', pad=2, alpha=0.8))

        plt.show()

# ==========================================
# FUNÇÕES DE EXEMPLOS PRÉ-CONFIGURADOS
# ==========================================

def exemplo_professor():
    modelo = TrelicaPlana()
    modelo.adicionar_no(0.0, 0.0, restr_x=1, restr_y=1)
    modelo.adicionar_no(4.0, 0.0, restr_y=1)
    modelo.adicionar_no(4.0, 3.0)
    modelo.adicionar_no(0.0, 3.0, fx=5.0, fy=-10.0)
    A, E = 0.01, 100.0
    modelo.adicionar_elemento(0, 1, A, E)
    modelo.adicionar_elemento(1, 2, A, E)
    modelo.adicionar_elemento(1, 3, A, E)
    modelo.adicionar_elemento(2, 3, A, E)
    modelo.adicionar_elemento(0, 3, A, E)
    return modelo

def exemplo_padrao():
    modelo = TrelicaPlana()
    modelo.adicionar_no(0.0, 0.0, restr_x=1, restr_y=1)
    modelo.adicionar_no(4.0, 0.0, restr_y=1)
    modelo.adicionar_no(4.0, 3.0, fx=5.0)
    modelo.adicionar_no(7.0, 0.0, fy=-10.0)
    A, E = 0.01, 210000.0
    modelo.adicionar_elemento(0, 1, A, E)
    modelo.adicionar_elemento(1, 2, A, E)
    modelo.adicionar_elemento(1, 3, A, E)
    modelo.adicionar_elemento(2, 3, A, E)
    modelo.adicionar_elemento(0, 2, A, E)
    return modelo

def exemplo_ponte():
    modelo = TrelicaPlana()
    modelo.adicionar_no(0.0, 0.0, restr_x=1, restr_y=1)
    modelo.adicionar_no(3.0, 0.0)
    modelo.adicionar_no(6.0, 0.0, restr_y=1)
    modelo.adicionar_no(0.0, 3.0)
    modelo.adicionar_no(3.0, 3.0, fy=-15.0)
    modelo.adicionar_no(6.0, 3.0)
    A, E = 0.01, 210000.0
    modelo.adicionar_elemento(0, 1, A, E)
    modelo.adicionar_elemento(1, 2, A, E)
    modelo.adicionar_elemento(3, 4, A, E)
    modelo.adicionar_elemento(4, 5, A, E)
    modelo.adicionar_elemento(0, 3, A, E)
    modelo.adicionar_elemento(1, 4, A, E)
    modelo.adicionar_elemento(2, 5, A, E)
    modelo.adicionar_elemento(1, 3, A, E)
    modelo.adicionar_elemento(1, 5, A, E)
    return modelo

def exemplo_balanco():
    modelo = TrelicaPlana()
    modelo.adicionar_no(0.0, 0.0, restr_x=1, restr_y=1)
    modelo.adicionar_no(0.0, 2.0, restr_x=1, restr_y=1)
    modelo.adicionar_no(2.0, 0.0)
    modelo.adicionar_no(2.0, 2.0)
    modelo.adicionar_no(4.0, 0.0, fy=-20.0)
    A, E = 0.02, 210000.0
    modelo.adicionar_elemento(0, 2, A, E)
    modelo.adicionar_elemento(2, 4, A, E)
    modelo.adicionar_elemento(1, 3, A, E)
    modelo.adicionar_elemento(0, 1, A, E)
    modelo.adicionar_elemento(2, 3, A, E)
    modelo.adicionar_elemento(1, 2, A, E)
    modelo.adicionar_elemento(3, 4, A, E)
    return modelo

def exemplo_pdf_cap3():
    modelo = TrelicaPlana()
    modelo.adicionar_no(0.0, 0.0, restr_x=1, restr_y=1)
    modelo.adicionar_no(4.0, 0.0)
    modelo.adicionar_no(8.0, 0.0, fy=-1200.0)
    modelo.adicionar_no(12.0, 0.0, restr_y=1)
    modelo.adicionar_no(4.0, 3.0, fx=400.0)
    modelo.adicionar_no(8.0, 3.0)
    A, E = 0.01, 210000.0
    modelo.adicionar_elemento(0, 1, A, E)
    modelo.adicionar_elemento(1, 2, A, E)
    modelo.adicionar_elemento(2, 3, A, E)
    modelo.adicionar_elemento(4, 5, A, E)
    modelo.adicionar_elemento(0, 4, A, E)
    modelo.adicionar_elemento(1, 4, A, E)
    modelo.adicionar_elemento(1, 5, A, E)
    modelo.adicionar_elemento(2, 5, A, E)
    modelo.adicionar_elemento(3, 5, A, E)
    return modelo

def exemplo_pdf_cap3_ex1():
    modelo = TrelicaPlana()
    modelo.adicionar_no(0.0, 0.0, restr_x=1, restr_y=1)
    modelo.adicionar_no(1.2, 0.0, fy=-1.2)
    modelo.adicionar_no(0.0, -0.5, restr_x=1, restr_y=0)
    A, E = 0.01, 210000.0
    modelo.adicionar_elemento(0, 1, A, E)
    modelo.adicionar_elemento(1, 2, A, E)
    modelo.adicionar_elemento(0, 2, A, E)
    return modelo

def exemplo_pdf_cap3_ex2():
    modelo = TrelicaPlana()
    modelo.adicionar_no(0.0, 3.0, restr_x=1, restr_y=1)
    modelo.adicionar_no(0.0, 0.0, restr_x=1, restr_y=1)
    modelo.adicionar_no(4.0, 0.0, fy=-0.6)
    modelo.adicionar_no(4.0, 3.0, fx=0.4)
    A, E = 0.01, 210000.0
    modelo.adicionar_elemento(0, 3, A, E)
    modelo.adicionar_elemento(1, 2, A, E)
    modelo.adicionar_elemento(2, 3, A, E)
    modelo.adicionar_elemento(1, 3, A, E)
    return modelo

def exemplo_professor_expandido():
    modelo = TrelicaPlana()
    modelo.adicionar_no(0.0, 0.0, restr_x=1, restr_y=1)
    modelo.adicionar_no(4.0, 0.0)
    modelo.adicionar_no(4.0, 3.0)
    modelo.adicionar_no(0.0, 3.0, fx=5.0, fy=-10.0)
    modelo.adicionar_no(8.0, 0.0, restr_y=1)
    modelo.adicionar_no(8.0, 3.0, fy=-10.0)
    A, E = 0.01, 100.0
    modelo.adicionar_elemento(0, 1, A, E)
    modelo.adicionar_elemento(1, 2, A, E)
    modelo.adicionar_elemento(1, 3, A, E)
    modelo.adicionar_elemento(2, 3, A, E)
    modelo.adicionar_elemento(0, 3, A, E)
    modelo.adicionar_elemento(1, 4, A, E)
    modelo.adicionar_elemento(2, 5, A, E)
    modelo.adicionar_elemento(4, 5, A, E)
    modelo.adicionar_elemento(1, 5, A, E)
    return modelo

def criar_modelo_interativo():
    modelo = TrelicaPlana()
    print(clr("\n  ENTRADA INTERATIVA DE DADOS", "header")); linha()
    n_nos = input_valor(clr("  Número de nós: ", "warn"), int)
    n_el  = input_valor(clr("  Número de elementos: ", "warn"), int)

    print(clr("\n  CONFIGURAÇÃO DOS NÓS", "header")); linha()
    print(clr("  Dica: Deixa em branco para assumir o valor 0.", "dim"))
    for i in range(n_nos):
        print(f"\n  Nó {i+1}:")
        x = input_valor(f"    X (m): ", float)
        y = input_valor(f"    Y (m): ", float)
        rx = input_valor(f"    Restrição em X (0=livre, 1=fixo): ", int)
        ry = input_valor(f"    Restrição em Y (0=livre, 1=fixo): ", int)
        fx = input_valor(f"    Força em X (kN): ", float)
        fy = input_valor(f"    Força em Y (kN): ", float)
        modelo.adicionar_no(x, y, rx, ry, fx, fy)

    print(clr("\n  CONFIGURAÇÃO DOS ELEMENTOS", "header")); linha()
    area_padrao = input_valor(clr("  Área comum para todos os elementos (m²): ", "warn"), float)
    modulo_padrao = input_valor(clr("  Módulo de elasticidade: ", "warn"), float)

    for i in range(n_el):
        print(f"\n  Elemento {i+1}:")
        while True:
            ini = input_valor(f"    Nó inicial (1-{n_nos}): ", int) - 1
            fim = input_valor(f"    Nó final (1-{n_nos}): ", int) - 1
            if 0 <= ini < n_nos and 0 <= fim < n_nos and ini != fim:
                break
            print(clr("    ✗ Conexão inválida. Verifica os índices.", "err"))
        modelo.adicionar_elemento(ini, fim, area_padrao, modulo_padrao)

    print(clr("\n  ✓ Estrutura montada com sucesso!", "ok"))
    return modelo

def banner():
    os.system("cls" if os.name == "nt" else "clear")
    print(clr("╔══════════════════════════════════════════════════════════╗", "title"))
    print(clr("║    ANÁLISE DE TRELIÇA PLANA — MEF (Orientado a Objetos)  ║", "title"))
    print(clr("╚══════════════════════════════════════════════════════════╝", "title"))
    print()

def main():
    banner()
    print(clr("  Escolhe uma opção:", "header"))
    print(clr("  [1] Digitar dados manualmente", "warn"))
    print(clr("  [2] Exemplo padrão (treliça simples, 4 nós)", "warn"))
    print(clr("  [3] Exemplo ponte (6 nós, 9 elementos)", "warn"))
    print(clr("  [4] Exemplo balanço / cantilever (5 nós, 7 elementos)", "warn"))
    print(clr("  [5] Exemplo do professor (4 nós, 5 elementos)", "warn"))
    print(clr("  [6] Exemplo Cap. 3 do PDF (6 nós, Método das Seções)", "warn"))
    print(clr("  [7] Exemplo Cap. 3 do PDF Ex.1 (3 nós, Método dos Nós)", "warn"))
    print(clr("  [8] Exemplo Cap. 3 do PDF Ex.2 (4 nós, Método dos Nós)", "warn"))
    print(clr("  [9] Exemplo do professor EXPANDIDO (6 nós, 9 elementos)", "warn"))
    linha()

    opcao = input(clr("  Opção: ", "bold")).strip()

    if opcao == "1":
        modelo = criar_modelo_interativo()
    elif opcao == "2":
        print(clr("  A carregar exemplo padrão...", "dim"))
        modelo = exemplo_padrao()
    elif opcao == "3":
        print(clr("  A carregar exemplo ponte...", "dim"))
        modelo = exemplo_ponte()
    elif opcao == "4":
        print(clr("  A carregar exemplo balanço...", "dim"))
        modelo = exemplo_balanco()
    elif opcao == "5":
        print(clr("  A carregar exemplo do professor...", "dim"))
        modelo = exemplo_professor()
    elif opcao == "6":
        print(clr("  A carregar exemplo do Cap. 3...", "dim"))
        modelo = exemplo_pdf_cap3()
    elif opcao == "7":
        print(clr("  A carregar exemplo 1 do Cap. 3...", "dim"))
        modelo = exemplo_pdf_cap3_ex1()
    elif opcao == "8":
        print(clr("  A carregar exemplo 2 do Cap. 3...", "dim"))
        modelo = exemplo_pdf_cap3_ex2()
    elif opcao == "9":
        print(clr("  A carregar exemplo do professor expandido...", "dim"))
        modelo = exemplo_professor_expandido()
    else:
        print(clr("  ✗ Opção inválida. A encerrar.", "err"))
        sys.exit(1)

    # Processamento e Saída
    modelo.resolver()
    modelo.imprimir_relatorio()
    modelo.plotar()

if __name__ == "__main__":
    main()