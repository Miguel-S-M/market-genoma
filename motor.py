"""Motor de evolução (etapa 1): mercado de MENTIRA + agentes que evoluem.

Regras desta versão (mude os parâmetros em P para experimentar):
- Cada agente tem 3 genes: tendencia (+ segue a alta, - aposta contra), risco, horizonte.
- Decide com dados até ontem e ganha/perde com o retorno de hoje (sem espiar o futuro).
- A cada época (muda o regime de mercado) os agentes são avaliados:
  passa a "campo" quem tem reputação (Sharpe) > 0; clona ao bater um degrau de lucro;
  sai se perder mais que limite_perda: HIBERNA se já foi bom (lucro >= 2 épocas no mesmo
  regime), senão é ELIMINADO. O dinheiro volta ao caixa geral.
- O grupo "generalista" é o controle: genomas aleatórios que não evoluem.
- Todo movimento de dinheiro vai para o livro-razão (tabela caixa).
Ainda não tem: teto/divisão 25-65-10, empréstimos, prazos, hierarquia ativa e dados reais.
"""
import datetime as dt
import json
import random
import statistics as st

P = dict(epocas=8, dias=40, pop=16, controle=4, capital=1000.0, caixa0=40000.0,
         capital_filho=500.0, reserva=2000.0, degraus=[100, 300, 900],
         limite_perda=0.15, custo=0.001, seed=7)
REG = {"alta calma": (0.0008, 0.006), "alta volátil": (0.0008, 0.02), "lateral": (0.0, 0.008),
       "queda": (-0.001, 0.012), "crise": (-0.003, 0.03)}  # (média, volatilidade) diárias
SETORES = {"tecnologia": 1.3, "energia": 0.9, "financeiro": 1.0, "varejo": 1.1}  # beta
EQUIPE = {5: "curto", 20: "medio", 60: "longo"}


def genes():
    return {"tendencia": round(random.uniform(-1, 1), 2), "risco": round(random.uniform(0.2, 1), 2),
            "horizonte_dias": random.choice([5, 20, 60])}


def mutar(g):
    f = dict(g)
    f["tendencia"] = round(max(-1, min(1, f["tendencia"] + random.gauss(0, 0.15))), 2)
    f["risco"] = round(max(0.1, min(1, f["risco"] + random.gauss(0, 0.1))), 2)
    if random.random() < 0.2:
        f["horizonte_dias"] = random.choice([5, 20, 60])
    return f


def rep(a):
    """Reputação = retorno médio diário / desvio, anualizado. Só vale com 40+ dias."""
    r = a["rets"]
    if len(r) < 40:
        return 0.0
    s = st.pstdev(r)
    return round(st.mean(r) / s * 15.8, 2) if s else 0.0


def rodar(c):
    random.seed(P["seed"])
    c.executescript("""
    DROP TABLE IF EXISTS genomas; DROP TABLE IF EXISTS patrimonio;
    DROP TABLE IF EXISTS operacoes; DROP TABLE IF EXISTS caixa;
    CREATE TABLE genomas(id INTEGER PRIMARY KEY, nome TEXT, equipe TEXT, setor TEXT, status TEXT,
        nascimento TEXT, morte TEXT, pai_id INTEGER, genes TEXT, capital_inicial REAL,
        capital_atual REAL, melhor_regime TEXT, reputacao REAL, evolui INTEGER);
    CREATE TABLE patrimonio(genoma_id INTEGER, dia INTEGER, valor REAL);
    CREATE TABLE operacoes(genoma_id INTEGER, data TEXT, ativo TEXT, lado TEXT, qtd INTEGER,
        preco REAL, resultado REAL);
    CREATE TABLE caixa(id INTEGER PRIMARY KEY AUTOINCREMENT, data TEXT, origem TEXT,
        destino TEXT, valor REAL, motivo TEXT);""")
    total = P["epocas"] * P["dias"]
    hoje = dt.date.today()
    dia2d = lambda d: (hoje - dt.timedelta(days=total - d)).isoformat()
    cx, led, ags = {"v": P["caixa0"]}, [], []

    def mov(d, origem, destino, val, motivo):
        led.append((dia2d(d), origem, destino, round(val, 2), motivo))

    def novo(d, gn, pai, evolui, cap):
        a = dict(id=len(ags) + 1, genes=gn, setor=random.choice(list(SETORES)),
                 equipe=EQUIPE[gn["horizonte_dias"]], evolui=evolui, status="treino", nasc=d,
                 morte=None, pai=pai, v=cap, v0=cap, pos=0.0, ent=cap, rets=[], curva=[],
                 ops=[], reg={}, clones=0, ep0=cap)
        a["nome"] = f"G-{a['id']:03d}"
        ags.append(a)
        cx["v"] -= cap
        mov(d, "caixa", a["nome"], cap, "clonagem" if pai else "capital inicial")
        return a

    mov(0, "externo", "caixa", P["caixa0"], "aporte inicial (simulado)")
    for i in range(P["pop"] + P["controle"]):
        a = novo(0, genes(), None, i < P["pop"], P["capital"])
        if i >= P["pop"]:
            a["equipe"] = "generalista"

    px = {s: [100.0] for s in SETORES}
    d = 0
    for _ in range(P["epocas"]):
        reg = random.choices(list(REG), [4, 2, 3, 2, 1])[0]
        mu, sg = REG[reg]
        for _ in range(P["dias"]):
            m = random.gauss(mu, sg)
            for s, b in SETORES.items():
                px[s].append(px[s][-1] * (1 + b * m + random.gauss(0, 0.004)))
            d += 1
            for a in [x for x in ags if x["status"] in ("treino", "campo")]:
                serie, h = px[a["setor"]], a["genes"]["horizonte_dias"]
                r = serie[-1] / serie[-2] - 1
                a["v"] *= 1 + a["pos"] * r          # posição decidida ontem
                a["rets"].append(a["pos"] * r)
                mom = serie[-1] / serie[-1 - h] - 1 if len(serie) > h else 0
                nova = a["genes"]["risco"] if mom * a["genes"]["tendencia"] > 0 else 0.0
                if (nova > 0) != (a["pos"] > 0):    # abriu ou fechou posição
                    a["v"] *= 1 - P["custo"]
                    p = round(serie[-1], 2)
                    q = max(1, int(a["v"] * max(nova, a["pos"]) / p))
                    ativo = "SIM-" + a["setor"][:4].upper()
                    if nova > 0:
                        a["ent"] = a["v"]
                        a["ops"].append((dia2d(d), ativo, "compra", q, p, 0.0))
                    else:
                        a["ops"].append((dia2d(d), ativo, "venda", q, p, round(a["v"] - a["ent"], 2)))
                a["pos"] = nova
                if d % 4 == 0:
                    a["curva"].append(a["v"])
        # fim da época: avaliar
        for a in [x for x in ags if x["status"] in ("treino", "campo")]:
            a["reg"].setdefault(reg, []).append(a["v"] / a["ep0"] - 1)
            a["ep0"] = a["v"]
            if not a["evolui"]:
                continue
            if a["status"] == "treino" and rep(a) > 0:
                a["status"] = "campo"
            if a["v"] < a["v0"] * (1 - P["limite_perda"]):
                hib = any(len(x) >= 2 and sum(x) > 0 for x in a["reg"].values())
                a["status"], a["morte"] = ("hibernado" if hib else "eliminado"), d
                cx["v"] += a["v"]
                mov(d, a["nome"], "caixa", a["v"], "hibernação" if hib else "desligamento")
            elif (a["clones"] < len(P["degraus"]) and a["v"] - a["v0"] >= P["degraus"][a["clones"]]
                  and cx["v"] >= P["capital_filho"] + P["reserva"]):
                novo(d, mutar(a["genes"]), a["id"], True, P["capital_filho"])
                a["clones"] += 1
        while (sum(1 for x in ags if x["evolui"] and x["status"] in ("treino", "campo")) < 8
               and cx["v"] >= P["capital"] + P["reserva"]):
            novo(d, genes(), None, True, P["capital"])

    for a in ags:
        melhor = (max(a["reg"], key=lambda k: sum(a["reg"][k]) / len(a["reg"][k]))
                  if a["reg"] else "sem dados")
        c.execute("INSERT INTO genomas VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (a["id"], a["nome"], a["equipe"], a["setor"], a["status"], dia2d(a["nasc"]),
                   dia2d(a["morte"]) if a["morte"] else None, a["pai"], json.dumps(a["genes"]),
                   a["v0"], a["v"], melhor, rep(a), int(a["evolui"])))
        c.executemany("INSERT INTO patrimonio VALUES (?,?,?)",
                      [(a["id"], i, v) for i, v in enumerate(a["curva"] or [a["v"]])])
        c.executemany("INSERT INTO operacoes VALUES (?,?,?,?,?,?,?)", [(a["id"], *o) for o in a["ops"]])
    c.executemany("INSERT INTO caixa(data,origem,destino,valor,motivo) VALUES (?,?,?,?,?)", led)
    c.commit()
