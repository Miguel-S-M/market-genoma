"""Paper trading com preços REAIS e dinheiro de MENTIRA.

1ª execução: cria 20 genomas (16 evoluem, 4 são controle "generalista") e os TREINA com
cerca de 1 ano de histórico real da B3. Depois, cada execução processa só os pregões
novos (o que acontece depois da data de corte é o teste de verdade, ninguém viu antes).
Estado completo em data/estado.json; para o site, gera data/genomas.json.
Regras iguais às do motor.py: decide com dados até ontem, ganha/perde com o retorno de
hoje, e a cada 20 pregões avalia (campo, clonar, hibernar ou eliminar).
Ainda não tem: acordar genomas hibernados quando o regime volta.
"""
import json
import random
import statistics as st
from pathlib import Path

import yfinance as yf
from exportar import ACOES, AGORA, IBOV, regime

DATA = Path(__file__).resolve().parent.parent / "data"
SETOR = {"PETR4": "energia", "VALE3": "mineração", "ITUB4": "financeiro", "BBDC4": "financeiro",
         "WEGE3": "industrial", "MGLU3": "varejo", "B3SA3": "financeiro", "ABEV3": "consumo"}
EQ = {5: "curto", 20: "medio", 60: "longo"}
P = dict(treino=260, epoca=20, pop=16, controle=4, capital=1000.0, caixa0=40000.0, filho=500.0,
         reserva=2000.0, degraus=[100, 300, 900], limite=0.15, custo=0.001)


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
    r = a["rets"]
    if len(r) < 40:
        return 0.0
    s = st.pstdev(r)
    return round(st.mean(r) / s * 15.8, 2) if s else 0.0


def vivo(a):
    return a["status"] in ("treino", "campo")


def mov(est, d, origem, destino, v, motivo):
    est["led"].append({"data": d, "origem": origem, "destino": destino, "valor": round(v, 2), "motivo": motivo})


def novo(est, d, gn, pai, ev, cap):
    ags, ativo = est["ags"], random.choice(ACOES)
    a = dict(id=len(ags) + 1, nome=f"G-{len(ags) + 1:03d}", ativo=ativo, setor=SETOR[ativo],
             equipe=EQ[gn["horizonte_dias"]], ev=ev, status="treino", nasc=d, morte=None, pai_id=pai,
             genes=gn, v0=cap, v=cap, pos=0.0, ent=cap, rets=[], curva=[], ops=[], reg={}, clones=0, ep0=cap)
    ags.append(a)
    est["cx"] -= cap
    mov(est, d, "caixa", a["nome"], cap, "clonagem" if pai else "capital inicial")
    return a


def fim_epoca(est, d, reg):
    ags = est["ags"]
    for a in [x for x in ags if vivo(x)]:
        a["reg"].setdefault(reg, []).append(a["v"] / a["ep0"] - 1)
        a["ep0"] = a["v"]
        if not a["ev"]:
            continue
        if a["status"] == "treino" and rep(a) > 0:
            a["status"] = "campo"
        if a["v"] < a["v0"] * (1 - P["limite"]):
            hib = any(len(x) >= 2 and sum(x) > 0 for x in a["reg"].values())
            a["status"], a["morte"] = ("hibernado" if hib else "eliminado"), d
            est["cx"] += a["v"]
            mov(est, d, a["nome"], "caixa", a["v"], "hibernação" if hib else "desligamento")
        elif (a["clones"] < len(P["degraus"]) and a["v"] - a["v0"] >= P["degraus"][a["clones"]]
              and est["cx"] >= P["filho"] + P["reserva"]):
            novo(est, d, mutar(a["genes"]), a["id"], True, P["filho"])
            a["clones"] += 1
    while sum(1 for x in ags if x["ev"] and vivo(x)) < 8 and est["cx"] >= P["capital"] + P["reserva"]:
        novo(est, d, genes(), None, True, P["capital"])


def dia(est, arr, datas, ibov, k):
    d = datas[k]
    for a in [x for x in est["ags"] if vivo(x)]:
        s, h = arr[a["ativo"]], a["genes"]["horizonte_dias"]
        r = float(s[k] / s[k - 1] - 1)
        a["v"] *= 1 + a["pos"] * r                   # posição decidida ontem
        a["rets"].append(a["pos"] * r)
        mom = float(s[k] / s[k - h] - 1)
        nova = a["genes"]["risco"] if mom * a["genes"]["tendencia"] > 0 else 0.0
        if (nova > 0) != (a["pos"] > 0):             # abriu ou fechou posição
            a["v"] *= 1 - P["custo"]
            p = round(float(s[k]), 2)
            q = max(1, int(a["v"] * max(nova, a["pos"]) / p))
            op = {"data": d, "ativo": a["ativo"], "qtd": q, "preco": p}
            if nova > 0:
                a["ent"] = a["v"]
                a["ops"].append({**op, "lado": "compra", "resultado": 0.0})
            else:
                a["ops"].append({**op, "lado": "venda", "resultado": round(a["v"] - a["ent"], 2)})
        a["pos"] = nova
        a["curva"].append(round(a["v"], 2))
    est["dias_ep"] += 1
    if est["dias_ep"] >= P["epoca"]:
        est["dias_ep"] = 0
        fim_epoca(est, d, regime(ibov, k))


def exportar(est):
    G = []
    for a in est["ags"]:
        r = a["reg"]
        melhor = max(r, key=lambda k: sum(r[k]) / len(r[k])) if r else "sem dados"
        cv = a["curva"][-120:]
        if len(cv) < 2:
            cv = [a["v0"], round(a["v"], 2)]
        G.append({"id": a["id"], "nome": a["nome"], "equipe": a["equipe"], "setor": a["setor"],
                  "status": a["status"], "nascimento": a["nasc"], "morte": a["morte"], "pai_id": a["pai_id"],
                  "clonagens": a["clones"], "genes": a["genes"], "capital_inicial": a["v0"],
                  "capital_atual": round(a["v"], 2), "melhor_regime": melhor, "reputacao": rep(a),
                  "curva": cv, "operacoes": a["ops"][-30:][::-1]})
    saida = {"atualizado": AGORA, "ultimo_pregao": est["ultimo"], "G": G, "caixa": est["led"]}
    (DATA / "genomas.json").write_text(json.dumps(saida, ensure_ascii=False), encoding="utf-8")
    (DATA / "estado.json").write_text(json.dumps(est, ensure_ascii=False), encoding="utf-8")


def main():
    tk = [a + ".SA" for a in ACOES]
    px = yf.download(tk + [IBOV], period="2y", interval="1d", auto_adjust=True, progress=False)["Close"]
    px = px.dropna(subset=tk).ffill()
    datas = [d.strftime("%Y-%m-%d") for d in px.index]
    arr = {a: px[a + ".SA"].to_numpy() for a in ACOES}
    f = DATA / "estado.json"
    if f.exists():
        est = json.loads(f.read_text(encoding="utf-8"))
    else:  # primeira vez: cria a população e treina com o histórico real
        random.seed(7)
        est = {"ultimo": datas[-P["treino"] - 1], "ags": [], "led": [], "cx": P["caixa0"], "dias_ep": 0}
        mov(est, est["ultimo"], "externo", "caixa", P["caixa0"], "aporte inicial (simulado)")
        for i in range(P["pop"] + P["controle"]):
            a = novo(est, est["ultimo"], genes(), None, i < P["pop"], P["capital"])
            if i >= P["pop"]:
                a["equipe"] = "generalista"
    novos = [k for k, d in enumerate(datas) if d > est["ultimo"]]
    if not novos:
        print("Sem pregões novos.")
        return
    random.seed(novos[-1])
    for k in novos:
        dia(est, arr, datas, px[IBOV], k)
    est["ultimo"] = datas[-1]
    exportar(est)
    print("Processados", len(novos), "pregões; último:", est["ultimo"])


if __name__ == "__main__":
    main()
