"""Paper trading v2: preços REAIS, dinheiro de MENTIRA, várias classes de ativos.

Classes: ações, imobiliário (FIIs), cripto, moedas e exterior. Cada genoma é preso a UM ativo.
Rotina diária: o genoma decide no FECHAMENTO do dia k-1 e a ordem sai na ABERTURA do dia k
(ações, FIIs e exterior). Cripto e moedas negociam 24h: a ordem sai no preço do momento da
decisão. Só entram pregões encerrados (candle de hoje é ignorado até 18h30, Brasília).

Hibernação e reativação POR REGIME, usando só o histórico do próprio genoma:
- cada dia o retorno do genoma é somado ao regime de mercado confirmado (3 dias seguidos);
- "bom no regime X" = pelo menos 15 dias em X e retorno acumulado >= 2%;
- o mercado vira para um regime em que o genoma é sabidamente ruim, e ele é bom em outro:
  hiberna (vende tudo, o dinheiro volta ao caixa, o genoma é guardado);
- o regime confirmado volta a ser um em que ele é bom: reativa com capital do caixa;
- queda além do limite: hiberna se tem algum regime bom, senão é eliminado. Quem tem
  reputação alta tolera uma queda maior (até 2x o limite).
Ainda não tem: empréstimos, teto 25/65/10, prazos, hierarquia ativa, cruzamento de genes.
"""
import json
import random
import statistics as st
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yfinance as yf
from exportar import AGORA, IBOV, baixar, regime

DATA = Path(__file__).resolve().parent.parent / "data"
USD = "USDBRL=X"
ATIVOS = {  # nome: ticker do Yahoo, classe, setor, em_usd (converter para reais)
    "PETR4": dict(t="PETR4.SA", classe="acoes", setor="energia"),
    "VALE3": dict(t="VALE3.SA", classe="acoes", setor="mineração"),
    "ITUB4": dict(t="ITUB4.SA", classe="acoes", setor="financeiro"),
    "BBDC4": dict(t="BBDC4.SA", classe="acoes", setor="financeiro"),
    "WEGE3": dict(t="WEGE3.SA", classe="acoes", setor="industrial"),
    "MGLU3": dict(t="MGLU3.SA", classe="acoes", setor="varejo"),
    "B3SA3": dict(t="B3SA3.SA", classe="acoes", setor="financeiro"),
    "ABEV3": dict(t="ABEV3.SA", classe="acoes", setor="consumo"),
    "HGLG11": dict(t="HGLG11.SA", classe="imobiliario", setor="imobiliário"),
    "KNRI11": dict(t="KNRI11.SA", classe="imobiliario", setor="imobiliário"),
    "MXRF11": dict(t="MXRF11.SA", classe="imobiliario", setor="imobiliário"),
    "BTC": dict(t="BTC-USD", classe="cripto", setor="cripto", usd=True),
    "ETH": dict(t="ETH-USD", classe="cripto", setor="cripto", usd=True),
    "USD": dict(t=USD, classe="moedas", setor="moeda"),
    "EUR": dict(t="EURBRL=X", classe="moedas", setor="moeda"),
    "IVVB11": dict(t="IVVB11.SA", classe="exterior", setor="exterior"),
}
H24 = {"cripto", "moedas"}      # negociam 24h: a ordem sai no preço do momento da decisão
DISP = []                       # ativos com dados suficientes (preenchido em main)
EQ = {5: "curto", 20: "medio", 60: "longo"}
P = dict(treino=260, epoca=20, pop=16, controle=4, capital=1000.0, caixa0=40000.0, filho=500.0,
         reserva=2000.0, degraus=[100, 300, 900], limite=0.15,
         custo=dict(acoes=0.001, imobiliario=0.001, exterior=0.001, moedas=0.002, cripto=0.003),
         confirma=3, min_dias=15, min_ret=0.02, espera=10, max_reat=3)


def hoje_fechado():
    agora = datetime.now(ZoneInfo("America/Sao_Paulo"))
    return agora.strftime("%Y-%m-%d"), agora.hour * 60 + agora.minute >= 18 * 60 + 30


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


def custo(a):
    return P["custo"][a["classe"]]


def mov(est, d, origem, destino, v, motivo):
    est["led"].append({"data": d, "origem": origem, "destino": destino, "valor": round(v, 2), "motivo": motivo})


def menos_povoada(est, ev):
    cont = {c: 0 for c in {ATIVOS[n]["classe"] for n in DISP}}
    for a in est["ags"]:
        if vivo(a) and a["ev"] == ev and a["classe"] in cont:
            cont[a["classe"]] += 1
    m = min(cont.values())
    return random.choice([c for c, v in cont.items() if v == m])


def novo(est, d, gn, pai, ev, cap, ativo=None):
    ags = est["ags"]
    if ativo is None:
        cl = menos_povoada(est, ev)
        ativo = random.choice([n for n in DISP if ATIVOS[n]["classe"] == cl])
    inf = ATIVOS[ativo]
    a = dict(id=len(ags) + 1, nome=f"G-{len(ags) + 1:03d}", ativo=ativo, classe=inf["classe"],
             setor=inf["setor"], equipe=EQ[gn["horizonte_dias"]], ev=ev, status="treino", nasc=d,
             morte=None, pai_id=pai, genes=gn, v0=cap, v=cap, pos=0.0, alvo=0.0, ent=cap, px=0.0,
             rets=[], curva=[], ops=[], rd={}, clones=0, v_hib=0.0, hib_dias=0, reat=0)
    ags.append(a)
    est["cx"] -= cap
    mov(est, d, "caixa", a["nome"], cap, "clonagem" if pai else "capital inicial")
    return a


def ativo_filho(a):
    if random.random() < 0.15:  # mutação rara: troca de ativo dentro da mesma classe
        return random.choice([n for n in DISP if ATIVOS[n]["classe"] == a["classe"]])
    return a["ativo"]


# ---- regimes: histórico do PRÓPRIO genoma, sem comparar com os colegas ----
def bons_regimes(a):
    return {r: v[0] / v[1] for r, v in a["rd"].items() if v[1] >= P["min_dias"] and v[0] >= P["min_ret"]}


def ruim_em(a, r):
    v = a["rd"].get(r)
    return bool(v) and v[1] >= P["min_dias"] and v[0] < 0


def confirmar_regime(est, bruto):
    """O analista de mercado só muda o regime depois de P['confirma'] dias seguidos no novo."""
    r = est.setdefault("regime", {"atual": bruto, "cand": None, "dias": 0})
    mudou = False
    if bruto == r["atual"]:
        r["cand"], r["dias"] = None, 0
    else:
        r["dias"] = r["dias"] + 1 if bruto == r["cand"] else 1
        r["cand"] = bruto
        if r["dias"] >= P["confirma"]:
            r["atual"], r["cand"], r["dias"], mudou = bruto, None, 0, True
    return r["atual"], mudou


def encerrar(est, d, a, status, motivo):
    """Vende tudo (custo de saída) e devolve o dinheiro ao caixa."""
    if a["pos"] > 0:
        a["v"] *= 1 - custo(a)
        q = max(1, int(a["v"] * a["pos"] / a["px"])) if a["px"] else 1
        a["ops"].append({"data": d, "ativo": a["ativo"], "qtd": q, "preco": round(a["px"], 2),
                         "lado": "venda", "resultado": round(a["v"] - a["ent"], 2)})
    a["pos"] = a["alvo"] = 0.0
    a["status"], a["morte"] = status, d
    if status == "hibernado":
        a["v_hib"], a["hib_dias"] = a["v"], 0
    est["cx"] += a["v"]
    mov(est, d, a["nome"], "caixa", a["v"], motivo)


def reativar(est, d, reg):
    cand = [a for a in est["ags"] if a["status"] == "hibernado" and a["ev"]
            and a["hib_dias"] >= P["espera"] and a["reat"] < P["max_reat"] and reg in bons_regimes(a)]
    cand.sort(key=lambda a: -bons_regimes(a)[reg])
    for a in cand:
        cap = max(P["filho"], min(a["v_hib"], P["capital"]))
        if est["cx"] < cap + P["reserva"]:
            break
        est["cx"] -= cap
        mov(est, d, "caixa", a["nome"], cap, f"reativação (mercado em {reg})")
        a.update(status="campo", morte=None, v=cap, v0=cap, pos=0.0, alvo=0.0, ent=cap, hib_dias=0)
        a["reat"] += 1


def fim_epoca(est, d, reg):
    ags = est["ags"]
    for a in [x for x in ags if vivo(x) and x["ev"]]:
        if a["status"] == "treino" and rep(a) > 0:
            a["status"] = "campo"
        lim = P["limite"] * (1 + min(max(rep(a), 0), 2) / 2)   # reputação alta tolera mais erro
        if a["v"] < a["v0"] * (1 - lim):
            if bons_regimes(a):
                encerrar(est, d, a, "hibernado", f"hibernação (queda em {reg}, tem regime bom)")
            else:
                encerrar(est, d, a, "eliminado", "desligamento (sem regime bom)")
        elif (a["clones"] < len(P["degraus"]) and a["v"] - a["v0"] >= P["degraus"][a["clones"]]
              and est["cx"] >= P["filho"] + P["reserva"]):
            novo(est, d, mutar(a["genes"]), a["id"], True, P["filho"], ativo_filho(a))
            a["clones"] += 1
    while sum(1 for x in ags if x["ev"] and vivo(x)) < 8 and est["cx"] >= P["capital"] + P["reserva"]:
        novo(est, d, genes(), None, True, P["capital"])


def dia(est, arr, opn, datas, ibov, k):
    d = datas[k]
    reg, mudou = confirmar_regime(est, regime(ibov, k))
    for a in est["ags"]:
        if a["status"] == "hibernado":
            a["hib_dias"] += 1
    for a in [x for x in est["ags"] if vivo(x) and x["ativo"] in arr]:
        s, o, h = arr[a["ativo"]], opn[a["ativo"]], a["genes"]["horizonte_dias"]
        pos, alvo, v_ini = a["pos"], a["alvo"], a["v"]
        a["v"] *= 1 + pos * float(o[k] / s[k - 1] - 1)           # até a abertura: posição antiga
        if (alvo > 0) != (pos > 0):                              # abriu ou fechou posição
            a["v"] *= 1 - custo(a)
            p = round(float(o[k]), 4 if a["classe"] in ("moedas",) else 2)
            q = max(1, int(a["v"] * max(alvo, pos) / p))
            op = {"data": d, "ativo": a["ativo"], "qtd": q, "preco": p}
            if alvo > 0:
                a["ent"] = a["v"]
                a["ops"].append({**op, "lado": "compra", "resultado": 0.0})
            else:
                a["ops"].append({**op, "lado": "venda", "resultado": round(a["v"] - a["ent"], 2)})
        a["v"] *= 1 + alvo * float(s[k] / o[k] - 1)             # da abertura ao fechamento
        ret = a["v"] / v_ini - 1
        a["rets"].append(ret)
        x = a["rd"].setdefault(reg, [0.0, 0])                    # histórico do genoma por regime
        x[0] += ret
        x[1] += 1
        a["pos"], a["px"] = alvo, float(s[k])
        mom = float(s[k] / s[k - h] - 1)                         # decisão de amanhã
        a["alvo"] = a["genes"]["risco"] if mom * a["genes"]["tendencia"] > 0 else 0.0
        a["curva"].append(round(a["v"], 2))
    if mudou:   # o mercado virou: quem é sabidamente ruim neste regime (e bom em outro) hiberna
        for a in [x for x in est["ags"] if x["status"] == "campo" and x["ev"]]:
            if ruim_em(a, reg) and bons_regimes(a):
                encerrar(est, d, a, "hibernado", f"hibernação (mercado virou para {reg})")
    reativar(est, d, reg)
    total = est["cx"] + sum(x["v"] for x in est["ags"] if vivo(x))
    est.setdefault("serie", []).append({"data": d, "lab": round(total, 2), "ibov": round(float(ibov.iloc[k]), 2)})
    est["dias_ep"] += 1
    if est["dias_ep"] >= P["epoca"]:
        est["dias_ep"] = 0
        fim_epoca(est, d, reg)


def cdi_acumulado(datas):
    """CDI acumulado (base 1) para cada data da série, via Banco Central (SGS 12, % ao dia)."""
    try:
        d0 = datetime.strptime(datas[0], "%Y-%m-%d").strftime("%d/%m/%Y")
        d1 = datetime.strptime(datas[-1], "%Y-%m-%d").strftime("%d/%m/%Y")
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados?formato=json&dataInicial={d0}&dataFinal={d1}"
        taxa = {datetime.strptime(x["data"], "%d/%m/%Y").strftime("%Y-%m-%d"): float(x["valor"]) / 100
                for x in json.loads(baixar(url))}
    except Exception as e:
        print("CDI falhou:", e)
        return [None] * len(datas)
    f, out = 1.0, []
    for i, d in enumerate(datas):
        if i:
            for dd, t in taxa.items():
                if datas[i - 1] < dd <= d:
                    f *= 1 + t
        out.append(round(f, 6))
    return out


def exportar(est):
    G = []
    for a in est["ags"]:
        rd = a["rd"]
        bons = bons_regimes(a)
        base = {r: v[0] / v[1] for r, v in rd.items() if v[1] >= 5}
        melhor = max(base, key=base.get) if base else "sem dados"
        cv = a["curva"][-120:]
        if len(cv) < 2:
            cv = [a["v0"], round(a["v"], 2)]
        G.append({"id": a["id"], "nome": a["nome"], "equipe": a["equipe"], "setor": a["setor"],
                  "classe": a["classe"], "ativo": a["ativo"], "status": a["status"],
                  "nascimento": a["nasc"], "morte": a["morte"], "pai_id": a["pai_id"],
                  "clonagens": a["clones"], "reativacoes": a["reat"], "genes": a["genes"],
                  "capital_inicial": a["v0"], "capital_atual": round(a["v"], 2),
                  "melhor_regime": melhor, "regimes_bons": sorted(bons), "reputacao": rep(a),
                  "curva": cv, "operacoes": a["ops"][-30:][::-1]})
    saida = {"atualizado": AGORA, "ultimo_pregao": est["ultimo"], "G": G, "caixa": est["led"],
             "regime_atual": est.get("regime", {}).get("atual")}
    if est.get("serie"):
        cdi = cdi_acumulado([x["data"] for x in est["serie"]])
        saida["benchmark"] = {"corte": est.get("corte"), "serie": [{**x, "cdi": c} for x, c in zip(est["serie"], cdi)]}
    (DATA / "genomas.json").write_text(json.dumps(saida, ensure_ascii=False), encoding="utf-8")
    (DATA / "estado.json").write_text(json.dumps(est, ensure_ascii=False), encoding="utf-8")


def carregar():
    """Baixa os preços e alinha tudo ao calendário da B3 (dias em que o Ibovespa operou)."""
    tks = sorted({v["t"] for v in ATIVOS.values()} | {IBOV, USD})
    bruto = yf.download(tks, period="2y", interval="1d", auto_adjust=True, progress=False)
    C, O = bruto["Close"], bruto["Open"]
    cal = C[IBOV].dropna().index
    hoje, fechado = hoje_fechado()
    if not fechado and cal[-1].strftime("%Y-%m-%d") >= hoje:
        cal = cal[:-1]                               # candle de hoje ainda em andamento
    C, O = C.reindex(cal).ffill(), O.reindex(cal).ffill()
    O = O.where(O.notna(), C)
    arr, opn = {}, {}
    for nome, inf in ATIVOS.items():
        if inf["t"] not in C.columns:
            print("sem dados:", nome)
            continue
        c = C[inf["t"]] * (C[USD] if inf.get("usd") else 1)
        if c.iloc[-(P["treino"] + 61):].isna().any():
            print("histórico insuficiente:", nome)
            continue
        o = c.shift(1).bfill() if inf["classe"] in H24 else O[inf["t"]]
        arr[nome], opn[nome] = c.to_numpy(), o.to_numpy()
    return C[IBOV], [d.strftime("%Y-%m-%d") for d in cal], arr, opn


def main():
    global DISP
    ibov, datas, arr, opn = carregar()
    DISP = sorted(arr)
    print("ativos disponíveis:", ", ".join(DISP))
    f = DATA / "estado.json"
    if f.exists():
        est = json.loads(f.read_text(encoding="utf-8"))
    else:  # primeira vez: cria a população e treina com o histórico real
        random.seed(7)
        est = {"ultimo": datas[-P["treino"] - 1], "ags": [], "led": [], "cx": P["caixa0"], "dias_ep": 0,
               "corte": datas[-1]}   # a partir daqui é teste de verdade; antes é treino retroativo
        mov(est, est["ultimo"], "externo", "caixa", P["caixa0"], "aporte inicial (simulado)")
        for i in range(P["pop"] + P["controle"]):
            novo(est, est["ultimo"], genes(), None, i < P["pop"], P["capital"])
            if i >= P["pop"]:
                est["ags"][-1]["equipe"] = "generalista"
    if "serie" not in est:
        i0 = datas.index(est["ultimo"])
        est["serie"] = [{"data": est["ultimo"], "lab": P["caixa0"], "ibov": round(float(ibov.iloc[i0]), 2)}]
    novos = [k for k, d in enumerate(datas) if d > est["ultimo"]]
    if not novos:
        print("Sem pregões novos.")
        return
    random.seed(novos[-1])
    for k in novos:
        dia(est, arr, opn, datas, ibov, k)
    est["ultimo"] = datas[-1]
    exportar(est)
    print("Processados", len(novos), "pregões; último:", est["ultimo"])


if __name__ == "__main__":
    main()
