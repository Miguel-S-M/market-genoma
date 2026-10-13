"""Paper trading v4.0 (base v3.1 + carteira com vários ativos por genoma): preços REAIS, dinheiro de MENTIRA, desenho completo do laboratório.

CADA RECURSO TEM UMA CHAVE em P["rec"] (True/False). Desligue um para medir sem ele.

Base (v2.1): 16 ativos em 5 classes; decisão no fechamento, ordem na abertura seguinte
(cripto e moedas: no preço do momento); caixa parado rende CDI; hibernação e reativação por
regime usando só o histórico do próprio genoma; limite de perda maior para quem tem reputação.

Rodando com a variável LAB_BASE=1, o script vira o laboratório BASE (data/base/): mesmas regras da
v2.1, camadas novas desligadas, para comparar se as camadas ajudam.

v4.0 - CARTEIRA DIVERSIFICADA: cada genoma tem um UNIVERSO de 2 a 5 ativos (gene n_ativos, de classes
diferentes) e mantém posições em vários deles ao mesmo tempo. Os pesos somam no máximo o gene de
risco (limitado pelas ordens da hierarquia), cada ativo tem teto de peso, e o custo de operação é
cobrado sobre o giro da carteira. Treino e Campo são separados: o fórum de um nunca é lido pelo
outro, e o Campo só começa com R$ 200 por genoma promovido.

Novos recursos (v3.0):
- hierarquia: 3 analistas de mercado votam o regime; o arquiteto de mercado (que NÃO vê preços,
  só os votos) decide; o arquiteto de operações define a exposição geral; arquitetos de
  horizonte (curto/médio/longo) e de mercado específico (por classe) refinam. Só emitem
  ORDENS DE MERCADO (teto de exposição); o genoma continua decidindo o que comprar. Genomas
  de baixa reputação obedecem ao teto; os de alta reputação podem passar até 25% dele. O
  fórum (média dos sinais ponderada por reputação) influencia o arquiteto de operações.
  O grupo generalista (controle) não recebe ordens.
- emprestimo: genoma com reputação alta pega até 5% do caixa geral ao abrir uma posição que
  considera lucrativa (regime atual é um de seus regimes bons); paga com juros de CDI ao fechar.
- teto: acima de R$ 5.000 o excedente se divide em 25% fundo de descendentes, 65% caixa geral
  e 10% banco de empréstimo do próprio genoma (que ele usa ao abrir posições).
- resgate: em queda brusca do Ibovespa, os piores genomas vendem e transferem parte do capital
  ao melhor até ele atingir o capital mínimo seguro.
- prazos: o genoma tem um gene de prazo (dias mínimos na posição); o prazo máximo permitido
  cresce com a reputação; stop de 10% interrompe o prazo.
- cruzamento: se os filhos de um genoma pioraram, o próximo filho cruza os genes com o melhor.
- medianos: genomas medianos também clonam (capital menor, no máximo 2 vezes).
- migracao: genomas novos só entram em ações, FIIs e exterior; quem prova reputação pode
  migrar para cripto e moedas (volta ao treino na nova classe).
- noticias: gene de peso das notícias; usa o sentimento por classe só nos pregões em tempo real
  (no treino retroativo não existe histórico de notícias, o sentimento vale 0).
- colmeia: carteira virtual que agrega os sinais dos genomas de campo ponderados por reputação.
  Só lê; não altera nenhum gene.
"""
import json
import os
import random
import statistics as st
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import yfinance as yf
from exportar import AGORA, IBOV, baixar, regime

DATA = Path(__file__).resolve().parent.parent / "data"
VERSAO = "4.0"
USD = "USDBRL=X"
ATIVOS = {
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
NOMES_CLASSE = {"acoes": "ações", "imobiliario": "imobiliário"}
H24 = {"cripto", "moedas"}
SEGURAS = {"acoes", "imobiliario", "exterior"}
RISCO = {"cripto", "moedas"}
CDI = {}
SENT = {}
DISP = []
EQ = {5: "curto", 20: "medio", 60: "longo"}
REGIMES = ["alta calma", "alta volátil", "lateral", "queda", "crise"]
P = dict(
    treino=260, epoca=20, pop=16, controle=4, capital=1000.0, capital_treino=1000.0, capital_campo=200.0, caixa0=5000.0, filho=500.0,
    reserva=500.0, prom_dias=20, prom_rep=0.5, forum_escala=0.05, max_peso=0.5, min_trade=0.02, emp_dias=20, forum_peso_classe=0.5, degraus=[100, 300, 900], limite=0.15,
    custo=dict(acoes=0.001, imobiliario=0.001, exterior=0.001, moedas=0.002, cripto=0.003),
    confirma=3, min_dias=15, min_ret=0.02, espera=10, max_reat=3,
    rec=dict(hierarquia=True, emprestimo=True, teto=True, resgate=True, prazos=True,
             cruzamento=True, medianos=True, migracao=False, noticias=True, colmeia=True, forum=True),
    emp_pct=0.05, rep_emp=1.0, emp_max_x=1.0,
    teto=600.0, teto_div=(0.25, 0.65, 0.10),
    resgate_1d=-0.03, resgate_5d=-0.06, capital_seguro=300.0, resgate_frac=0.5, resgate_espera=10,
    prazo_base=5, prazo_rep=10, prazo_max=60, stop=0.10,
    med_lucro=0.03, filho_med=300.0, max_vivos=60,
    rep_migra=0.5, p_migra=0.3, peso_noticia_escala=0.02, vol_alvo=0.25,
    exp_regime={"alta calma": 1.0, "alta volátil": 0.8, "lateral": 0.7, "queda": 0.5, "crise": 0.25},
    fator_horizonte={"alta calma": dict(curto=0.8, medio=1.0, longo=1.0),
                     "alta volátil": dict(curto=0.5, medio=0.8, longo=1.0),
                     "lateral": dict(curto=0.6, medio=0.7, longo=0.8),
                     "queda": dict(curto=0.4, medio=0.5, longo=0.6),
                     "crise": dict(curto=0.2, medio=0.25, longo=0.3)},
)


DATA_RAIZ = DATA
if os.environ.get("LAB_BASE") == "1":   # laboratório BASE: mesmos dados, todas as camadas novas desligadas
    for _k in P["rec"]:
        P["rec"][_k] = False
    DATA = DATA / "base"
    DATA.mkdir(exist_ok=True)


def rec(nome):
    return P["rec"].get(nome, False)


def hoje_fechado():
    agora = datetime.now(ZoneInfo("America/Sao_Paulo"))
    return agora.strftime("%Y-%m-%d"), agora.hour * 60 + agora.minute >= 18 * 60 + 30


# ---------------------------------------------------------------- genes e reputação
def genes():
    return {"tendencia": round(random.uniform(-1, 1), 2), "risco": round(random.uniform(0.2, 1), 2),
            "horizonte_dias": random.choice([5, 20, 60]), "prazo": random.choice([0, 3, 5, 10, 20]),
            "peso_noticias": round(random.uniform(0, 1), 2),
            "peso_forum": round(random.uniform(0, 1), 2),
            "n_ativos": random.choice([2, 3, 4, 5])}


def mutar(g):
    f = dict(g)
    f["tendencia"] = round(max(-1, min(1, f["tendencia"] + random.gauss(0, 0.15))), 2)
    f["risco"] = round(max(0.1, min(1, f["risco"] + random.gauss(0, 0.1))), 2)
    if random.random() < 0.2:
        f["horizonte_dias"] = random.choice([5, 20, 60])
    if random.random() < 0.3:
        f["prazo"] = random.choice([0, 3, 5, 10, 20])
    f["peso_noticias"] = round(max(0, min(1, f.get("peso_noticias", 0.5) + random.gauss(0, 0.1))), 2)
    f["peso_forum"] = round(max(0, min(1, f.get("peso_forum", 0.5) + random.gauss(0, 0.1))), 2)
    if random.random() < 0.2:
        f["n_ativos"] = max(2, min(5, f.get("n_ativos", 3) + random.choice([-1, 1])))
    return f


def cruzar(g1, g2):
    return {k: random.choice([g1[k], g2.get(k, g1[k])]) for k in g1}


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


def prazo_efetivo(a):
    if not rec("prazos"):
        return 0
    pmax = min(P["prazo_max"], P["prazo_base"] + P["prazo_rep"] * max(rep(a), 0))
    return int(min(a["genes"].get("prazo", 0), pmax))


def mov(est, d, origem, destino, v, motivo, cart="campo"):
    est["led"].append({"data": d, "origem": origem, "destino": destino, "valor": round(v, 2), "motivo": motivo,
                       "carteira": cart})


# ---------------------------------------------------------------- população
def menos_povoada(est, ev):
    classes = {ATIVOS[n]["classe"] for n in DISP}
    if ev and rec("migracao"):
        classes &= SEGURAS
    cont = {c: 0 for c in classes}
    for a in est["ags"]:
        if vivo(a) and a["ev"] == ev and a["classe"] in cont:
            cont[a["classe"]] += 1
    m = min(cont.values())
    return random.choice([c for c, v in cont.items() if v == m])


def novo(est, d, gn, pai, ev, cap, universo=None, de_fundo=None, motivo=None):
    ags = est["ags"]
    univ = montar_universo(est, ev, gn.get("n_ativos", 3), universo)
    dom, setor = perfil(univ)
    a = dict(id=len(ags) + 1, nome=f"G-{len(ags) + 1:03d}", universo=univ, ativo=", ".join(univ),
             classe=dom, setor=setor, equipe=EQ[gn["horizonte_dias"]], ev=ev, status="treino", nasc=d,
             morte=None, pai_id=pai, genes=gn, v0=cap, v=cap, pos={}, alvo={}, ent=cap, px={}, ent_px={},
             rets=[], curva=[], ops=[], rd={}, clones=0, clones_med=0, v_hib=0.0, hib_dias=0, reat=0,
             cap_in=cap, cap_out=0.0, fundo=0.0, banco=0.0, banco_usado=0.0, divida=0.0, juros=0.0,
             dias_pos={}, div_dias=0, migrou=False, rets_f=[], pts_treino=0.0)
    ags.append(a)
    if de_fundo and de_fundo[0] is not None:
        pai_a, usado = de_fundo
        pai_a["fundo"] -= usado
        mov(est, d, f"fundo {pai_a['nome']}", a["nome"], usado, "clonagem (fundo de descendentes)")
        if cap - usado > 0.005:
            est["cx"] -= cap - usado
            mov(est, d, "caixa", a["nome"], cap - usado, motivo or "clonagem")
    else:   # todo genoma nasce no TREINO, com dinheiro fictício que não sai do caixa de campo
        mov(est, d, "treino (fictício)", a["nome"], cap, motivo or ("clonagem" if pai else "capital inicial"), "treino")
    return a


def perfil(univ):
    """Classe dominante e rótulo de setor de um universo de ativos."""
    cls = [ATIVOS[n]["classe"] for n in univ]
    dom = Counter(cls).most_common(1)[0][0]
    return dom, ("diversificado: " + "+".join(sorted(set(cls))) if len(set(cls)) > 1 else dom)


def montar_universo(est, ev, n, base=None):
    """Escolhe n ativos de classes diferentes sempre que possível; o primeiro equilibra a população."""
    pool = [x for x in DISP if not (ev and rec("migracao") and ATIVOS[x]["classe"] not in SEGURAS)]
    univ = [x for x in (base or []) if x in DISP]
    if not univ:
        cl = menos_povoada(est, ev)
        univ.append(random.choice([x for x in pool if ATIVOS[x]["classe"] == cl]))
    while len(univ) < n:
        usadas = {ATIVOS[x]["classe"] for x in univ}
        livres = [x for x in pool if x not in univ]
        novas = [x for x in livres if ATIVOS[x]["classe"] not in usadas] or livres
        univ.append(random.choice(novas))
    while len(univ) > n:
        univ.pop(random.randrange(1, len(univ)))
    return univ


# ---------------------------------------------------------------- regimes (histórico do próprio genoma)
def bons_regimes(a):
    return {r: v[0] / v[1] for r, v in a["rd"].items() if v[1] >= P["min_dias"] and v[0] >= P["min_ret"]}


def ruim_em(a, r):
    v = a["rd"].get(r)
    return bool(v) and v[1] >= P["min_dias"] and v[0] < 0


def confirmar_regime(est, bruto):
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


# ---------------------------------------------------------------- hierarquia
def regime_b(c, k):
    """Analista 2: tendência curta (média de 100 dias, retorno e volatilidade de 10 dias)."""
    s = c.iloc[: k + 1]
    if len(s) < 101:
        return "lateral"
    ma, ret = s.tail(100).mean(), s.iloc[-1] / s.iloc[-11] - 1
    vol = float(s.pct_change().tail(10).std() * np.sqrt(252))
    if vol > 0.40 and ret < -0.05:
        return "crise"
    if s.iloc[-1] < ma and ret < 0:
        return "queda"
    if s.iloc[-1] > ma and ret > 0:
        return "alta volátil" if vol > 0.28 else "alta calma"
    return "lateral"


def regime_c(arr, ibov, k):
    """Analista 3: amplitude (quantas ações estão acima da média de 50 dias)."""
    nomes = [n for n in arr if ATIVOS[n]["classe"] == "acoes"]
    if k < 50 or not nomes:
        return "lateral"
    acima = np.mean([arr[n][k] > np.mean(arr[n][k - 49: k + 1]) for n in nomes])
    s = ibov.iloc[: k + 1]
    ret = s.iloc[-1] / s.iloc[-21] - 1
    vol = float(s.pct_change().tail(20).std() * np.sqrt(252))
    if acima <= 0.35:
        return "crise" if vol > 0.35 and ret < -0.08 else "queda"
    if acima >= 0.6 and ret > 0:
        return "alta volátil" if vol > 0.25 else "alta calma"
    return "lateral"


def vol_classe(arr, k):
    out = {}
    for cl in {ATIVOS[n]["classe"] for n in arr}:
        v = []
        for n in arr:
            if ATIVOS[n]["classe"] == cl and k >= 21:
                r = np.diff(np.log(arr[n][k - 20: k + 1]))
                v.append(float(np.std(r) * np.sqrt(252)))
        out[cl] = float(np.mean(v)) if v else P["vol_alvo"]
    return out


def forum(est):
    """Média dos sinais (posição de ontem) dos genomas de campo, ponderada pela reputação."""
    num = den = 0.0
    for a in est["ags"]:
        if a["status"] == "campo" and a["ev"]:
            w = max(rep(a), 0.05)
            num += w * (1.0 if sum(a["alvo"].values()) > 0.02 else 0.0)
            den += w
    return round(num / den, 3) if den else 0.5


def postar(est, d, a, ativo, direcao):
    est["mid"] = est.get("mid", 0) + 1
    forca = round(min(1.0, max(rep(a), 0) / 2), 2)
    cl = ATIVOS[ativo]["classe"]
    est.setdefault("msgs", []).append({
        "id": est["mid"], "data": d, "autor": a["nome"], "autor_id": a["id"], "carteira": a["status"],
        "classe": cl, "ativo": ativo, "dir": direcao, "forca": forca,
        "texto": f"{a['nome']} ({a['equipe']}, reputação {rep(a):.2f}) {'comprou' if direcao > 0 else 'saiu de'} "
                 f"{ativo}: sinal de {'alta' if direcao > 0 else 'cautela'} para {NOMES_CLASSE.get(cl, cl)}.",
        "vistos": [], "seguiram": [], "impactados": [], "ignoraram": [], "impacto_rs": 0.0})


def forum_efeito(est, a, vis, sc, decide):
    """Decide com e sem as mensagens do dia anterior (só da mesma carteira); a diferença é o impacto, medido em R$ depois."""
    sem = decide(sc)
    if not vis:
        return sem
    pf = a["genes"].get("peso_forum", 0.5)
    adj, contrib = {n: 0.0 for n in sc}, []
    for m in vis:
        for n in sc:
            if m["ativo"] == n:
                f = 1.0
            elif ATIVOS[m["ativo"]]["classe"] == ATIVOS[n]["classe"]:
                f = P["forum_peso_classe"]
            else:
                continue
            c = pf * P["forum_escala"] * m["dir"] * m["forca"] * f
            adj[n] += c
            contrib.append((m, n, c))
    com = decide({n: sc[n] + adj[n] for n in sc})
    deltas = {n: round(com.get(n, 0.0) - sem.get(n, 0.0), 4) for n in sc}
    por_msg = {}
    for m, n, c in contrib:
        por_msg.setdefault(m["id"], [m, 0.0, 0.0])
        por_msg[m["id"]][1] += abs(c)
        por_msg[m["id"]][2] += deltas[n] if c * deltas[n] > 0 else 0.0
    tot = sum(v[1] for v in por_msg.values()) or 1.0
    partes = []
    for m in vis:
        m["vistos"].append(a["id"])
        if m["id"] not in por_msg or por_msg[m["id"]][1] < 1e-9:
            m["ignoraram"].append({"id": a["id"], "motivo": "peso do genoma baixo"})
            continue
        seguiu = com.get(m["ativo"], 0.0) > 0 if m["dir"] > 0 else (m["ativo"] in sc and com.get(m["ativo"], 0.0) == 0)
        if seguiu:
            m["seguiram"].append(a["id"])
            dl = round(por_msg[m["id"]][2], 4)
            if dl != 0:
                m["impactados"].append({"id": a["id"], "delta": dl, "rs": 0.0})
                partes.append((m["id"], por_msg[m["id"]][1] / tot))
        else:
            m["ignoraram"].append({"id": a["id"], "motivo": "decisão própria prevaleceu"})
    if partes:
        a["pend"] = {"deltas": {n: v for n, v in deltas.items() if v}, "v": a["v"], "m": partes}
    return com


def hierarquia(est, arr, ibov, k):
    """Retorna (regime bruto votado, ordens do dia). Os arquitetos só veem os votos, não os preços."""
    s = ibov
    votos = [("Tendência (média de 200 dias)", regime(s, k)), ("Curto prazo (100 dias / 10 dias)", regime_b(s, k)),
             ("Amplitude (ações acima da média de 50 dias)", regime_c(arr, s, k))]
    cont = Counter(v for _, v in votos).most_common()
    escolhido = cont[0][0] if cont[0][1] >= 2 else votos[0][1]    # sem maioria: analista 1 desempata
    conf = round(cont[0][1] / 3, 2) if cont[0][1] >= 2 else round(1 / 3, 2)
    return escolhido, conf, votos


def ordens_do_dia(est, arr, d, reg, conf, votos, k):
    fo = forum(est)
    exp = P["exp_regime"][reg]
    if fo > 0.65:
        exp += 0.1
    elif fo < 0.25:
        exp -= 0.1
    exp = round(max(0.1, min(1.0, exp)), 2)
    fh = P["fator_horizonte"][reg]
    teto_eq = {e: round(exp * fh[e], 3) for e in ("curto", "medio", "longo")}
    vols = vol_classe(arr, k)
    cap_cl = {c: round(min(1.0, P["vol_alvo"] / max(v, 0.05)), 3) for c, v in vols.items()}
    return {"data": d,
            "analistas": [{"nome": n, "regime": r} for n, r in votos],
            "arquiteto_mercado": {"regime": reg, "confianca": conf},
            "arquiteto_operacoes": {"exposicao_geral": exp, "forum": fo},
            "arquitetos_horizonte": teto_eq,
            "arquitetos_mercado_especifico": cap_cl,
            "cap": {e: {c: round(teto_eq[e] * cap_cl[c], 3) for c in cap_cl} for e in teto_eq}}


def exposicao(a, ordens):
    """Exposição total máxima (gene de risco limitado pelo teto do arquiteto do horizonte) e teto por classe."""
    E, caps = a["genes"]["risco"], {}
    if ordens and a["ev"] and a["equipe"] in ordens["arquitetos_horizonte"]:
        infl = min(max(rep(a), 0), 2) / 2           # reputação alta tem mais voz: passa até 25% do teto
        E = min(E, min(1.0, ordens["arquitetos_horizonte"][a["equipe"]] * (1 + 0.25 * infl)))
        caps = {c: min(1.0, v * (1 + 0.25 * infl)) for c, v in ordens["arquitetos_mercado_especifico"].items()}
    return E, caps


def alvos(a, scores, E, caps):
    """Carteira-alvo: divide a exposição entre os ativos com sinal positivo, respeitando os tetos de peso."""
    cand = [n for n, sc in scores.items() if sc > 0]
    if not cand:
        return {}
    w = E / len(cand)
    out = {}
    for n in cand:
        x = min(w, P["max_peso"], caps.get(ATIVOS[n]["classe"], 1.0))
        if x >= 0.01:
            out[n] = round(x, 4)
    return out


# ---------------------------------------------------------------- dinheiro
def vender_posicao(est, d, a):
    if a["pos"]:
        tot = 0.0
        for n, w in a["pos"].items():
            p = a["px"].get(n, 0.0)
            tot += w * P["custo"][ATIVOS[n]["classe"]]
            if p:
                q = max(1, int(a["v"] * w / p))
                a["ops"].append({"data": d, "ativo": n, "qtd": q, "preco": round(p, 4 if ATIVOS[n]["classe"] == "moedas" else 2),
                                 "lado": "venda", "resultado": round(q * (p - a["ent_px"].get(n, p)), 2)})
        a["v"] *= 1 - tot
    a["pos"], a["alvo"], a["dias_pos"], a["ent_px"] = {}, {}, {}, {}


def quitar(est, d, a):
    """Paga o empréstimo (com juros de CDI) e devolve o banco do genoma. Retorna o fluxo líquido de caixa do genoma."""
    fluxo = 0.0
    owed = a["divida"] + a["juros"]
    if owed > 0:
        pay = min(a["v"], owed)           # se perdeu mais do que tinha, o prejuízo fica com o caixa geral
        a["v"] -= pay
        a["cap_out"] += pay
        est["cx"] += pay
        fluxo -= pay
        mov(est, d, a["nome"], "caixa", pay, "pagamento de empréstimo")
        a["divida"] = a["juros"] = 0.0
    if a["banco_usado"] > 0:
        x = min(a["v"], a["banco_usado"])
        a["v"] -= x
        a["banco"] += x
        a["cap_out"] += x
        fluxo -= x
        mov(est, d, a["nome"], f"banco {a['nome']}", x, "devolução ao banco do genoma")
        a["banco_usado"] = 0.0
    return fluxo


def tomar_credito(est, d, a, reg):
    """Ao abrir posição considerada lucrativa: usa o banco próprio e, se a reputação permite, empréstimo do caixa."""
    fluxo = 0.0
    if a["status"] != "campo" or reg not in bons_regimes(a):
        return fluxo
    if rec("teto") and a["banco"] > 0.005:
        x = a["banco"]
        a["banco"], a["banco_usado"], a["div_dias"] = 0.0, x, 0
        a["v"] += x
        a["cap_in"] += x
        fluxo += x
        mov(est, d, f"banco {a['nome']}", a["nome"], x, "uso do banco do genoma")
    if rec("emprestimo") and rep(a) >= P["rep_emp"] and a["divida"] == 0:
        emp = min(P["emp_pct"] * est["cx"], a["v"] * P["emp_max_x"])
        if emp > 1 and est["cx"] - emp >= P["reserva"]:
            est["cx"] -= emp
            a["divida"], a["div_dias"] = emp, 0
            a["v"] += emp
            a["cap_in"] += emp
            fluxo += emp
            mov(est, d, "caixa", a["nome"], emp, "empréstimo (reputação alta)")
    return fluxo


def aplicar_teto(est, d, a):
    if not rec("teto") or a["status"] != "campo" or a["divida"] > 0 or a["banco_usado"] > 0 or a["v"] <= P["teto"]:
        return
    ex = a["v"] - P["teto"]
    f, c, b = (ex * x for x in P["teto_div"])
    a["v"] = P["teto"]
    a["cap_out"] += ex
    a["fundo"] += f
    a["banco"] += b
    est["cx"] += c
    mov(est, d, a["nome"], "caixa", c, "excedente acima do teto (65%)")
    mov(est, d, a["nome"], f"fundo {a['nome']}", f, "excedente acima do teto (25% descendentes)")
    mov(est, d, a["nome"], f"banco {a['nome']}", b, "excedente acima do teto (10% banco)")


def encerrar(est, d, a, status, motivo):
    vender_posicao(est, d, a)
    quitar(est, d, a)
    cart = "campo" if a["status"] == "campo" else "treino"
    a["status"], a["morte"] = status, d
    if status == "hibernado":
        a["v_hib"], a["hib_dias"] = a["v"], 0
    a["cap_out"] += a["v"]
    if cart == "campo":
        est["cx"] += a["v"]
        mov(est, d, a["nome"], "caixa", a["v"], motivo)
    else:   # dinheiro de treino é fictício: não volta ao caixa de campo
        mov(est, d, a["nome"], "treino (descartado)", a["v"], motivo, "treino")
    for nome, chave in (("fundo", "fundo"), ("banco", "banco")):
        if a[chave] > 0.005:
            est["cx"] += a[chave]
            mov(est, d, f"{nome} {a['nome']}", "caixa", a[chave], f"{nome} devolvido ao caixa")
            a[chave] = 0.0


def reativar(est, d, reg):
    cand = [a for a in est["ags"] if a["status"] == "hibernado" and a["ev"]
            and a["hib_dias"] >= P["espera"] and a["reat"] < P["max_reat"] and reg in bons_regimes(a)]
    cand.sort(key=lambda a: -bons_regimes(a)[reg])
    for a in cand:      # volta sempre pelo TREINO, sem pular a prova fora da amostra
        cap = P["capital_treino"]
        mov(est, d, "treino (fictício)", a["nome"], cap, f"reativação no treino (mercado em {reg})", "treino")
        a.update(status="treino", morte=None, v=cap, v0=cap, pos={}, alvo={}, ent=cap, hib_dias=0, dias_pos={}, ent_px={}, px={},
                 rets_f=[], cap_in=cap, cap_out=0.0)
        a["reat"] += 1


def flush_cdi(est, d):
    v = est.get("cdi_acum", 0.0)
    if v > 0.005:
        mov(est, d, "CDI", "caixa", v, "rendimento do caixa (CDI)")
        est["cdi_acum"] = 0.0


def resgate(est, d, k, ibov):
    """Queda brusca: os piores vendem e transferem parte do capital ao melhor, até o capital mínimo seguro."""
    if not rec("resgate") or k < 6:
        return
    r1 = float(ibov.iloc[k] / ibov.iloc[k - 1] - 1)
    r5 = float(ibov.iloc[k] / ibov.iloc[k - 5] - 1)
    if not (r1 <= P["resgate_1d"] or r5 <= P["resgate_5d"]):
        return
    if k - est.get("ult_resgate", -999) < P["resgate_espera"]:
        return
    campo = [a for a in est["ags"] if a["status"] == "campo" and a["ev"] and rep(a) > 0]
    if not campo:
        return
    melhor = max(campo, key=rep)
    if melhor["v"] >= P["capital_seguro"] or melhor["divida"] > 0:
        return
    pool = [a for a in est["ags"] if a["status"] == "campo" and a["ev"] and a is not melhor and a["divida"] == 0 and len(a["rets"]) >= 20]
    pool.sort(key=lambda a: sum(a["rets"][-20:]))
    est["ult_resgate"] = k
    for a in pool[: max(1, len(pool) // 4)]:
        falta = P["capital_seguro"] - melhor["v"]
        if falta <= 0:
            break
        vender_posicao(est, d, a)
        x = min(a["v"] * P["resgate_frac"], falta)
        if x < 1:
            continue
        frac = x / a["v"]
        a["v"] -= x
        a["v0"] *= 1 - frac
        a["cap_out"] += x
        melhor["v"] += x
        melhor["v0"] += x
        melhor["cap_in"] += x
        mov(est, d, a["nome"], melhor["nome"], x, f"resgate em queda brusca (para {melhor['nome']})")


def rep_f(a):
    """Reputação só com o que aconteceu DEPOIS do corte (fora da amostra)."""
    r = a["rets_f"]
    if len(r) < P["prom_dias"]:
        return None
    sd = st.pstdev(r)
    return round(st.mean(r) / sd * 15.8, 2) if sd else 0.0


def promover(est, d, a):
    """Treino -> campo: só com prova depois do corte (dias mínimos, reputação, retorno > 0) e caixa de campo."""
    r = rep_f(a)
    if r is None or r < P["prom_rep"] or sum(a["rets_f"]) <= 0:
        return
    cap = P["capital_campo"]
    pai = next((x for x in est["ags"] if x["id"] == a["pai_id"]), None)
    usado = min(pai["fundo"], cap) if pai else 0.0
    if est["cx"] < cap - usado + P["reserva"]:
        return
    vender_posicao(est, d, a)
    a["pts_treino"] = round(a["v"] - a["v0"], 2)      # o ganho do treino fica só como pontuação
    mov(est, d, a["nome"], "treino (descartado)", a["v"], "fim do treino: ganho vira pontuação, não capital", "treino")
    if usado > 0.005:
        pai["fundo"] -= usado
        mov(est, d, f"fundo {pai['nome']}", a["nome"], usado, "promoção a campo (fundo de descendentes)")
    est["cx"] -= cap - usado
    mov(est, d, "caixa", a["nome"], cap - usado, "promoção a campo (capital de campo)")
    a.update(status="campo", v=cap, v0=cap, ent=cap, cap_in=cap, cap_out=0.0, dias_pos={}, ent_px={})


# ---------------------------------------------------------------- fim de época
def fim_epoca(est, d, reg):
    ags = est["ags"]
    flush_cdi(est, d)
    cloned = set()
    for a in [x for x in ags if vivo(x) and x["ev"]]:
        if a["status"] == "treino":
            promover(est, d, a)
        lim = P["limite"] * (1 + min(max(rep(a), 0), 2) / 2)
        if a["v"] < a["v0"] * (1 - lim):
            if bons_regimes(a):
                encerrar(est, d, a, "hibernado", f"hibernação (queda em {reg}, tem regime bom)")
            else:
                encerrar(est, d, a, "eliminado", "desligamento (sem regime bom)")
            continue
        nvivos = sum(1 for x in ags if vivo(x) and x["ev"])
        if (a["clones"] < len(P["degraus"]) and a["v"] - a["v0"] >= P["degraus"][a["clones"]] * a["v0"] / P["capital_treino"]
                and nvivos < P["max_vivos"]):
            clonar(est, d, a, P["capital_treino"], 0.0, "clones")
            cloned.add(a["id"])
    if rec("medianos"):
        campo = [x for x in ags if x["status"] == "campo" and x["ev"] and rep(x) > 0]
        if len(campo) >= 4:
            reps = sorted(rep(x) for x in campo)
            q25, q75 = reps[len(reps) // 4], reps[(3 * len(reps)) // 4]
            for a in campo:
                nvivos = sum(1 for x in ags if vivo(x) and x["ev"])
                if (a["id"] not in cloned and a["clones_med"] < 2 and q25 <= rep(a) < q75 and nvivos < P["max_vivos"]
                        and a["v"] - a["v0"] >= P["med_lucro"] * a["v0"]):
                    clonar(est, d, a, P["capital_treino"], 0.0, "clones_med")
    if rec("migracao"):
        for a in [x for x in ags if x["status"] == "campo" and x["ev"] and x["classe"] in SEGURAS and not x["migrou"]]:
            if rep(a) >= P["rep_migra"] and random.random() < P["p_migra"]:
                alvos = [n for n in DISP if ATIVOS[n]["classe"] in RISCO]
                if alvos:
                    migrar(est, d, a, random.choice(alvos))
    while sum(1 for x in ags if x["ev"] and vivo(x)) < 8:
        novo(est, d, genes(), None, True, P["capital_treino"])


def clonar(est, d, a, cap, usado, campo_contador):
    gn, motivo = mutar(a["genes"]), "clonagem"
    if campo_contador == "clones" and rec("cruzamento"):
        filhos = [x for x in est["ags"] if x["pai_id"] == a["id"] and len(x["rets"]) >= P["epoca"]]
        if filhos:
            lucro = lambda x: (x["v"] + x["cap_out"] - x["cap_in"]) / x["cap_in"]
            if st.mean(lucro(x) for x in filhos) < lucro(a):       # os descendentes pioraram
                campo = [x for x in est["ags"] if x["status"] == "campo" and x["ev"] and x is not a and rep(x) > 0]
                if campo:
                    mel = max(campo, key=rep)
                    gn, motivo = mutar(cruzar(a["genes"], mel["genes"])), f"clonagem (cruzamento com {mel['nome']})"
    univ = list(a["universo"])
    if random.random() < 0.15:            # mutação rara do universo: troca um ativo
        livres = [n for n in DISP if n not in univ]
        if livres:
            univ[random.randrange(len(univ))] = random.choice(livres)
    novo(est, d, gn, a["id"], True, cap, univ, de_fundo=(a, usado) if usado > 0.005 else None, motivo=motivo)
    a[campo_contador] += 1


def migrar(est, d, a, ativo):
    """Genoma provado passa a incluir um ativo de risco no universo e volta ao treino (capital fictício)."""
    seguras = [i for i, n in enumerate(a["universo"]) if ATIVOS[n]["classe"] in SEGURAS]
    if not seguras or ativo in a["universo"]:
        return
    vender_posicao(est, d, a)
    if a["status"] == "campo":              # o dinheiro de campo volta ao caixa; no treino ele é fictício
        a["cap_out"] += a["v"]
        est["cx"] += a["v"]
        mov(est, d, a["nome"], "caixa", a["v"], f"migração para {ativo}: capital de campo devolvido")
        a["v"] = P["capital_treino"]
        mov(est, d, "treino (fictício)", a["nome"], a["v"], f"migração para {ativo}", "treino")
        a["cap_in"], a["cap_out"] = a["v"], 0.0
    i = random.choice(seguras)
    velho = a["universo"][i]
    a["universo"][i] = ativo
    dom, setor = perfil(a["universo"])
    mov(est, d, a["nome"], a["nome"], 0.0, f"migração {velho} → {ativo} ({ATIVOS[ativo]['classe']})", "treino")
    a.update(ativo=", ".join(a["universo"]), classe=dom, setor=setor, status="treino", migrou=True, rd={},
             rets=[], rets_f=[], v0=a["v"], px={})


# ---------------------------------------------------------------- colmeia
def colmeia_dia(est, arr, k, d):
    c = est.setdefault("colmeia", {"v": P["caixa0"], "pesos": {}})
    if c["pesos"] and rec("colmeia"):
        r = sum(w * float(arr[n][k] / arr[n][k - 1] - 1) for n, w in c["pesos"].items() if n in arr)
        c["v"] *= 1 + r + (1 - sum(c["pesos"].values())) * taxa_cdi(d)
    else:
        c["v"] *= 1 + taxa_cdi(d)


def colmeia_pesos(est):
    c = est["colmeia"]
    tot, w = 0.0, {}
    for a in est["ags"]:
        if a["status"] == "campo" and a["ev"] and rep(a) > 0:
            tot += rep(a)
            for n, x in a["alvo"].items():
                w[n] = w.get(n, 0.0) + rep(a) * x
    c["pesos"] = {n: round(v / tot, 4) for n, v in w.items()} if tot else {}


# ---------------------------------------------------------------- CDI
def carregar_cdi(d0, d1):
    try:
        a = datetime.strptime(d0, "%Y-%m-%d").strftime("%d/%m/%Y")
        b = datetime.strptime(d1, "%Y-%m-%d").strftime("%d/%m/%Y")
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados?formato=json&dataInicial={a}&dataFinal={b}"
        return {datetime.strptime(x["data"], "%d/%m/%Y").strftime("%Y-%m-%d"): float(x["valor"]) / 100
                for x in json.loads(baixar(url))}
    except Exception as e:
        print("CDI falhou:", e)
        return {}


def taxa_cdi(d):
    if d in CDI:
        return CDI[d]
    ult = max(CDI) if CDI else None
    return CDI[ult] if ult and d > ult else 0.0


def cdi_acumulado(datas):
    f, out = 1.0, []
    for i, d in enumerate(datas):
        if i:
            f *= 1 + taxa_cdi(d)
        out.append(round(f, 6))
    return out


# ---------------------------------------------------------------- o dia
def dia(est, arr, opn, datas, ibov, k):
    d = datas[k]
    ultimo_dia = k == len(datas) - 1
    if rec("hierarquia"):
        bruto, conf, votos = hierarquia(est, arr, ibov, k)
    else:
        bruto, conf, votos = regime(ibov, k), 1.0, []
    reg, mudou = confirmar_regime(est, bruto)
    ordens = ordens_do_dia(est, arr, d, reg, conf, votos, k) if rec("hierarquia") else None
    ganho = est["cx"] * taxa_cdi(d)
    est["cx"] += ganho
    est["cdi_acum"] = est.get("cdi_acum", 0.0) + ganho
    colmeia_dia(est, arr, k, d)
    for a in est["ags"]:
        if a["status"] == "hibernado":
            a["hib_dias"] += 1
    usa_sent = rec("noticias") and ultimo_dia and bool(SENT)
    fila = [m for m in est.get("msgs", []) if m["data"] == datas[k - 1]] if rec("forum") else []
    msgs_id = {m["id"]: m for m in est.get("msgs", [])}
    leitura = []
    for a in [x for x in est["ags"] if vivo(x)]:
        univ = [n for n in a["universo"] if n in arr]
        if not univ:
            continue
        h = a["genes"]["horizonte_dias"]
        v_ini, fluxo = a["v"], 0.0
        pend = a.pop("pend", None)       # impacto (R$) das mensagens da véspera, medido no retorno de hoje
        if pend:
            imp = sum(dl * float(arr[n][k] / arr[n][k - 1] - 1) for n, dl in pend["deltas"].items() if n in arr) * pend["v"]
            for mid, sh in pend["m"]:
                m = msgs_id.get(mid)
                if m:
                    m["impacto_rs"] = round(m["impacto_rs"] + imp * sh, 2)
                    for x in m["impactados"]:
                        if x["id"] == a["id"]:
                            x["rs"] = round(x["rs"] + imp * sh, 2)
        if a["divida"] > 0:
            a["juros"] += a["divida"] * taxa_cdi(d)
        if a["divida"] > 0 or a["banco_usado"] > 0:
            a["div_dias"] += 1
        pos, alvo = a["pos"], a["alvo"]
        # 1) da véspera até a abertura: a carteira antiga rende (ou perde) o retorno da noite
        r_on = {n: float(opn[n][k] / arr[n][k - 1] - 1) for n in set(pos) | set(alvo) if n in arr}
        ret_on = sum(w * r_on.get(n, 0.0) for n, w in pos.items())
        a["v"] *= 1 + ret_on
        pos1 = {n: w * (1 + r_on.get(n, 0.0)) / (1 + ret_on) for n, w in pos.items()}
        # 2) na abertura: rebalanceia para a carteira-alvo decidida ontem; custo sobre o giro
        expo0, expo1 = sum(pos1.values()), sum(alvo.values())
        giro, trades = 0.0, []
        for n in set(pos1) | set(alvo):
            dw = alvo.get(n, 0.0) - pos1.get(n, 0.0)
            giro += abs(dw) * P["custo"][ATIVOS[n]["classe"]]
            if abs(dw) >= P["min_trade"]:
                trades.append((n, dw))
        a["v"] *= 1 - giro
        if expo0 < 0.02 <= expo1:
            fluxo += tomar_credito(est, d, a, reg)           # abriu a carteira
        elif expo1 < 0.02 <= expo0:
            fluxo += quitar(est, d, a)                       # zerou a carteira
        elif (a["divida"] > 0 or a["banco_usado"] > 0) and a["div_dias"] >= P["emp_dias"]:
            fluxo += quitar(est, d, a)                       # prazo máximo do empréstimo
        for n, dw in trades:
            p = float(opn[n][k])
            cl = ATIVOS[n]["classe"]
            q, pr = max(1, int(a["v"] * abs(dw) / p)), round(p, 4 if cl == "moedas" else 2)
            if dw > 0:
                w0 = pos1.get(n, 0.0)
                a["ent_px"][n] = (a["ent_px"].get(n, p) * w0 + p * dw) / (w0 + dw) if w0 > 0.005 else p
                a["ops"].append({"data": d, "ativo": n, "qtd": q, "preco": pr, "lado": "compra", "resultado": 0.0})
            else:
                res = round(q * (p - a["ent_px"].get(n, p)), 2)
                if alvo.get(n, 0.0) < 0.005:
                    a["ent_px"].pop(n, None)
                a["ops"].append({"data": d, "ativo": n, "qtd": q, "preco": pr, "lado": "venda", "resultado": res})
        # 3) da abertura ao fechamento: a carteira nova rende
        r_in = {n: float(arr[n][k] / opn[n][k] - 1) for n in alvo if n in arr}
        ret_in = sum(w * r_in[n] for n, w in alvo.items() if n in r_in)
        a["v"] *= 1 + ret_in
        pos2 = {n: w * (1 + r_in.get(n, 0.0)) / (1 + ret_in) for n, w in alvo.items()}
        a["pos"] = {n: round(w, 6) for n, w in pos2.items() if w >= 0.001}
        a["px"] = {n: float(arr[n][k]) for n in univ}
        a["dias_pos"] = {n: a["dias_pos"].get(n, 0) + 1 for n in a["pos"]}
        ret = (a["v"] - fluxo) / v_ini - 1
        a["rets"].append(ret)
        if d > est["corte"]:
            a["rets_f"].append(ret)
        x = a["rd"].setdefault(reg, [0.0, 0])
        x[0] += ret
        x[1] += 1
        # 4) decisão para amanhã: um sinal por ativo do universo
        tend = a["genes"]["tendencia"]
        base = {n: float(arr[n][k] / arr[n][k - h] - 1) * tend for n in univ}
        sc = dict(base)
        if usa_sent:
            for n in univ:
                sc[n] += a["genes"].get("peso_noticias", 0.0) * SENT.get(ATIVOS[n]["classe"], 0.0) * P["peso_noticia_escala"]
        E, caps = exposicao(a, ordens)
        decide = lambda s_: alvos(a, s_, E, caps)
        if usa_sent and a["ev"]:
            d0, d1 = decide(base), decide(sc)
            ef = {n: round(d1.get(n, 0.0) - d0.get(n, 0.0), 4) for n in univ if abs(d1.get(n, 0.0) - d0.get(n, 0.0)) > 1e-9}
            leitura.append({"id": a["id"], "nome": a["nome"], "carteira": a["status"], "peso": a["genes"].get("peso_noticias", 0.0),
                            "classes": sorted({ATIVOS[n]["classe"] for n in univ}),
                            "efeitos": [{"ativo": n, "classe": ATIVOS[n]["classe"],
                                         "sentimento": SENT.get(ATIVOS[n]["classe"], 0.0), "delta": v} for n, v in ef.items()]})
        vis = [m for m in fila if m["autor_id"] != a["id"] and m["carteira"] == a["status"]     # treino e campo NÃO se leem
               and (m["ativo"] in univ or any(ATIVOS[n]["classe"] == m["classe"] for n in univ))]
        novo_alvo = forum_efeito(est, a, vis, sc, decide) if a["ev"] else decide(sc)
        if rec("prazos"):
            pe = prazo_efetivo(a)
            for n, w in a["pos"].items():
                if (novo_alvo.get(n, 0.0) < 0.01 and a["dias_pos"].get(n, 0) < pe and n in arr
                        and arr[n][k] > a["ent_px"].get(n, float(arr[n][k])) * (1 - P["stop"])):
                    novo_alvo[n] = min(w, P["max_peso"])         # prazo pedido: mantém a posição
            tot = sum(novo_alvo.values())
            if tot > 1.0:
                novo_alvo = {n: w / tot for n, w in novo_alvo.items()}
        a["alvo"] = novo_alvo
        if rec("forum") and a["ev"] and rep(a) > 0.2:
            mud = [(n, novo_alvo.get(n, 0.0) - a["pos"].get(n, 0.0)) for n in univ
                   if (novo_alvo.get(n, 0.0) >= P["min_trade"]) != (a["pos"].get(n, 0.0) >= P["min_trade"])]
            mud.sort(key=lambda t: -abs(t[1]))
            for n, dl in mud[:2]:
                postar(est, d, a, n, 1 if dl > 0 else -1)
        a["curva"].append(round(a["v"], 2))
        aplicar_teto(est, d, a)
    if usa_sent:
        est.setdefault("sent_hist", {})[d] = dict(SENT)
        est["leitura_noticias"] = {"data": d, "sentimento": dict(SENT), "agentes": leitura}
    resgate(est, d, k, ibov)
    if mudou:
        for a in [x for x in est["ags"] if x["status"] == "campo" and x["ev"]]:
            if ruim_em(a, reg) and bons_regimes(a):
                encerrar(est, d, a, "hibernado", f"hibernação (mercado virou para {reg})")
    reativar(est, d, reg)
    colmeia_pesos(est)
    ms = est.get("msgs", [])
    est["msgs"] = [m for m in ms if m["carteira"] == "campo"][-150:] + [m for m in ms if m["carteira"] != "campo"][-150:]
    est["msgs"].sort(key=lambda m: m["id"])
    est["ordens"] = ordens
    est["forum"] = ordens["arquiteto_operacoes"]["forum"] if ordens else None
    total = est["cx"] + sum(x["v"] + x["fundo"] + x["banco"] for x in est["ags"] if x["status"] == "campo")
    tpts = sum(x["v"] - x["v0"] for x in est["ags"] if x["status"] == "treino")
    est.setdefault("serie", []).append({"data": d, "lab": round(total, 2), "ibov": round(float(ibov.iloc[k]), 2),
                                        "colmeia": round(est["colmeia"]["v"], 2), "treino_pts": round(tpts, 2)})
    est["dias_ep"] += 1
    if est["dias_ep"] >= P["epoca"]:
        est["dias_ep"] = 0
        fim_epoca(est, d, reg)


# ---------------------------------------------------------------- saída
def exportar(est, arr, datas):
    G = []
    for a in est["ags"]:
        rd = a["rd"]
        bons = bons_regimes(a)
        base = {r: v[0] / v[1] for r, v in rd.items() if v[1] >= 5}
        pos = {r: m for r, m in base.items() if m > 0}
        melhor = max(pos, key=pos.get) if pos else ("sem dados" if not base else "nenhum")
        i0 = datas.index(a["nasc"]) if a["nasc"] in datas else None
        i1 = datas.index(a["morte"]) if a["morte"] in datas and not vivo(a) else len(datas) - 1
        bh = None
        ub = [n for n in a["universo"] if n in arr]
        if i0 is not None and ub and i1 > i0:      # comprar e segurar o universo inteiro, com pesos iguais
            bh = round(sum(float(arr[n][i1] / arr[n][i0] - 1) for n in ub) / len(ub), 4)
        lt = (a["v"] if vivo(a) else 0.0) + a["cap_out"] - a["cap_in"]
        cv = a["curva"][-120:]
        if len(cv) < 2:
            cv = [a["v0"], round(a["v"], 2)]
        G.append({"id": a["id"], "nome": a["nome"], "equipe": a["equipe"], "setor": a["setor"],
                  "classe": a["classe"], "ativo": a["ativo"], "ativos": a["universo"], "status": a["status"],
                  "carteira_atual": {n: round(w, 4) for n, w in a["pos"].items()},
                  "nascimento": a["nasc"], "morte": a["morte"], "pai_id": a["pai_id"],
                  "clonagens": a["clones"] + a["clones_med"], "reativacoes": a["reat"], "migrou": a["migrou"],
                  "genes": a["genes"], "prazo_efetivo": prazo_efetivo(a),
                  "capital_inicial": a["v0"], "capital_atual": round(a["v"], 2),
                  "capital_investido": round(a["cap_in"], 2), "lucro_total": round(lt, 2), "bh": bh,
                  "fundo": round(a["fundo"], 2), "banco": round(a["banco"], 2), "divida": round(a["divida"], 2),
                  "melhor_regime": melhor, "regimes_bons": sorted(bons), "reputacao": rep(a),
                  "curva": cv, "operacoes": a["ops"][-30:][::-1],
                  "carteira": "campo" if a["status"] == "campo" else "treino",
                  "pts_treino": round(a["v"] - a["v0"], 2) if a["status"] == "treino" else a["pts_treino"]})
    saida = {"atualizado": AGORA, "ultimo_pregao": est["ultimo"], "G": G, "caixa": est["led"],
             "regime_atual": est.get("regime", {}).get("atual"), "versao": VERSAO + (" (base)" if os.environ.get("LAB_BASE") == "1" else ""), "params": P,
             "hierarquia": est.get("ordens"),
             "colmeia": {"valor": round(est["colmeia"]["v"], 2), "pesos": est["colmeia"]["pesos"]}}
    campo = [a for a in est["ags"] if a["status"] == "campo"]
    treino = [a for a in est["ags"] if a["status"] == "treino"]
    fl = [x for x in est["led"] if x.get("carteira", "campo") == "campo"]
    dif = round(est["cx"] - (sum(x["valor"] for x in fl if x["destino"] == "caixa")
                             - sum(x["valor"] for x in fl if x["origem"] == "caixa")), 2)
    saida["carteiras"] = {
        "campo": {"caixa": round(est["cx"], 2), "investido": round(sum(a["v"] + a["fundo"] + a["banco"] for a in campo), 2),
                  "n": len(campo), "capital_por_genoma": P["capital_campo"]},
        "treino": {"pontuacao": round(sum(a["v"] - a["v0"] for a in treino), 2), "n": len(treino),
                   "capital_ficticio": P["capital_treino"]}}
    saida["saude"] = {"ledger_fecha": abs(dif) < 1.0, "diferenca": dif, "corte": est.get("corte"),
                      "pregoes_processados": len(est.get("serie", []))}
    if not saida["saude"]["ledger_fecha"]:
        print("ALERTA: o livro-razão de campo não fecha:", dif)
    ms = est.get("msgs", [])
    msgs = [m for m in ms if m["carteira"] == "campo"][-100:] + [m for m in ms if m["carteira"] != "campo"][-100:]
    for m in msgs:
        n = len(m["impactados"])
        m["nivel"] = "alto" if n >= 3 or abs(m["impacto_rs"]) >= 5 else ("médio" if n >= 1 else "baixo")
    saida["forum"] = msgs
    saida["noticias_agentes"] = est.get("leitura_noticias")
    if est.get("serie"):
        cdi = cdi_acumulado([x["data"] for x in est["serie"]])
        saida["benchmark"] = {"corte": est.get("corte"), "serie": [{**x, "cdi": c} for x, c in zip(est["serie"], cdi)]}
    (DATA / "genomas.json").write_text(json.dumps(saida, ensure_ascii=False), encoding="utf-8")
    (DATA / "estado.json").write_text(json.dumps(est, ensure_ascii=False), encoding="utf-8")


def carregar():
    tks = sorted({v["t"] for v in ATIVOS.values()} | {IBOV, USD})
    bruto = yf.download(tks, period="2y", interval="1d", auto_adjust=True, progress=False)
    C, O = bruto["Close"], bruto["Open"]
    cal = C[IBOV].dropna().index
    hoje, fechado = hoje_fechado()
    if not fechado and cal[-1].strftime("%Y-%m-%d") >= hoje:
        cal = cal[:-1]
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


def carregar_sentimento():
    """Sentimento por classe (data/sentimento.json), só se for recente (até 48h)."""
    try:
        j = json.loads((DATA_RAIZ / "sentimento.json").read_text(encoding="utf-8"))
        idade = datetime.now(timezone.utc) - datetime.fromisoformat(j["atualizado"])
        if idade.total_seconds() > 48 * 3600:
            return {}
        return {c: (v["media"] if v.get("quantidade", 0) >= 3 else 0.0) for c, v in j.get("por_classe", {}).items()}
    except Exception as e:
        print("sentimento indisponível:", e)
        return {}


def main():
    global DISP
    ibov, datas, arr, opn = carregar()
    DISP = sorted(arr)
    print("ativos disponíveis:", ", ".join(DISP))
    CDI.update(carregar_cdi(datas[-300], datas[-1]))
    if not CDI:
        print("Sem CDI do Banco Central: rodada abortada, estado intocado.")
        return
    SENT.update(carregar_sentimento())
    f = DATA / "estado.json"
    if f.exists():
        est = json.loads(f.read_text(encoding="utf-8"))
    else:
        random.seed(7)
        est = {"ultimo": datas[-P["treino"] - 1], "ags": [], "led": [], "cx": P["caixa0"], "dias_ep": 0,
               "corte": datas[-1], "versao": VERSAO}
        mov(est, est["ultimo"], "externo", "caixa", P["caixa0"], "aporte inicial (simulado)")
        for i in range(P["pop"] + P["controle"]):
            novo(est, est["ultimo"], genes(), None, i < P["pop"], P["capital_treino"])
            if i >= P["pop"]:
                est["ags"][-1]["equipe"] = "generalista"
    if "serie" not in est:
        i0 = datas.index(est["ultimo"])
        est["serie"] = [{"data": est["ultimo"], "lab": P["caixa0"], "ibov": round(float(ibov.iloc[i0]), 2),
                         "colmeia": P["caixa0"]}]
    novos = [k for k, d in enumerate(datas) if d > est["ultimo"]]
    if not novos:
        print("Sem pregões novos.")
        return
    random.seed(novos[-1])
    for k in novos:
        dia(est, arr, opn, datas, ibov, k)
    est["ultimo"] = datas[-1]
    flush_cdi(est, est["ultimo"])
    exportar(est, arr, datas)
    print("Processados", len(novos), "pregões; último:", est["ultimo"])


if __name__ == "__main__":
    main()
