"""Exporta dados REAIS para a pasta data/ (roda no GitHub Actions, ver .github/workflows).

Gera: data/mercado.json (preços, Ibovespa, Selic e dólar), data/analise.json (regime de
mercado por regras simples) e data/noticias.json (manchetes + sentimento por palavras-chave).
Para testar na sua máquina: pip install yfinance numpy  e  python scripts/exportar.py
"""
import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np
import yfinance as yf

OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)
ACOES = ["PETR4", "VALE3", "ITUB4", "BBDC4", "WEGE3", "MGLU3", "B3SA3", "ABEV3"]
IBOV = "^BVSP"
AGORA = datetime.now(timezone.utc).isoformat(timespec="seconds")
FEEDS = [
    ("Google Notícias", "https://news.google.com/rss/search?q=Ibovespa+OR+bolsa+OR+mercado+financeiro+when:2d&hl=pt-BR&gl=BR&ceid=BR:pt-419"),
    ("InfoMoney", "https://www.infomoney.com.br/feed/"),
    ("Google Notícias", "https://news.google.com/rss/search?q=bitcoin+OR+criptomoedas+when:2d&hl=pt-BR&gl=BR&ceid=BR:pt-419"),
    ("Google Notícias", "https://news.google.com/rss/search?q=d%C3%B3lar+OR+c%C3%A2mbio+OR+euro+when:2d&hl=pt-BR&gl=BR&ceid=BR:pt-419"),
    ("Google Notícias", "https://news.google.com/rss/search?q=fundos+imobili%C3%A1rios+OR+FII+when:2d&hl=pt-BR&gl=BR&ceid=BR:pt-419"),
    ("Google Notícias", "https://news.google.com/rss/search?q=Wall+Street+OR+Nasdaq+OR+Fed+when:2d&hl=pt-BR&gl=BR&ceid=BR:pt-419"),
]
CLASSES = {
    "cripto": ["bitcoin", "cripto", "ethereum", "btc"],
    "moedas": ["dólar", "dolar", "euro", "câmbio", "cambio"],
    "imobiliario": ["fii", "imobiliári", "imobiliari"],
    "exterior": ["wall street", "nasdaq", "s&p", "fed ", "nova york"],
}
POS = ["alta", "sobe", "dispara", "lucro", "recorde", "avança", "ganha", "otimismo", "valoriza"]
NEG = ["queda", "cai", "despenca", "prejuízo", "perde", "recua", "crise", "pessimismo", "desvaloriza", "risco"]


def salvar(nome, obj):
    (OUT / nome).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def baixar(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=20).read()


def mercado():
    tk = [a + ".SA" for a in ACOES] + [IBOV]
    df = yf.download(tk, period="2y", interval="1d", auto_adjust=True, progress=False)["Close"]
    df = df.dropna(how="all")

    def item(t, nome):
        s = df[t].dropna()
        return {"ticker": nome, "preco": round(float(s.iloc[-1]), 2),
                "var_dia": round(float(s.iloc[-1] / s.iloc[-2] - 1), 4),
                "hist": [round(float(x), 2) for x in s.tail(60)]}

    macro = {}
    for nome, cod in (("selic_meta", 432), ("dolar", 1)):  # séries do Banco Central (SGS)
        try:
            url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{cod}/dados/ultimos/1?formato=json"
            macro[nome] = float(json.loads(baixar(url))[0]["valor"])
        except Exception as e:
            print("BCB falhou:", nome, e)
            macro[nome] = None
    salvar("mercado.json", {"atualizado": AGORA, "ibovespa": item(IBOV, "IBOV"),
                            "acoes": [item(a + ".SA", a) for a in ACOES], "macro": macro})
    return df[IBOV].dropna()


def regime(c, i):
    """Classifica o dia i usando só dados até o dia i (sem espiar o futuro)."""
    s = c.iloc[: i + 1]
    ma, ret20 = s.tail(200).mean(), s.iloc[-1] / s.iloc[-21] - 1
    vol = float(s.pct_change().tail(20).std() * np.sqrt(252))
    if vol > 0.35 and ret20 < -0.08:
        return "crise"
    if s.iloc[-1] < ma and ret20 < 0:
        return "queda"
    if s.iloc[-1] > ma and ret20 > 0:
        return "alta volátil" if vol > 0.25 else "alta calma"
    return "lateral"


def analise(c):
    n = len(c)
    serie = [{"data": c.index[i].strftime("%Y-%m-%d"), "fechamento": round(float(c.iloc[i])),
              "regime": regime(c, i)} for i in range(n - 120, n)]
    vol, ma = float(c.pct_change().tail(20).std() * np.sqrt(252)), c.tail(200).mean()
    salvar("analise.json", {
        "atualizado": AGORA, "regime_atual": serie[-1]["regime"],
        "indicadores": {"retorno_20d": round(float(c.iloc[-1] / c.iloc[-21] - 1) * 100, 1),
                        "vol_20d": round(vol * 100, 1),
                        "vs_media200": round(float(c.iloc[-1] / ma - 1) * 100, 1)},
        "serie": serie,
        "regras": ["Regras simples sobre o Ibovespa (ponto de partida, para você ajustar e testar):",
                   "Crise: volatilidade acima de 35% ao ano e queda maior que 8% em 20 dias.",
                   "Queda: abaixo da média de 200 dias e com retorno negativo em 20 dias.",
                   "Alta: acima da média de 200 dias e com retorno positivo; volátil se a volatilidade passa de 25%.",
                   "Lateral: todo o resto."]})


PESO_PALAVRAS = {"selic": 3, "copom": 3, "banco central": 2, "juros": 2, "inflação": 2, "ipca": 2, "pib": 2,
                 "recessão": 3, "fed": 3, "guerra": 3, "tarifa": 2, "petróleo": 2, "opep": 2, "crise": 3,
                 "calote": 3, "rebaixa": 2, "eleição": 2, "recorde": 2, "dividendos": 1, "balanço": 1,
                 "bitcoin": 1, "dólar": 1, "petrobras": 1, "vale": 1}
PARADAS = {"para", "como", "mais", "sobre", "após", "pelo", "pela", "este", "esta", "isso", "seus", "suas", "quando", "entre"}


def _tokens(t):
    return {w for w in re.findall(r"[a-zà-ú0-9]{4,}", t.lower()) if w not in PARADAS}


def pontuar_impacto(itens, agora):
    """Dá a cada notícia um índice de impacto e as razões, de forma transparente (sem prometer previsão)."""
    toks = [_tokens(i["titulo"]) for i in itens]
    for n, it in enumerate(itens):
        similares = {itens[m]["fonte"] for m in range(len(itens)) if m != n and toks[n] and toks[m]
                     and len(toks[n] & toks[m]) / len(toks[n] | toks[m]) >= 0.3}
        cobertura = 1 + len(similares)
        try:
            horas = max(0.0, (agora - datetime.fromisoformat(it["data"])).total_seconds() / 3600)
        except Exception:
            horas = 48.0
        recencia = 1 / (1 + horas / 12)
        achadas = [w for w in PESO_PALAVRAS if w in it["titulo"].lower()]
        kw = min(6, sum(PESO_PALAVRAS[w] for w in achadas))
        sent = abs(float(it.get("sentimento", 0)))
        score = (1 + 2 * sent + 0.4 * kw) * (1 + 0.5 * (cobertura - 1)) * (0.5 + recencia)
        motivos = []
        if cobertura > 1:
            motivos.append(f"tema coberto por {cobertura} veículos")
        if achadas:
            motivos.append("palavras de alto impacto: " + ", ".join(achadas[:3]))
        if sent > 0:
            motivos.append(f"sentimento {'positivo' if it['sentimento'] > 0 else 'negativo'} ({it['sentimento']:+.2f})")
        motivos.append("publicada há %s" % (f"{horas:.0f} h" if horas >= 1 else "menos de 1 h") if horas < 48 else "data não informada ou antiga")
        it["impacto"], it["cobertura"], it["motivos"] = round(score, 2), cobertura, motivos
    return itens


def classes_da_noticia(t):
    t = t.lower()
    achadas = [c for c, ws in CLASSES.items() if any(w in t for w in ws)]
    return achadas or ["acoes"]


def sentimento(t):
    t = t.lower()
    p, n = sum(w in t for w in POS), sum(w in t for w in NEG)
    return round((p - n) / max(p + n, 1), 2)


def noticias():
    """Atualiza notícias sem apagar o último conjunto válido se os feeds falharem."""
    itens, feeds_ok = [], 0
    anterior = OUT / "noticias.json"
    for fonte, url in FEEDS:
        try:
            raiz = ET.fromstring(baixar(url))
            feeds_ok += 1
            for it in raiz.iter("item"):
                t = (it.findtext("title") or "").strip()
                link = (it.findtext("link") or "").strip()
                if not t or not link:
                    continue
                try:
                    d = parsedate_to_datetime(it.findtext("pubDate")).astimezone(timezone.utc).isoformat(timespec="minutes")
                except Exception:
                    d = ""
                itens.append({"titulo": t, "fonte": it.findtext("source") or fonte,
                              "link": link, "data": d, "sentimento": sentimento(t),
                              "classes": classes_da_noticia(t)})
        except Exception as e:
            print("feed falhou:", fonte, e)

    # Se todos os feeds falharem, não sobrescreve notícias válidas com lista vazia.
    if feeds_ok == 0:
        print("Todos os feeds falharam; mantendo noticias.json anterior.")
        return

    # Deduplicação por link, mantendo o item mais recente encontrado.
    unicos = {}
    for item in itens:
        unicos[item["link"]] = item
    itens = sorted(unicos.values(), key=lambda x: x["data"], reverse=True)[:60]

    # Um feed respondeu, mas não trouxe itens: preservar a última versão não vazia.
    if not itens and anterior.exists():
        try:
            antigo = json.loads(anterior.read_text(encoding="utf-8"))
            itens = antigo.get("itens", [])
        except Exception:
            itens = []

    itens = pontuar_impacto(itens, datetime.now(timezone.utc))
    destaque = max(itens, key=lambda x: x["impacto"]) if itens else None   # a notícia mais impactante do momento
    payload = {"atualizado": AGORA, "destaque": destaque, "itens": itens}
    salvar("noticias.json", payload)

    # Saída agregada e aditiva para o painel/integração dos Genomas.
    valores = [float(x.get("sentimento", 0)) for x in itens if isinstance(x.get("sentimento"), (int, float))]
    media = round(sum(valores) / len(valores), 3) if valores else 0.0
    por_classe = {}
    for c in list(CLASSES) + ["acoes"]:
        v = [float(x["sentimento"]) for x in itens if c in x.get("classes", []) and isinstance(x.get("sentimento"), (int, float))]
        por_classe[c] = {"quantidade": len(v), "media": round(sum(v) / len(v), 3) if v else 0.0}
    if media > 0.15:
        sinal = "positivo"
    elif media < -0.15:
        sinal = "negativo"
    else:
        sinal = "neutro"
    salvar("sentimento.json", {
        "atualizado": AGORA,
        "origem": "noticias.json",
        "metodo": "média de pontuações por palavras-chave; sinal informativo, não previsão",
        "quantidade": len(valores),
        "media": media,
        "por_classe": por_classe,
        "sinal": sinal,
    })


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "noticias":   # modo leve, usado de 15 em 15 minutos
        noticias()
        sys.exit(0)
    try:
        analise(mercado())
    except Exception as e:
        print("mercado/análise falhou:", e)
    noticias()
