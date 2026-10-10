"""Exporta dados REAIS para a pasta data/ (roda no GitHub Actions, ver .github/workflows).

Gera: data/mercado.json (preços, Ibovespa, Selic e dólar), data/analise.json (regime de
mercado por regras simples) e data/noticias.json (manchetes + sentimento por palavras-chave).
Para testar na sua máquina: pip install yfinance numpy  e  python scripts/exportar.py
"""
import json
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
]
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
                              "link": link, "data": d, "sentimento": sentimento(t)})
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
    itens = sorted(unicos.values(), key=lambda x: x["data"], reverse=True)[:40]

    # Um feed respondeu, mas não trouxe itens: preservar a última versão não vazia.
    if not itens and anterior.exists():
        try:
            antigo = json.loads(anterior.read_text(encoding="utf-8"))
            itens = antigo.get("itens", [])
        except Exception:
            itens = []

    payload = {"atualizado": AGORA, "itens": itens}
    salvar("noticias.json", payload)

    # Saída agregada e aditiva para o painel/integração dos Genomas.
    valores = [float(x.get("sentimento", 0)) for x in itens if isinstance(x.get("sentimento"), (int, float))]
    media = round(sum(valores) / len(valores), 3) if valores else 0.0
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
        "sinal": sinal,
    })


if __name__ == "__main__":
    try:
        analise(mercado())
    except Exception as e:
        print("mercado/análise falhou:", e)
    noticias()
