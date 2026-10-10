"""Backend inicial do laboratório de agentes evolutivos.

Como rodar (na pasta do projeto):
    pip install -r requirements.txt
    uvicorn main:app --reload
Abra http://127.0.0.1:8000
Na 1ª execução cria agentes.db com dados de EXEMPLO (simulação, nada é real).
Depois, o motor de evolução vai gravar nestas mesmas tabelas.
"""
import datetime as dt
import json
import random
import sqlite3
from pathlib import Path

import motor
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

BASE = Path(__file__).parent
DB = BASE / "lab.db"
REGIMES = ["alta calma", "alta volátil", "lateral", "queda", "crise"]
app = FastAPI(title="Agentes Evolutivos")


def con():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


def seed_antigo(c):  # não é mais usado: o motor.py gera os dados
    """Gera 14 genomas de mentira, só para o painel ter o que mostrar."""
    random.seed(42)
    hoje = dt.date.today()
    for i in range(1, 15):
        idade = random.randint(40, 400)
        nasc = hoje - dt.timedelta(days=idade)
        status = random.choices(["campo", "treino", "hibernado", "eliminado"], [5, 3, 3, 3])[0]
        morte = hoje - dt.timedelta(days=random.randint(0, idade // 2)) if status == "eliminado" else None
        deriva, v, curva = random.uniform(-0.004, 0.006), 1000.0, []
        for _ in range(60):
            v *= 1 + random.gauss(deriva, 0.02)
            curva.append(v)
        genes = {
            "risco": round(random.random(), 2),
            "tecnico": round(random.random(), 2),
            "noticias": round(random.random(), 2),
            "geopolitica": round(random.random(), 2),
            "horizonte_dias": random.choice([5, 20, 60, 120]),
        }
        c.execute(
            "INSERT INTO genomas VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (i, f"G-{i:03d}", random.choice(["curto", "medio", "longo", "generalista"]),
             random.choice(["tecnologia", "energia", "financeiro", "varejo"]), status,
             nasc.isoformat(), morte.isoformat() if morte else None,
             random.randint(1, i - 1) if i > 4 else None, json.dumps(genes),
             1000.0, curva[-1], random.choice(REGIMES)),
        )
        c.executemany("INSERT INTO patrimonio VALUES (?,?,?)", [(i, d, x) for d, x in enumerate(curva)])
        for k in range(6):
            c.execute(
                "INSERT INTO operacoes VALUES (?,?,?,?,?,?,?)",
                (i, (hoje - dt.timedelta(days=k * 5)).isoformat(),
                 random.choice(["PETR4", "VALE3", "ITUB4", "WEGE3", "MGLU3"]),
                 random.choice(["compra", "venda"]), random.randint(1, 20) * 10,
                 round(random.uniform(10, 60), 2), round(random.gauss(5, 60), 2)),
            )


def init():
    c = con()
    c.executescript(
        """
        CREATE TABLE IF NOT EXISTS genomas(
            id INTEGER PRIMARY KEY, nome TEXT, equipe TEXT, setor TEXT, status TEXT,
            nascimento TEXT, morte TEXT, pai_id INTEGER, genes TEXT,
            capital_inicial REAL, capital_atual REAL, melhor_regime TEXT);
        CREATE TABLE IF NOT EXISTS patrimonio(genoma_id INTEGER, dia INTEGER, valor REAL);
        CREATE TABLE IF NOT EXISTS operacoes(
            genoma_id INTEGER, data TEXT, ativo TEXT, lado TEXT, qtd INTEGER, preco REAL, resultado REAL);
        """
    )
    if c.execute("SELECT COUNT(*) FROM genomas").fetchone()[0] == 0:
        motor.rodar(c)
    c.commit()
    c.close()


init()

SEL = "SELECT g.*, (SELECT COUNT(*) FROM genomas f WHERE f.pai_id = g.id) AS clonagens FROM genomas g"


def linha(r):
    d = dict(r)
    d["genes"] = json.loads(d["genes"])
    return d


@app.get("/api/genomas")
def listar():
    c = con()
    out = [linha(r) for r in c.execute(SEL)]
    c.close()
    return out


@app.get("/api/genomas/{gid}")
def detalhe(gid: int):
    c = con()
    g = c.execute(SEL + " WHERE g.id = ?", (gid,)).fetchone()
    if not g:
        c.close()
        raise HTTPException(404, "Genoma não encontrado")
    d = linha(g)
    d["curva"] = [r[0] for r in c.execute("SELECT valor FROM patrimonio WHERE genoma_id=? ORDER BY dia", (gid,))]
    d["operacoes"] = [dict(r) for r in c.execute(
        "SELECT data, ativo, lado, qtd, preco, resultado FROM operacoes WHERE genoma_id=? ORDER BY data DESC LIMIT 30", (gid,))]
    c.close()
    return d


@app.get("/api/caixa")
def caixa():
    c = con()
    t = [dict(r) for r in c.execute("SELECT * FROM caixa ORDER BY id DESC")]
    ent = sum(x["valor"] for x in t if x["destino"] == "caixa")
    sai = sum(x["valor"] for x in t if x["origem"] == "caixa")
    c.close()
    return {"saldo": ent - sai, "transacoes": t}


@app.get("/")
def home():
    return FileResponse(BASE / "index.html")
