"""[VERSÃO MOB] HTML interativo das curvas de sobrevivência por MOB (months on book).
Análogo a gerar_html_sobrevivencia.py, lendo as tabelas _mob e gravando um HTML novo
(sobrevivencia_interativa_mob.html). Não toca na versão atual.

Relógio: t = meses desde a ENTRADA na janela (MOB). Mesma interatividade (seleção por
categoria/grupo, escala-Y dinâmica, toggle de extrapolação Weibull, tooltip com mediana/IQR).

Uso:  /tmp/consig_venv/bin/python gerar_html_sobrevivencia_mob.py
Saída: outputs/sobrevivencia_interativa_mob.html
"""
import json, shutil
import pandas as pd
from matplotlib import cm, colors

KM = "outputs/tables/sobrevivencia_km_mob_2023.csv"
RES = "outputs/tables/sobrevivencia_resumo_mob_2023.csv"
EXT = "outputs/tables/sobrevivencia_weibull_extrap_mob_2023.csv"
PAR = "outputs/tables/sobrevivencia_weibull_params_mob_2023.csv"
MONO = "outputs/tables/sobrevivencia_weibull_estatisticas_mono_mob_2023.csv"
OUT = "outputs/sobrevivencia_interativa_mob.html"
TMP = "/tmp/sobrevivencia_interativa_mob.html"
H = 12
H_EXT = 36

km = pd.read_csv(KM).rename(columns={"mob": "mes"})   # MOB no eixo do tempo
res = pd.read_csv(RES).set_index("categoria")
ext = pd.read_csv(EXT)
par = pd.read_csv(PAR).set_index("categoria")
mono = pd.read_csv(MONO).set_index("categoria")
ks = sorted(km["categoria"].unique())

cmap = cm.get_cmap("RdYlGn_r"); norm = colors.Normalize(vmin=min(ks), vmax=max(ks))
cor = {k: colors.to_hex(cmap(norm(k))) for k in ks}

series = []
for k in ks:
    s = km[km["categoria"] == k].sort_values("mes")
    S = [round(float(v), 5) for v in s["S"].tolist()]
    r = res.loc[k]
    med = r["mediana_meses"]; med = None if pd.isna(med) else int(med)
    w = ext[ext["categoria"] == k].sort_values("mes")
    W = [round(float(v), 5) for v in w["S_weibull"].tolist()]
    pr = par.loc[k]; mo = mono.loc[k]
    series.append({
        "k": int(k), "cor": cor[k], "S": S, "W": W,
        "n": int(r["n"]), "risco12": round(float(r["risco_deslig_12m_KM"]) * 100, 2),
        "rmst": round(float(r["RMST12_meses"]), 2),
        "perdidos": round(float(r["meses_perdidos"]), 2),
        "mediana": med,
        "shape": round(float(pr["shape_p"]), 2),
        "risco36": round(float(pr["risco_36m"]) * 100, 1),
        "q1": round(float(mo["q1_meses_mono"]), 1),
        "medm": round(float(mo["mediana_meses_mono"]), 1),
        "q3": round(float(mo["q3_meses_mono"]), 1),
        "pct_novo": round(float(r["pct_novo"]), 1),
    })
DATA = json.dumps(series, ensure_ascii=False)

GROUPS = [
    ("Risco Mínimo",      [1, 2],                   "#1a9850"),
    ("Risco Baixo",       [3, 4, 5, 6],             "#86cb66"),
    ("Risco Médio-Baixo", [7, 8, 9, 10, 11],        "#c9a227"),
    ("Risco Médio",       [12, 13, 14, 15, 16, 17], "#fb8d3d"),
    ("Risco Alto",        [18, 19, 20, 21, 22, 23], "#d73027"),
]
GROUPS_JSON = json.dumps([{"nome": n, "cats": c, "cor": col} for n, c, col in GROUPS], ensure_ascii=False)

HTML = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<title>Sobrevivência do emprego por categoria — visão MOB — 2023</title>
<style>
  :root { --bg:#fff; --ink:#1a1a1a; --muted:#666; --grid:#e6e6e6; }
  * { box-sizing:border-box; }
  body { font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif; color:var(--ink);
         margin:0; padding:18px 22px; background:var(--bg); }
  h1 { font-size:18px; margin:0 0 2px; }
  .sub { color:var(--muted); font-size:13px; margin:0 0 14px; }
  .wrap { display:flex; gap:18px; align-items:flex-start; flex-wrap:wrap; }
  .chart { flex:1 1 720px; min-width:520px; }
  .panel { flex:0 0 300px; }
  .btns { margin-bottom:8px; }
  .btns button { font-size:12px; padding:4px 10px; margin-right:6px; cursor:pointer;
                 border:1px solid #ccc; border-radius:6px; background:#f7f7f7; }
  .btns button:hover { background:#ececec; }
  .list { max-height:560px; overflow:auto; border:1px solid #eee; border-radius:8px; padding:6px 8px; }
  .row { display:flex; align-items:center; gap:8px; padding:3px 4px; font-size:12.5px;
         border-radius:5px; cursor:pointer; }
  .row:hover { background:#f4f4f4; }
  .row input { cursor:pointer; }
  .sw { width:22px; height:4px; border-radius:2px; flex:0 0 auto; }
  .txt { display:flex; flex-direction:column; line-height:1.25; }
  .t1 { font-weight:600; }
  .t2 { color:var(--muted); font-size:11px; font-variant-numeric:tabular-nums; }
  svg { width:100%; height:auto; display:block; }
  .axis { stroke:#999; stroke-width:1; }
  .grid { stroke:var(--grid); stroke-width:1; }
  .tick { fill:var(--muted); font-size:11px; }
  .alabel { fill:var(--ink); font-size:12px; }
  .curve { fill:none; stroke-width:1.8; }
  .ext { fill:none; stroke-width:1.5; stroke-dasharray:5 4; }
  .dot { stroke:#fff; stroke-width:0.6; }
  .cg.hl .curve { stroke-width:3.4; }
  .cg.hl .ext { stroke-width:3; }
  .cg.hl .dot { stroke-width:1.4; }
  .ctl { font-size:12.5px; margin-bottom:8px; display:flex; align-items:center; gap:7px; }
  .ctl input { cursor:pointer; }
  #guide { stroke:#888; stroke-dasharray:4 3; stroke-width:1; visibility:hidden; }
  #tip { position:fixed; pointer-events:none; background:#111; color:#fff; font-size:11.5px;
         padding:7px 9px; border-radius:6px; max-width:240px; visibility:hidden; z-index:9;
         line-height:1.45; box-shadow:0 2px 8px rgba(0,0,0,.25); }
  #tip b { color:#fff; }
  #tip .ti { display:flex; gap:6px; align-items:center; }
  #tip .ti i { width:10px; height:10px; border-radius:2px; display:inline-block; }
  .note { color:var(--muted); font-size:11.5px; margin-top:8px; }
</style>
</head>
<body>
  <h1>Sobrevivência do emprego por categoria — <b>visão MOB</b> (holdout 2023)</h1>
  <p class="sub">Relógio = meses desde a ENTRADA na janela (MOB): pré-existentes entram em jan;
     novos entram no mês de admissão. S(t) = P(continuar empregado após t meses de vínculo observado).
     Clique nas categorias para exibir/ocultar.</p>
  <div class="wrap">
    <div class="chart">
      <svg id="svg" viewBox="0 0 760 470" preserveAspectRatio="xMidYMid meet"></svg>
      <p class="note">Sólido + bolinhas = observado (12 MOB, Kaplan-Meier com entrada tardia).
        Tracejado = extrapolação Weibull (regressão pura, até 36 MOB) — projeção. "med~" = mediana projetada.</p>
    </div>
    <div class="panel">
      <div class="btns">
        <button onclick="setAll(true)">Todos</button>
        <button onclick="setAll(false)">Nenhum</button>
        <button onclick="invert()">Inverter</button>
      </div>
      <div class="btns" id="grpbtns"></div>
      <label class="ctl"><input type="checkbox" id="extchk" onchange="toggleExt(this.checked)">
        Extrapolação Weibull (até 36 MOB) <span style="color:#666">— linha tracejada</span></label>
      <div class="list" id="list"></div>
    </div>
  </div>
  <div id="tip"></div>

<script>
const DATA = __DATA__;
const GROUPS = __GROUPS__;
const H = __H__;
const H_EXT = __HEXT__;
let showExt=false;
let Hview=H;
const W=760, HT=470, M={l:58,r:14,t:14,b:42};
const PW=W-M.l-M.r, PH=HT-M.t-M.b;
const xPix = m => M.l + (m/Hview)*PW;
let yMin=0, yMax=1;
const yPix = s => M.t + (1-(s-yMin)/(yMax-yMin))*PH;
const svg = document.getElementById('svg');
const NS='http://www.w3.org/2000/svg';
function el(t,a){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e;}

const visible=new Set(DATA.map(s=>s.k));
const paths={};
const guide=el('line',{id:'guide',y1:M.t,y2:M.t+PH});

function computeDomain(){
  if(!visible.size){ yMin=0; yMax=1; return; }
  let lo=1;
  DATA.forEach(s=>{ if(!visible.has(s.k)) return;
    for(const v of s.S) if(v<lo) lo=v;
    if(showExt) for(let m=H;m<=H_EXT;m++) if(s.W[m]<lo) lo=s.W[m];
  });
  const pad=0.04*(1-lo)+0.005;
  yMin=Math.max(0, lo-pad); yMax=1.0;
}

function drawAxes(){
  const range=yMax-yMin, dec = range<0.04 ? 3 : 2, NT=6;
  for(let i=0;i<=NT;i++){ const s=yMin+range*i/NT, y=yPix(s);
    svg.appendChild(el('line',{class:'grid',x1:M.l,y1:y,x2:W-M.r,y2:y}));
    const tx=el('text',{class:'tick',x:M.l-8,y:y+3,'text-anchor':'end'}); tx.textContent=s.toFixed(dec); svg.appendChild(tx);
  }
  const xstep = Hview>12 ? 3 : 1;
  for(let m=0;m<=Hview;m+=xstep){ const x=xPix(m);
    svg.appendChild(el('line',{class:'grid',x1:x,y1:M.t,x2:x,y2:M.t+PH}));
    const tx=el('text',{class:'tick',x:x,y:M.t+PH+16,'text-anchor':'middle'}); tx.textContent=m; svg.appendChild(tx);
  }
  if(showExt){ const xv=xPix(H);
    svg.appendChild(el('line',{class:'axis',x1:xv,y1:M.t,x2:xv,y2:M.t+PH,'stroke-dasharray':'2 3',stroke:'#999'})); }
  svg.appendChild(el('line',{class:'axis',x1:M.l,y1:M.t,x2:M.l,y2:M.t+PH}));
  svg.appendChild(el('line',{class:'axis',x1:M.l,y1:M.t+PH,x2:W-M.r,y2:M.t+PH}));
  const yl=el('text',{class:'alabel','text-anchor':'middle',transform:`translate(15,${M.t+PH/2}) rotate(-90)`});
  yl.textContent='S(t) = P(continuar empregado)'; svg.appendChild(yl);
  const xl=el('text',{class:'alabel',x:M.l+PW/2,y:HT-6,'text-anchor':'middle'});
  xl.textContent='MOB — meses desde a entrada na janela'; svg.appendChild(xl);
}

function linePath(arr,m0,m1){
  let d=`M ${xPix(m0)} ${yPix(arr[m0])}`;
  for(let m=m0+1;m<=m1;m++){ d+=` L ${xPix(m)} ${yPix(arr[m])}`; }
  return d;
}
function drawCurves(){
  for(const k in paths) delete paths[k];
  DATA.forEach(s=>{
    if(!visible.has(s.k)) return;
    const g=el('g',{class:'cg','data-k':s.k});
    if(showExt){ g.appendChild(el('path',{class:'ext',d:linePath(s.W,H,H_EXT),stroke:s.cor})); }
    g.appendChild(el('path',{class:'curve',d:linePath(s.S,0,H),stroke:s.cor}));
    for(let m=0;m<=H;m++){
      g.appendChild(el('circle',{class:'dot',cx:xPix(m),cy:yPix(s.S[m]),r:3,fill:s.cor}));
    }
    paths[s.k]=g; svg.appendChild(g);
  });
}

function render(){ Hview = showExt ? H_EXT : H; svg.innerHTML=''; computeDomain(); drawAxes(); drawCurves(); svg.appendChild(guide); }
function toggleExt(on){ showExt=on; render(); }

function fmt(v){ return Math.round(v); }
function buildList(){
  const list=document.getElementById('list');
  DATA.forEach(s=>{
    const row=el2('label','row'); row.dataset.k=s.k;
    row.innerHTML=`<input type="checkbox" checked data-k="${s.k}">
      <span class="sw" style="background:${s.cor}"></span>
      <div class="txt">
        <span class="t1">Cat ${s.k} · risco ${s.risco12}%</span>
        <span class="t2">med ${fmt(s.medm)}m · IQR ${fmt(s.q1)}–${fmt(s.q3)}m · ${s.pct_novo}% novos</span>
      </div>`;
    row.querySelector('input').addEventListener('change',e=>toggle(s.k,e.target.checked));
    row.addEventListener('mouseenter',()=>highlight(s.k,true));
    row.addEventListener('mouseleave',()=>highlight(s.k,false));
    list.appendChild(row);
  });
}
function el2(t,c){const e=document.createElement(t);e.className=c;return e;}

function buildGroupBtns(){
  const c=document.getElementById('grpbtns');
  GROUPS.forEach(gr=>{
    const b=document.createElement('button');
    b.textContent=gr.nome; b.style.borderColor=gr.cor; b.style.color=gr.cor; b.style.fontWeight='600';
    b.title=`Categorias ${gr.cats[0]}–${gr.cats[gr.cats.length-1]}`;
    b.onclick=()=>selectCats(gr.cats);
    c.appendChild(b);
  });
}
function syncChecks(){ DATA.forEach(s=>{ const c=document.querySelector(`input[data-k="${s.k}"]`); if(c)c.checked=visible.has(s.k); }); }
function toggle(k,on){ if(on)visible.add(k); else visible.delete(k); render(); }
function highlight(k,on){ if(paths[k]) paths[k].classList.toggle('hl', on && visible.has(k)); }
function selectCats(cats){ visible.clear(); cats.forEach(k=>visible.add(k)); syncChecks(); render(); }
function setAll(on){ visible.clear(); if(on) DATA.forEach(s=>visible.add(s.k)); syncChecks(); render(); }
function invert(){ DATA.forEach(s=>{ if(visible.has(s.k)) visible.delete(s.k); else visible.add(s.k); }); syncChecks(); render(); }

const tip=document.getElementById('tip');
svg.addEventListener('mousemove',ev=>{
  const r=svg.getBoundingClientRect();
  const sx=(ev.clientX-r.left)*(W/r.width);
  let m=Math.round((sx-M.l)/PW*Hview); m=Math.max(0,Math.min(Hview,m));
  if(sx<M.l-4||sx>W-M.r+4){ tip.style.visibility='hidden'; guide.style.visibility='hidden'; return; }
  guide.setAttribute('x1',xPix(m)); guide.setAttribute('x2',xPix(m)); guide.style.visibility='visible';
  const val = s => m<=H ? s.S[m] : s.W[m];
  const vis=DATA.filter(s=>visible.has(s.k)).sort((a,b)=>val(b)-val(a));
  if(!vis.length){ tip.style.visibility='hidden'; return; }
  let html=`<b>MOB ${m}</b>${m>H?' (Weibull)':''} &nbsp; S(t):<br>`;
  vis.forEach(s=>{ html+=`<div class="ti"><i style="background:${s.cor}"></i>Cat ${s.k}: <b>${(val(s)*100).toFixed(1)}%</b> <span style="color:#bbb">· med ${fmt(s.medm)}m (IQR ${fmt(s.q1)}–${fmt(s.q3)})</span></div>`; });
  tip.innerHTML=html;
  tip.style.left=Math.min(ev.clientX+14, window.innerWidth-250)+'px';
  tip.style.top=(ev.clientY+14)+'px'; tip.style.visibility='visible';
});
svg.addEventListener('mouseleave',()=>{ tip.style.visibility='hidden'; guide.style.visibility='hidden'; });

buildList(); buildGroupBtns(); render();
</script>
</body>
</html>
"""

HTML = (HTML.replace("__DATA__", DATA)
            .replace("__GROUPS__", GROUPS_JSON)
            .replace("__HEXT__", str(H_EXT))
            .replace("__H__", str(H)))
with open(TMP, "w", encoding="utf-8") as f:
    f.write(HTML)
shutil.copy(TMP, OUT)
print(f"FIM -> {OUT} ({len(HTML)/1024:.1f} KB) | {len(ks)} categorias")
