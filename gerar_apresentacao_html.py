"""Converte a apresentação (gerar_apresentacao.py) para um DECK em HTML autossuficiente
(offline, sem CDN), com navegação por teclado/botões.

- Slides estáticos: renderizados pelo próprio deck (matplotlib) e embutidos como PNG base64.
- Slides B1 e B2 (curvas de sobrevivência e extrapolação Weibull): viram INTERATIVOS —
  o usuário escolhe quais categorias plotar (chips por categoria + botões por grupo de risco),
  com escala-Y dinâmica e tooltip, no mesmo espírito de outputs/sobrevivencia_interativa.html.

Uso:  MPLCONFIGDIR=/tmp/mpl /tmp/consig_venv/bin/python gerar_apresentacao_html.py
Saída: outputs/apresentacao_risco_desligamento.html
"""
import os, runpy, base64, json, shutil
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import pandas as pd
from matplotlib import cm, colors

DUMP = "/tmp/apresentacao_png"
os.environ["DECK_DUMP_PNG"] = DUMP
OUT = "outputs/apresentacao_risco_desligamento.html"
TMP = "/tmp/apresentacao_risco_desligamento.html"

# ---------- 1. roda o deck (gera PDF + dump dos slides em PNG) ----------
print("renderizando slides via gerar_apresentacao.py ...")
ns = runpy.run_path("gerar_apresentacao.py")
NP = len(ns["pages"])
B1, B2 = NP - 3, NP - 2          # índices dos slides interativos (surv_curva, surv_weibull)
print(f"{NP} slides; interativos: B1={B1}, B2={B2}")

def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ---------- 2. dados de sobrevivência (mesma fonte do HTML interativo) ----------
km = pd.read_csv("outputs/tables/sobrevivencia_km_2023.csv")
ext = pd.read_csv("outputs/tables/sobrevivencia_weibull_extrap_2023.csv")
mono = pd.read_csv("outputs/tables/sobrevivencia_weibull_estatisticas_mono_2023.csv").set_index("categoria")
res = pd.read_csv("outputs/tables/sobrevivencia_resumo_2023.csv").set_index("categoria")
ks = sorted(km["categoria"].unique())
cmap = cm.get_cmap("RdYlGn_r"); norm = colors.Normalize(vmin=min(ks), vmax=max(ks))
cor = {k: colors.to_hex(cmap(norm(k))) for k in ks}

series = []
for k in ks:
    S = [round(float(v), 5) for v in km[km.categoria == k].sort_values("mes")["S"].tolist()]
    W = [round(float(v), 5) for v in ext[ext.categoria == k].sort_values("mes")["S_weibull"].tolist()]
    mo = mono.loc[k]
    series.append({"k": int(k), "cor": cor[k], "S": S, "W": W,
                   "risco12": round(float(res.loc[k, "risco_deslig_12m_KM"]) * 100, 1),
                   "q1": round(float(mo["q1_meses_mono"]), 1),
                   "medm": round(float(mo["mediana_meses_mono"]), 1),
                   "q3": round(float(mo["q3_meses_mono"]), 1)})
DATA = json.dumps(series, ensure_ascii=False)
GROUPS = [("Mínimo", [1, 2], "#1a9850"), ("Baixo", [3, 4, 5, 6], "#86cb66"),
          ("Médio-Baixo", [7, 8, 9, 10, 11], "#c9a227"),
          ("Médio", [12, 13, 14, 15, 16, 17], "#fb8d3d"),
          ("Alto", [18, 19, 20, 21, 22, 23], "#d73027")]
GROUPS_JSON = json.dumps([{"nome": n, "cats": c, "cor": col} for n, c, col in GROUPS], ensure_ascii=False)

# ---------- 3. slides interativos (header + bullets + área do gráfico) ----------
NAVY = "#14233f"
def bullets_html(items):
    out = []
    for b, t in items:
        if b:
            out.append(f'<div class="b"><span class="bi">▸</span>{t}</div>')
        else:
            out.append(f'<div class="bh">{t}</div>')
    return "\n".join(out)

B1_TXT = bullets_html([
    (False, "A ideia"),
    (True, "O modelo prevê QUEM/SE é desligado; a sobrevivência mede QUANDO."),
    (True, "S(t) = probabilidade de continuar empregado após t meses."),
    (False, "Dos microdados (RAIS)"),
    (True, "Evento = dispensa s/ justa causa; tempo = mês do desligamento."),
    (True, "Censura: quem fica ativo (ou sai por outro motivo) não é 'evento'."),
    (False, "Kaplan–Meier"),
    (True, "S(t) = Π (nₘ−dₘ)/nₘ — usa a censura sem viés, mês a mês."),
    (True, "RMST(12) = área sob S(t) = meses esperados de emprego no ano."),
])
B2_TXT = bullets_html([
    (False, "O problema"),
    (True, "12 meses de dado não enxergam além de 12m (a curva ainda está alta)."),
    (False, "Solução: forma paramétrica de Weibull"),
    (True, "S(t) = exp(−(t/λ)ᵖ);  hazard ∝ t^(p−1)  (p>1 sobe, p<1 cai)."),
    (True, "Ajuste por regressão pura: ln(−ln S) = p·ln t + ln α (OLS, 12 pts)."),
    (True, "R² médio ≈ 0,99 — extrapola a curva até 36 meses (tracejado)."),
    (False, "Ressalva"),
    (True, "Ignora a sazonalidade de dezembro; projeção >12m é suposição."),
])

def interactive_slide(kicker, title, txt, chart_id):
    return f'''<div class="slide cust">
  <div class="hb"><span class="kick">{kicker}</span><span class="ttl">{title}</span></div>
  <div class="txt">{txt}</div>
  <div class="chartwrap">
    <div class="ctrls">
      <div class="grp" id="grp-{chart_id}"></div>
      <div class="chips" id="chips-{chart_id}"></div>
    </div>
    <svg id="svg-{chart_id}" viewBox="0 0 760 470" preserveAspectRatio="xMidYMid meet"></svg>
  </div>
</div>'''

# ---------- 4. monta todos os slides ----------
slides = []
for i in range(NP):
    if i == B1:
        slides.append(interactive_slide("TEMPO ATÉ O DESLIGAMENTO · SOBREVIVÊNCIA",
                      "Curvas de sobrevivência por categoria (Kaplan-Meier)", B1_TXT, "km"))
    elif i == B2:
        slides.append(interactive_slide("TEMPO ATÉ O DESLIGAMENTO · EXTRAPOLAÇÃO",
                      "Estendendo as curvas além de 12 meses (Weibull)", B2_TXT, "weib"))
    else:
        slides.append(f'<div class="slide"><img class="full" src="data:image/png;base64,{b64(f"{DUMP}/slide_{i:02d}.png")}"></div>')
SLIDES = "\n".join(slides)

# ---------- 5. template HTML ----------
HTML = r"""<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="utf-8">
<title>Risco de Desligamento — apresentação</title>
<style>
  :root{ --navy:#14233f; --ink:#1b2430; --grey:#5b6675; }
  *{box-sizing:border-box;} html,body{margin:0;height:100%;background:#0d1626;font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;}
  .deck{height:100%;display:flex;align-items:center;justify-content:center;}
  /* --u = 1% da LARGURA do palco -> fontes proporcionais ao slide, iguais ao matplotlib */
  .stage{position:relative;width:min(100vw,177.78vh);height:min(56.25vw,100vh);--u:min(1vw,1.7778vh);background:#fff;box-shadow:0 6px 30px rgba(0,0,0,.5);overflow:hidden;}
  .slide{position:absolute;inset:0;display:none;}
  .slide.active{display:block;}
  .full{width:100%;height:100%;object-fit:contain;}
  /* slide interativo */
  .cust{background:#fff;}
  .hb{position:absolute;top:0;left:0;right:0;height:14%;background:var(--navy);border-left:6px solid #f4a722;
      display:flex;flex-direction:column;justify-content:center;padding-left:2.6%;}
  /* tamanhos = mesmos pontos do matplotlib (fig 13,33in): título 20pt, kicker 11,5pt, bullets 12,6pt */
  .kick{color:#9fc0e8;font-weight:bold;letter-spacing:.04em;font-size:calc(var(--u)*1.20);}
  .ttl{color:#fff;font-weight:bold;font-size:calc(var(--u)*2.08);}
  .txt{position:absolute;left:3.5%;top:19%;width:38%;}
  .bh{font-weight:bold;color:var(--ink);font-size:calc(var(--u)*1.31);margin:calc(var(--u)*0.95) 0 calc(var(--u)*0.2);}
  .b{color:var(--ink);font-size:calc(var(--u)*1.31);margin:calc(var(--u)*0.28) 0;padding-left:1.5em;text-indent:-1.5em;line-height:1.3;}
  .bi{color:#f4a722;font-weight:bold;margin-right:.5em;}
  .chartwrap{position:absolute;left:43%;top:16%;width:55%;height:80%;display:flex;flex-direction:column;}
  .ctrls{flex:0 0 auto;margin-bottom:.4vh;}
  .grp{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:4px;}
  .grp button{font-size:clamp(7px,1.15vh,10px);padding:1px 6px;border-radius:5px;border:1px solid #bbb;background:#f7f7f7;cursor:pointer;font-weight:600;}
  .chips{display:flex;flex-wrap:wrap;gap:2px;}
  .chip{width:clamp(12px,1.95vh,18px);height:clamp(12px,1.95vh,18px);border-radius:4px;border:1.5px solid var(--c);
        background:var(--c);color:#fff;font-size:clamp(6px,1.0vh,9px);font-weight:700;cursor:pointer;padding:0;line-height:1;}
  .chip.off{background:#fff;color:#bbb;border-color:#ddd;}
  svg{flex:1 1 auto;width:100%;min-height:0;}
  .grid{stroke:#e6e6e6;stroke-width:1;} .ax{stroke:#999;stroke-width:1;} .tk{fill:#666;font-size:11px;} .al{fill:#1b2430;font-size:12px;}
  .cv{fill:none;stroke-width:1.7;} .ext{fill:none;stroke-width:1.4;stroke-dasharray:5 4;} .dt{stroke:#fff;stroke-width:.5;}
  .bound{stroke:#999;stroke-width:1;stroke-dasharray:2 3;} .guide{stroke:#888;stroke-dasharray:4 3;stroke-width:1;visibility:hidden;}
  /* navegação */
  .nav{position:absolute;top:9px;right:14px;display:flex;align-items:center;gap:9px;z-index:20;
       background:rgba(255,255,255,.16);color:#fff;border:1px solid rgba(255,255,255,.28);border-radius:18px;padding:3px 11px;font-size:12.5px;}
  .nav button{background:none;border:none;color:#fff;font-size:17px;cursor:pointer;line-height:1;padding:0 4px;}
  .nav button:hover{color:#f4a722;}
  #tip{position:fixed;pointer-events:none;background:#111;color:#fff;font-size:11px;padding:6px 8px;border-radius:6px;
       max-width:200px;visibility:hidden;z-index:99;line-height:1.4;box-shadow:0 2px 8px rgba(0,0,0,.3);}
</style></head>
<body>
<div class="deck"><div class="stage" id="stage">
__SLIDES__
  <div class="nav"><button onclick="go(-1)" title="anterior (←)">‹</button><span id="counter"></span><button onclick="go(1)" title="próximo (→)">›</button></div>
</div></div>
<div id="tip"></div>
<script>
const DATA=__DATA__, GROUPS=__GROUPS__;
/* ---------- navegação do deck ---------- */
const slides=[...document.querySelectorAll('.slide')]; let cur=0;
const counter=document.getElementById('counter');
function show(n){ cur=Math.max(0,Math.min(slides.length-1,n));
  slides.forEach((s,i)=>s.classList.toggle('active',i===cur));
  counter.textContent=(cur+1)+' / '+slides.length; }
function go(d){ show(cur+d); }
document.addEventListener('keydown',e=>{ if(e.key==='ArrowRight'||e.key==='PageDown')go(1);
  else if(e.key==='ArrowLeft'||e.key==='PageUp')go(-1);
  else if(e.key==='Home')show(0); else if(e.key==='End')show(slides.length-1); });
show(0);

/* ---------- fábrica de gráfico de sobrevivência interativo ---------- */
const tip=document.getElementById('tip');
function makeChart(svgId, chipsId, grpId, showExt, xmax){
  const svg=document.getElementById(svgId), chips=document.getElementById(chipsId), grp=document.getElementById(grpId);
  const NS='http://www.w3.org/2000/svg', W=760,HT=470,M={l:54,r:12,t:10,b:38},PW=W-M.l-M.r,PH=HT-M.t-M.b,H=12;
  let yMin=0,yMax=1;
  const xPix=m=>M.l+(m/xmax)*PW, yPix=s=>M.t+(1-(s-yMin)/(yMax-yMin))*PH;
  function el(t,a){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e;}
  const visible=new Set(DATA.map(s=>s.k));
  const guide=el('line',{class:'guide',y1:M.t,y2:M.t+PH});
  function domain(){ if(!visible.size){yMin=0;yMax=1;return;} let lo=1;
    DATA.forEach(s=>{ if(!visible.has(s.k))return; for(const v of s.S)if(v<lo)lo=v;
      if(showExt)for(let m=H;m<=xmax;m++)if(s.W[m]<lo)lo=s.W[m]; });
    const pad=0.04*(1-lo)+0.005; yMin=Math.max(0,lo-pad); yMax=1; }
  function axes(){ const range=yMax-yMin,dec=range<0.04?3:2,NT=5;
    for(let i=0;i<=NT;i++){const s=yMin+range*i/NT,y=yPix(s);
      svg.appendChild(el('line',{class:'grid',x1:M.l,y1:y,x2:W-M.r,y2:y}));
      const t=el('text',{class:'tk',x:M.l-6,y:y+3,'text-anchor':'end'});t.textContent=s.toFixed(dec);svg.appendChild(t);}
    const step=xmax>12?3:1;
    for(let m=0;m<=xmax;m+=step){const x=xPix(m);
      svg.appendChild(el('line',{class:'grid',x1:x,y1:M.t,x2:x,y2:M.t+PH}));
      const t=el('text',{class:'tk',x:x,y:M.t+PH+15,'text-anchor':'middle'});t.textContent=m;svg.appendChild(t);}
    if(showExt){const xv=xPix(H);svg.appendChild(el('line',{class:'bound',x1:xv,y1:M.t,x2:xv,y2:M.t+PH}));}
    svg.appendChild(el('line',{class:'ax',x1:M.l,y1:M.t,x2:M.l,y2:M.t+PH}));
    svg.appendChild(el('line',{class:'ax',x1:M.l,y1:M.t+PH,x2:W-M.r,y2:M.t+PH}));
    const yl=el('text',{class:'al','text-anchor':'middle',transform:'translate(14,'+(M.t+PH/2)+') rotate(-90)'});yl.textContent='S(t) = P(continuar empregado)';svg.appendChild(yl);
    const xl=el('text',{class:'al',x:M.l+PW/2,y:HT-4,'text-anchor':'middle'});xl.textContent='meses desde jan/2023';svg.appendChild(xl);
  }
  function lpath(arr,m0,m1){let d='M '+xPix(m0)+' '+yPix(arr[m0]);for(let m=m0+1;m<=m1;m++)d+=' L '+xPix(m)+' '+yPix(arr[m]);return d;}
  function curves(){ DATA.forEach(s=>{ if(!visible.has(s.k))return;
    if(showExt)svg.appendChild(el('path',{class:'ext',d:lpath(s.W,H,xmax),stroke:s.cor}));
    svg.appendChild(el('path',{class:'cv',d:lpath(s.S,0,H),stroke:s.cor}));
    for(let m=0;m<=H;m++)svg.appendChild(el('circle',{class:'dt',cx:xPix(m),cy:yPix(s.S[m]),r:2.4,fill:s.cor})); }); }
  function syncChips(){ chips.querySelectorAll('.chip').forEach(c=>c.classList.toggle('off',!visible.has(+c.dataset.k))); }
  function render(){ svg.innerHTML=''; domain(); axes(); curves(); svg.appendChild(guide); syncChips(); }
  // chips por categoria
  DATA.forEach(s=>{ const b=document.createElement('button'); b.className='chip'; b.dataset.k=s.k; b.textContent=s.k;
    b.style.setProperty('--c',s.cor);
    b.onclick=()=>{ if(visible.has(s.k))visible.delete(s.k); else visible.add(s.k); render(); }; chips.appendChild(b); });
  // botões de grupo + todos/nenhum
  function gbtn(label,fn,col){ const b=document.createElement('button'); b.textContent=label; if(col){b.style.borderColor=col;b.style.color=col;} b.onclick=fn; grp.appendChild(b); }
  gbtn('Todos',()=>{DATA.forEach(s=>visible.add(s.k));render();});
  gbtn('Nenhum',()=>{visible.clear();render();});
  GROUPS.forEach(g=>gbtn(g.nome,()=>{visible.clear();g.cats.forEach(k=>visible.add(k));render();},g.cor));
  // tooltip
  svg.addEventListener('mousemove',ev=>{ const r=svg.getBoundingClientRect(); const sx=(ev.clientX-r.left)*(W/r.width);
    let m=Math.round((sx-M.l)/PW*xmax); m=Math.max(0,Math.min(xmax,m));
    if(sx<M.l-4||sx>W-M.r+4){tip.style.visibility='hidden';guide.style.visibility='hidden';return;}
    guide.setAttribute('x1',xPix(m));guide.setAttribute('x2',xPix(m));guide.style.visibility='visible';
    const val=s=>m<=H?s.S[m]:s.W[m];
    const vis=DATA.filter(s=>visible.has(s.k)).sort((a,b)=>val(b)-val(a));
    if(!vis.length){tip.style.visibility='hidden';return;}
    let html='<b>Mês '+m+'</b>'+(m>H?' (Weibull)':'')+'<br>';
    vis.slice(0,12).forEach(s=>{html+='<span style="color:'+s.cor+'">■</span> Cat '+s.k+': <b>'+(val(s)*100).toFixed(1)+'%</b><br>';});
    if(vis.length>12)html+='… +'+(vis.length-12)+' categorias';
    tip.innerHTML=html; tip.style.left=Math.min(ev.clientX+12,window.innerWidth-190)+'px'; tip.style.top=(ev.clientY+12)+'px'; tip.style.visibility='visible';
  });
  svg.addEventListener('mouseleave',()=>{tip.style.visibility='hidden';guide.style.visibility='hidden';});
  render();
}
makeChart('svg-km','chips-km','grp-km',false,12);
makeChart('svg-weib','chips-weib','grp-weib',true,36);
</script>
</body></html>"""

HTML = (HTML.replace("__SLIDES__", SLIDES)
            .replace("__DATA__", DATA)
            .replace("__GROUPS__", GROUPS_JSON))
with open(TMP, "w", encoding="utf-8") as f:
    f.write(HTML)
shutil.copy(TMP, OUT)
print(f"FIM -> {OUT} ({len(HTML)/1024/1024:.1f} MB, {NP} slides, 2 interativos)")
