// axon-chat: conversa fundamentada nos experts.

const $ = (id) => document.getElementById(id);
const conversa = $("conversa");
let corrente = null;
let modelo = null;

const rolar = () => (conversa.scrollTop = conversa.scrollHeight);

function bloco(classe, quem, texto) {
  const d = document.createElement("div");
  d.className = "msg " + classe;
  if (quem) {
    const q = document.createElement("div");
    q.className = "quem";
    q.textContent = quem;
    d.appendChild(q);
  }
  const c = document.createElement("div");
  c.className = "corpo";
  c.textContent = texto || "";
  d.appendChild(c);
  conversa.appendChild(d);
  rolar();
  return c;
}

async function atualizarEstado() {
  try {
    const s = await window.axon.saude();
    const e = s.experts || {};
    $("e-experts").textContent =
      e.estado === "pronto"
        ? `${(e.carregados || []).length} expert(s)`
        : `carregando ${(e.carregados || []).length}/${e.total || "?"}`;
    $("e-experts").className = e.estado === "pronto" ? "ok" : "";
    $("e-ollama").textContent = s.ollama ? "ollama no ar" : "ollama fora";
    $("e-ollama").className = s.ollama ? "ok" : "ruim";
    const sel = $("modelo");
    const lista = s.modelos || [];
    if (sel.options.length !== lista.length) {
      const escolhido = sel.value;
      sel.innerHTML = "";
      lista.forEach((m) => {
        const o = document.createElement("option");
        o.value = o.textContent = m;
        sel.appendChild(o);
      });
      if (lista.includes(escolhido)) sel.value = escolhido;
    }
    modelo = sel.value || lista[0] || null;
  } catch {
    $("e-experts").textContent = "backend fora";
    $("e-experts").className = "ruim";
  }
}

function terminar() {
  corrente = null;
  $("enviar").hidden = false;
  $("parar").hidden = true;
  const p = conversa.querySelector(".pensando");
  if (p) p.remove();
}

window.axon.ouvir((ev) => {
  switch (ev.tipo) {
    case "boot":
      atualizarEstado();
      break;

    case "rota": {
      const d = document.createElement("div");
      d.className = "rota";
      const cam = (ev.passagens || [])
        .slice(0, 2)
        .map((p) => p.caminho.join(" › "))
        .join("  ·  ");
      d.innerHTML = `<b>${ev.expert}</b>${cam ? "  " + cam : ""}`;
      conversa.appendChild(d);
      $("rota-atual").textContent = ev.expert;
      rolar();
      break;
    }

    case "delta":
      if (!corrente || !corrente.elemento) {
        const p = conversa.querySelector(".pensando");
        if (p) p.remove();
        corrente = corrente || {};
        corrente.elemento = bloco("ia", "axon", "");
      }
      corrente.elemento.textContent += ev.texto;
      rolar();
      break;

    case "fim":
      $("e-tokens").textContent =
        `${ev.tokens_entrada} / ${ev.tokens_saida} tokens · ` +
        `${(ev.ms_prefill / 1000).toFixed(0)}s de prefill`;
      terminar();
      break;

    case "cancelado":
      bloco("erro", null, "cancelado");
      terminar();
      break;

    case "erro":
      bloco("erro", "erro", ev.erro);
      terminar();
      break;
  }
});

$("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const campo = $("campo");
  const texto = campo.value.trim();
  if (!texto || corrente) return;

  const vazio = conversa.querySelector(".vazio");
  if (vazio) vazio.remove();

  bloco("eu", "você", texto);
  campo.value = "";
  campo.style.height = "auto";
  $("enviar").hidden = true;
  $("parar").hidden = false;

  const d = document.createElement("div");
  d.className = "pensando";
  d.textContent = "roteando…";
  conversa.appendChild(d);
  rolar();

  const r = await window.axon.perguntar(texto, modelo);
  corrente = { run_id: r.run_id, elemento: null };
  d.textContent = "o modelo está lendo o material (isto demora em CPU)…";
});

$("parar").addEventListener("click", async () => {
  if (corrente && corrente.run_id) await window.axon.cancelar(corrente.run_id);
});

$("campo").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("form").requestSubmit();
  }
});
$("campo").addEventListener("input", (e) => {
  e.target.style.height = "auto";
  e.target.style.height = Math.min(e.target.scrollHeight, 128) + "px";
});

atualizarEstado();
setInterval(atualizarEstado, 5000);
