// Shopping list editor: progressive enhancement for slr/server/list_web.py.
//
// Without this script every button is a normal form post and the page reloads. With it:
// - forms marked data-ajax are sent in the background and the page is patched in place,
//   keeping keyboard focus on the same control;
// - the list is polled once a second so changes from the bodycam appear on their own;
// - the catalog can be filtered with the search box;
// - voice commands: the "Speak a command" button streams the microphone to the laptop
//   (speech recognition runs there, offline); items are added while the shopper talks, and
//   a summary is spoken when listening stops ("Added chips and milk. 3 items on your list.").
(function () {
  "use strict";

  var items = document.getElementById("items");
  var status = document.getElementById("status");
  var version = items.dataset.version;
  var busy = false;

  // Drop ?msg= from the address so a reload does not show the old message again.
  if (location.search) history.replaceState(null, "", location.pathname);

  function say(message) {
    // Clear first so a repeated identical message is announced again by screen readers.
    status.textContent = "";
    setTimeout(function () { status.textContent = message; }, 50);
  }

  function restoreFocus(key, wasInList) {
    if (!key) return;
    // Try the same control, then a neighbour: "Make current" disappears once its row is
    // current, and a removed row takes its buttons with it.
    var m = /^(current|remove)-(\d+)$/.exec(key);
    var candidates = [key];
    if (m) candidates.push("remove-" + m[2], "current-" + m[2], "remove-" + (m[2] - 1));
    for (var i = 0; i < candidates.length; i++) {
      var el = document.querySelector('[data-focus="' + candidates[i] + '"]');
      if (el && !el.disabled) { el.focus(); return; }
    }
    if (wasInList) document.getElementById("list-heading").focus();
  }

  function apply(state, force) {
    if (state.version === version && !force) return false;
    var active = document.activeElement;
    var key = active && active.dataset ? active.dataset.focus : null;
    var wasInList = items.contains(active);

    version = state.version;
    items.dataset.version = version;
    items.innerHTML = state.html;

    document.querySelectorAll(".tile").forEach(function (tile) {
      var n = state.counts[tile.dataset.item] || 0;
      tile.classList.toggle("on", n > 0);
      tile.querySelector(".count").textContent = !n ? "" : n === 1 ? "on list" : "on list (" + n + ")";
    });
    document.querySelectorAll(".whole button").forEach(function (b) { b.disabled = state.empty; });

    restoreFocus(key, wasInList);
    return true;
  }

  document.addEventListener("submit", function (event) {
    var form = event.target;
    if (!form.hasAttribute("data-ajax")) return;
    event.preventDefault();
    var question = form.getAttribute("data-confirm");
    if (question && !window.confirm(question)) return;

    fetch(form.action, {
      method: "POST",
      headers: { "Accept": "application/json" },
      body: new URLSearchParams(new FormData(form))
    }).then(function (response) {
      if (!response.ok) throw new Error("HTTP " + response.status);
      return response.json();
    }).then(function (reply) {
      apply(reply.state, true);
      say(reply.message);
      if (form.hasAttribute("data-say")) {   // a typed voice command
        speak(reply.message);
        form.reset();
      }
    }).catch(function () {
      form.submit();  // server unreachable: fall back to an ordinary form post
    });
  });

  function refresh() {
    if (busy || document.hidden) return;
    busy = true;
    fetch("/list/state.json", { cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (state) {
        // Not while listening: a screen reader saying "list" could be heard as a command.
        if (state && apply(state) && !rec) say("Shopping list updated.");
      })
      .catch(function () { /* server restarting: try again next tick */ })
      .then(function () { busy = false; });
  }
  setInterval(refresh, 1000);
  document.addEventListener("visibilitychange", function () { if (!document.hidden) refresh(); });

  // Search box: only shown when this script runs.
  var q = document.getElementById("q");
  document.getElementById("search").hidden = false;
  q.addEventListener("input", function () {
    var term = q.value.trim().toLowerCase();
    var any = false;
    document.querySelectorAll(".tile").forEach(function (tile) {
      tile.hidden = term !== "" && tile.dataset.search.indexOf(term) === -1;
      if (!tile.hidden) any = true;
    });
    document.querySelectorAll(".grid").forEach(function (grid) {
      var shown = grid.querySelector(".tile:not([hidden])") !== null;
      grid.hidden = !shown;
      grid.previousElementSibling.hidden = !shown;  // the aisle heading
    });
    document.getElementById("no-match").hidden = any;
  });

  // ------------------------------------------------------------------ voice commands
  //
  // Live listening: press once and talk. The microphone goes to the laptop in quarter-second
  // pieces, and each item goes on the list as soon as the laptop is sure of it. At each pause
  // the page says what changed ("Added 3 water."). When listening stops, the whole list is
  // read back. Press again, Alt+V or Escape to stop; it also stops by itself after a few
  // seconds of quiet.
  //
  // The page must never hear itself: "Added chips" spoken into an open microphone would add
  // chips again. So while the page talks or beeps, silence is sent instead of the microphone,
  // and the screen-reader live region is only updated once listening has stopped.

  var talk = document.getElementById("talk");
  var speakBox = document.getElementById("speak-replies");
  var heardLine = document.getElementById("heard");
  var TARGET_RATE = 16000;     // what the laptop's speech model expects
  var SEND_MS = 250;           // how often a piece of audio is sent
  var QUIET_STOP_MS = 5000;    // stop after this much quiet once the shopper has spoken
  var START_WAIT_MS = 8000;    // ... or this long if they never say anything
  var MAX_MS = 120000;         // hard limit for one press
  var audio = null;            // one AudioContext, created on the first press
  var rec = null;              // the listening session in progress, or null
  var muteUntil = 0;           // send silence until this time (performance.now())

  ["voice", "voice-hint", "speak-option"].forEach(function (id) {
    document.getElementById(id).hidden = false;
  });

  // Remember the "read replies aloud" choice in this browser (storage may be blocked).
  try {
    if (localStorage.getItem("speakReplies") === "0") speakBox.checked = false;
  } catch (e) { /* private window: keep the default */ }
  speakBox.addEventListener("change", function () {
    try { localStorage.setItem("speakReplies", speakBox.checked ? "1" : "0"); } catch (e) { }
  });

  function pickVoice() {
    // Prefer a voice installed on this computer: online voices fail without internet.
    var voices = window.speechSynthesis ? speechSynthesis.getVoices() : [];
    var english = voices.filter(function (v) { return /^en/i.test(v.lang) && v.localService; });
    return english[0] || voices.filter(function (v) { return v.localService; })[0] || null;
  }

  // Speak with the browser's own voice. ``always`` ignores the checkbox: while listening the
  // page must talk itself, because a screen reader cannot be kept out of the microphone.
  function speak(message, always, rate) {
    if (!message || !window.speechSynthesis || (!always && !speakBox.checked)) return;
    speechSynthesis.cancel();
    var u = new SpeechSynthesisUtterance(message);
    var v = pickVoice();
    if (v) u.voice = v;
    if (rate) u.rate = rate;   // live confirmations a little faster, to keep the mic open
    // Mute the microphone for about as long as the sentence takes (some browsers never report
    // the end of speech, which must not leave the microphone muted for good), and stop
    // muting shortly after the voice really finishes.
    var now = performance.now();
    var estimate = now + 600 + message.length * 75 / (rate || 1);
    muteUntil = Math.max(muteUntil, estimate);
    u.onend = u.onerror = function () {
      muteUntil = Math.min(muteUntil, performance.now() + 400);
    };
    speechSynthesis.speak(u);
  }

  function reply(message) {
    say(message);
    speak(message);
  }

  function muted() {
    return performance.now() < muteUntil;
  }

  function tone(freq, ms) {
    if (!audio) return;
    var osc = audio.createOscillator();
    var gain = audio.createGain();
    var t = audio.currentTime;
    osc.frequency.value = freq;
    gain.gain.setValueAtTime(0.0001, t);
    gain.gain.exponentialRampToValueAtTime(0.25, t + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + ms / 1000);
    osc.connect(gain).connect(audio.destination);
    osc.start(t);
    osc.stop(t + ms / 1000 + 0.02);
    muteUntil = Math.max(muteUntil, performance.now() + ms + 150);
  }

  function setListening(on) {
    talk.setAttribute("aria-pressed", on ? "true" : "false");
    talk.classList.toggle("on", on);
  }

  function post(url, body) {
    return fetch(url, {
      method: "POST",
      headers: { "Accept": "application/json", "Content-Type": "application/octet-stream" },
      body: body || ""
    }).then(function (response) {
      if (!response.ok) throw new Error("HTTP " + response.status);
      return response.json();
    });
  }

  function start() {
    if (rec) return;
    if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      reply("The microphone only works when this page is opened as localhost on the laptop. " +
            "You can type a command instead.");
      return;
    }
    if (window.speechSynthesis) speechSynthesis.cancel();
    var Ctx = window.AudioContext || window.webkitAudioContext;
    if (!audio) audio = new Ctx();
    audio.resume();
    var r = rec = { chunks: [], inflight: false, closed: false };
    heardLine.textContent = "";
    Promise.all([
      post("/list/voice/start?rate=" + TARGET_RATE),
      navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true }
      })
    ]).then(function (results) {
      var answer = results[0], stream = results[1];
      if (rec !== r || !answer.ok) {
        stream.getTracks().forEach(function (t) { t.stop(); });
        if (rec === r) { rec = null; reply(answer.message); }
        return;
      }
      r.session = answer.session;
      r.stream = stream;
      r.source = audio.createMediaStreamSource(stream);
      // ScriptProcessor is old but needs no extra file and works in every browser.
      r.node = audio.createScriptProcessor(2048, 1, 1);
      r.rate = audio.sampleRate;
      r.startedAt = r.lastVoice = performance.now();
      r.noise = [];      // loudness of the first few blocks = background noise
      r.loud = 0;        // blocks in a row above the speech threshold
      r.spoke = false;
      r.node.onaudioprocess = function (event) { onAudio(r, event); };
      tone(880, 120);    // "talk now"; the microphone is muted while it plays
      r.source.connect(r.node);
      r.node.connect(audio.destination);   // outputs silence; needed for the callback
      r.timer = setInterval(function () { pump(r); }, SEND_MS);
      setListening(true);
    }).catch(function () {
      if (rec === r) rec = null;
      reply("I can't use the microphone, or the laptop did not answer. Allow microphone access " +
            "for this page, or type a command instead.");
    });
  }

  function onAudio(r, event) {
    if (r.closed) return;
    var data = event.inputBuffer.getChannelData(0);
    var now = performance.now();
    if (muted()) {
      // Our own voice or a tick: send silence of the same length and do not count it.
      r.chunks.push(new Float32Array(data.length));
      r.lastVoice = now;
      return;
    }
    r.chunks.push(new Float32Array(data));
    var sum = 0;
    for (var i = 0; i < data.length; i++) sum += data[i] * data[i];
    var rms = Math.sqrt(sum / data.length);
    if (r.noise.length < 4) { r.noise.push(rms); return; }
    // The quietest early block, in case the shopper started talking right after the beep.
    var threshold = Math.min(0.06, Math.max(0.012, Math.min.apply(null, r.noise) * 2.5));
    if (rms > threshold) {
      r.loud += 1;
      if (r.loud >= 2) { r.spoke = true; r.lastVoice = now; }
    } else {
      r.loud = 0;
    }
    if (r.spoke && now - r.lastVoice > QUIET_STOP_MS) stop();
    else if (!r.spoke && now - r.startedAt > START_WAIT_MS) stop();
    else if (now - r.startedAt > MAX_MS) stop();
  }

  function toPcm16(chunks, rate) {
    // Join the blocks, resample to 16000 per second (linear interpolation), make 16-bit ints.
    var total = 0;
    chunks.forEach(function (c) { total += c.length; });
    var all = new Float32Array(total);
    var at = 0;
    chunks.forEach(function (c) { all.set(c, at); at += c.length; });
    var ratio = rate / TARGET_RATE;
    var n = Math.floor(total / ratio);
    var out = new Int16Array(n);
    for (var i = 0; i < n; i++) {
      var pos = i * ratio;
      var j = Math.floor(pos);
      var frac = pos - j;
      var s = all[j] + ((all[j + 1] || 0) - all[j]) * frac;
      s = Math.max(-1, Math.min(1, s));
      out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return out;
  }

  // Send what was recorded since the last piece. One piece at a time, so they stay in order.
  function pump(r) {
    if (r.inflight || !r.chunks.length) return Promise.resolve();
    var pcm = toPcm16(r.chunks.splice(0), r.rate);
    r.inflight = true;
    return post("/list/voice/chunk?session=" + r.session, pcm.buffer).then(function (answer) {
      if (!answer.ok) { stop(); reply(answer.message); return; }
      handle(answer, true);
    }).catch(function () {
      stop();
      reply("I couldn't reach the laptop. Please try again.");
    }).then(function () {
      r.inflight = false;
      if (r.whenIdle) { var f = r.whenIdle; r.whenIdle = null; f(); }
    });
  }

  // Act on events from the laptop. Item events show on screen at once; "say" events come
  // when the shopper pauses ("Added 3 water.") and are spoken then, while listening.
  // Returns the sentences that were not spoken (used for the read-back at the end).
  function handle(answer, live) {
    if (answer.state) apply(answer.state, true);
    var unspoken = [];
    (answer.events || []).forEach(function (ev) {
      if (ev.kind === "item") {
        heardLine.textContent = ev.message;
      } else if (ev.kind === "say") {
        heardLine.textContent = ev.message;
        if (live) speak(ev.message, true, 1.15);
        else unspoken.push(ev.message);
      }
    });
    return unspoken;
  }

  function stop() {
    var r = rec;
    if (!r) return;
    rec = null;
    r.closed = true;
    clearInterval(r.timer);
    setListening(false);
    if (r.node) { r.node.onaudioprocess = null; r.node.disconnect(); }
    if (r.source) r.source.disconnect();
    if (r.stream) r.stream.getTracks().forEach(function (t) { t.stop(); });
    tone(440, 120);
    if (r.session) finish(r);
  }

  function finish(r) {
    // Wait for the piece on its way, send the rest, then ask for the summary.
    if (r.inflight) { r.whenIdle = function () { finish(r); }; return; }
    pump(r).then(function () {
      return post("/list/voice/end?session=" + r.session);
    }).then(function (answer) {
      var parts = handle(answer, false);
      if (answer.message) parts.push(answer.message);
      if (parts.length) reply(parts.join(" "));
    }).catch(function () {
      reply("I couldn't reach the laptop. Please try again.");
    });
  }

  function toggle() {
    if (rec) stop();
    else start();
  }

  talk.addEventListener("click", toggle);
  document.addEventListener("keydown", function (event) {
    if (event.altKey && (event.key === "v" || event.key === "V" || event.code === "KeyV")) {
      event.preventDefault();
      talk.focus();
      toggle();
    } else if (event.key === "Escape" && rec) {
      stop();
    }
  });
  if (window.speechSynthesis) speechSynthesis.getVoices();   // start loading the voice list
  if (!location.hash) talk.focus();
})();
