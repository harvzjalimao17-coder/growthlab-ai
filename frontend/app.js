"use strict";

const API_BASE = window.location.origin;

const pages = {

  chat: document.getElementById("chatPage"),

  documents: document.getElementById("documentsPage"),

  settings: document.getElementById("settingsPage")

};

const pageTitles = {

  chat: "AI Chat",

  documents: "Documents",

  settings: "Settings"

};

const navButtons = document.querySelectorAll(".nav-item");

const pageTitle = document.getElementById("pageTitle");

const sidebar = document.getElementById("sidebar");

const menuButton = document.getElementById("menuButton");

const chatForm = document.getElementById("chatForm");

const chatInput = document.getElementById("chatInput");

const chatMessages = document.getElementById("chatMessages");

const sendButton = document.getElementById("sendButton");

const fileInput = document.getElementById("fileInput");

const uploadButton = document.getElementById("uploadButton");

const uploadStatus = document.getElementById("uploadStatus");

const refreshDocuments = document.getElementById("refreshDocuments");

const documentList = document.getElementById("documentList");

const documentCount = document.getElementById("documentCount");

const statusDot = document.getElementById("statusDot");

const connectionText = document.getElementById("connectionText");

const settingsConnection = document.getElementById("settingsConnection");

let isSending = false;

let growthlabSessionId = sessionStorage.getItem("growthlab_ai_session_id") || null;

let isUploading = false;

// --------------------------------------------------

// Voice assistant state

// --------------------------------------------------

const SpeechRecognition =

  window.SpeechRecognition || window.webkitSpeechRecognition;

const speechSupported = Boolean(SpeechRecognition);

const voiceOutputSupported = "speechSynthesis" in window;

let recognition = null;

let isListening = false;

let recognitionRequested = false;

let voiceRepliesEnabled = voiceOutputSupported && localStorage.getItem("growthlab_ai_voice_enabled") === "true";

let speechFailed = false;

let recognitionBaseText = "";

let finalTranscript = "";

let interimTranscript = "";

let manuallyStopped = false;

// The microphone never sends messages automatically.

// Speech is placed in the input for user review.

const voiceButton = document.createElement("button");

voiceButton.type = "button";

voiceButton.id = "voiceButton";

voiceButton.textContent = "🎙 Mic";

voiceButton.title = "Start or stop speech recognition";

voiceButton.setAttribute("aria-label", "Start voice input");

const voiceToggle = document.createElement("button");

voiceToggle.type = "button";

voiceToggle.id = "voiceToggle";

voiceToggle.title = "Toggle spoken AI replies";

voiceToggle.setAttribute("aria-label", "Toggle voice replies");

const voiceStatus = document.createElement("div");

voiceStatus.id = "voiceStatus";

voiceStatus.setAttribute("role", "status");

voiceStatus.setAttribute("aria-live", "polite");

function applyVoiceButtonStyle(button) {

  button.style.border = "1px solid #474747";

  button.style.borderRadius = "10px";

  button.style.background = "#202020";

  button.style.color = "#ffffff";

  button.style.padding = "10px 13px";

  button.style.cursor = "pointer";

  button.style.fontSize = "15px";

  button.style.fontWeight = "600";

  button.style.whiteSpace = "nowrap";

}

applyVoiceButtonStyle(voiceButton);

applyVoiceButtonStyle(voiceToggle);

voiceStatus.style.fontSize = "12px";

voiceStatus.style.color = "#b8b8b8";

voiceStatus.style.marginTop = "6px";

voiceStatus.style.minHeight = "16px";

function setVoiceStatus(message, isError = false) {

  voiceStatus.textContent = message;

  voiceStatus.style.color = isError ? "#ff7777" : "#b8b8b8";

}

function updateVoiceControls() {

  const recording = isListening || recognitionRequested;

  voiceButton.textContent = recording ? "■ Stop" : "🎙 Mic";

  voiceButton.setAttribute("aria-pressed", String(recording));

  voiceButton.style.background = recording ? "#b91c1c" : "#252a32";

  voiceButton.disabled = !speechSupported || (isSending && !recording);

  voiceButton.style.opacity = voiceButton.disabled ? "0.5" : "1";

  voiceButton.setAttribute(

    "aria-label",

    recording ? "Stop voice input" : "Start voice input"

  );

  voiceToggle.textContent = voiceRepliesEnabled ? "🔊 AI Voice On" : "🔇 AI Voice Off";

  voiceToggle.setAttribute("aria-pressed", String(voiceRepliesEnabled));

  voiceToggle.style.borderColor =

    voiceRepliesEnabled ? "#b91c1c" : "#474747";

  voiceToggle.disabled = !voiceOutputSupported;

  voiceToggle.style.opacity = voiceToggle.disabled ? "0.5" : "1";

}

const voiceControls = document.createElement("div");

voiceControls.style.display = "flex";

voiceControls.style.alignItems = "center";

voiceControls.style.gap = "8px";

voiceControls.style.flexWrap = "wrap";

voiceControls.append(voiceButton, voiceToggle);

sendButton.parentNode.insertBefore(voiceControls, sendButton);

chatForm.insertAdjacentElement("afterend", voiceStatus);

updateVoiceControls();

if (!speechSupported) {

  setVoiceStatus(

    "Speech recognition is unavailable in this browser.",

    true

  );

} else {

  setVoiceStatus(

    "Voice ready. Click Mic, speak, then review and press Send."

  );

}

// --------------------------------------------------

// Navigation

// --------------------------------------------------

function showPage(pageName) {

  if (!pages[pageName]) {

    return;

  }

  for (const [name, page] of Object.entries(pages)) {

    page.classList.toggle("active", name === pageName);

  }

  navButtons.forEach((button) => {

    const active = button.dataset.page === pageName;

    button.classList.toggle("active", active);

    button.setAttribute("aria-current", active ? "page" : "false");

  });

  pageTitle.textContent = pageTitles[pageName];

  sidebar.classList.remove("open");

  if (pageName === "documents") {

    loadDocuments();

  }

  if (pageName === "settings") {

    checkConnection();

  }

  if (pageName === "chat") {

    chatInput.focus();

  }

}

navButtons.forEach((button) => {

  button.addEventListener("click", () => {

    showPage(button.dataset.page);

  });

});

menuButton.addEventListener("click", () => {

  sidebar.classList.toggle("open");

});

// --------------------------------------------------

// API helper

// --------------------------------------------------

async function apiRequest(path, options = {}) {

  const response = await fetch(`${API_BASE}${path}`, options);

  let data;

  try {

    data = await response.json();

  } catch {

    throw new Error("The server returned an invalid response.");

  }

  if (!response.ok) {

    let message = `Request failed (${response.status}).`;

    if (typeof data.detail === "string") {

      message = data.detail;

    } else if (Array.isArray(data.detail)) {

      message = data.detail

        .map((item) => item.msg || "Invalid input")

        .join(", ");

    }

    throw new Error(message);

  }

  return data;

}

// --------------------------------------------------

// Backend connection

// --------------------------------------------------

async function checkConnection() {

  connectionText.textContent = "Checking connection...";

  settingsConnection.textContent = "Checking...";

  statusDot.classList.remove("online", "offline");

  try {

    const data = await apiRequest("/health");

    if (data.status === "ok" && data.ollama === "connected") {

      connectionText.textContent = "AI Connected";

      settingsConnection.textContent = "Connected";

      statusDot.classList.add("online");

    } else {

      throw new Error("AI is not ready.");

    }

  } catch {

    connectionText.textContent = "AI Disconnected";

    settingsConnection.textContent = "Disconnected";

    statusDot.classList.add("offline");

  }

}

// --------------------------------------------------

// Chat interface

// --------------------------------------------------

function removeWelcomeScreen() {

  const welcome = chatMessages.querySelector(".welcome-screen");

  if (welcome) {

    welcome.remove();

  }

}

function scrollChatToBottom() {

  chatMessages.scrollTop = chatMessages.scrollHeight;

}

function addSources(message, sources) {
  if (!Array.isArray(sources) || sources.length === 0) return;
  const content = message.querySelector(".message-content");
  const sourceContainer = document.createElement("div");
  sourceContainer.className = "message-sources";
  sources.forEach((source) => {
    if (source.type === "web" && /^https?:\/\//i.test(source.url || "")) {
      const link = document.createElement("a");
      link.className = "source-chip";
      link.href = source.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = source.title || "Web source";
      link.title = source.snippet || source.url;
      link.style.color = "#ff7777";
      link.style.textDecoration = "underline";
      sourceContainer.appendChild(link);
    } else {
      const chip = document.createElement("div");
      chip.className = "source-chip";
      const name = source.document || "Unknown document";
      const similarity = Number(source.similarity);
      chip.textContent = Number.isFinite(similarity)
        ? `${name} - ${Math.round(similarity * 100)}% match` : name;
      chip.title = source.section || name;
      sourceContainer.appendChild(chip);
    }
  });
  content.appendChild(sourceContainer);
}


// GROWTHLAB_SAFE_MARKDOWN_V1: DOM-only formatting, never inject HTML.
function renderClickableLinks(bubble, text) {
  const value = String(text || "").replace(/\\n/g, "\n");
  bubble.replaceChildren();

  function addInline(parent, value) {
    // Markdown links, bare URLs, inline code, bold and italic.
    const regex = /\[([^\]\r\n]+)\]\((https?:\/\/[^\s)]+)\)|(https?:\/\/[^\s<>"']+)|`([^`]+)`|\*\*([^*\n]+)\*\*|\*([^*\n]+)\*/g;
    let last = 0;
    let match;
    while ((match = regex.exec(value)) !== null) {
      if (match.index > last) parent.appendChild(document.createTextNode(value.slice(last, match.index)));
      if (match[1] || match[3]) {
        const raw = match[2] || match[3];
        const url = raw.replace(/[.,;!?]+$/, "");
        const trailing = raw.slice(url.length);
        try {
          const parsed = new URL(url);
          if (!['http:', 'https:'].includes(parsed.protocol)) throw new Error('Unsafe URL');
          const link = document.createElement('a');
          link.href = parsed.href;
          link.target = '_blank';
          link.rel = 'noopener noreferrer';
          link.textContent = match[1] || url;
          link.style.color = '#ff7777';
          link.style.textDecoration = 'underline';
          parent.appendChild(link);
          if (trailing) parent.appendChild(document.createTextNode(trailing));
        } catch { parent.appendChild(document.createTextNode(match[0])); }
      } else {
        const el = document.createElement(match[4] ? 'code' : match[5] ? 'strong' : 'em');
        el.textContent = match[4] || match[5] || match[6];
        parent.appendChild(el);
      }
      last = regex.lastIndex;
    }
    if (last < value.length) parent.appendChild(document.createTextNode(value.slice(last)));
  }

  const lines = value.split('\n');
  let list = null;
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      list = null;
      const space = document.createElement('div');
      space.style.height = '0.55em';
      bubble.appendChild(space);
      continue;
    }
    const bullet = /^[-*]\s+(.+)$/.exec(trimmed);
    if (bullet) {
      if (!list) {
        list = document.createElement('ul');
        list.style.paddingLeft = '1.4em';
        bubble.appendChild(list);
      }
      const li = document.createElement('li');
      addInline(li, bullet[1]);
      list.appendChild(li);
      continue;
    }
    list = null;
    const heading = /^#{1,4}\s+(.+)$/.exec(trimmed);
    const block = document.createElement(heading ? 'strong' : 'div');
    if (heading) {
      block.style.display = 'block';
      block.style.marginTop = '0.4em';
      addInline(block, heading[1]);
    } else {
      addInline(block, line);
    }
    bubble.appendChild(block);
  }
}

function addMessage(role, text, sources = []) {

  removeWelcomeScreen();

  const message = document.createElement("div");

  message.className = `message ${role}`;

  const avatar = document.createElement("div");

  avatar.className = "message-avatar";

  avatar.textContent = role === "user" ? "U" : "G";

  const content = document.createElement("div");

  content.className = "message-content";

  const label = document.createElement("div");

  label.className = "message-label";

  label.textContent = role === "user" ? "You" : "GrowthLab AI";

  const bubble = document.createElement("div");

  bubble.className = "message-bubble";

  renderClickableLinks(bubble, text);

  content.append(label, bubble);

  message.append(avatar, content);

  addSources(message, sources);

  chatMessages.appendChild(message);

  scrollChatToBottom();

  return message;

}

// --------------------------------------------------

// Voice output

// --------------------------------------------------

function stopSpeaking() {

  if (voiceOutputSupported) {

    window.speechSynthesis.cancel();

  }

}

function speakAnswer(answer) {

  if (!voiceRepliesEnabled || !voiceOutputSupported) {

    return;

  }

  stopSpeaking();

  const spokenText = String(answer)

    .replace(/https?:\/\/\S+/g, "our website")

    .replace(/[*#`_]/g, "")

    .slice(0, 2500)

    .trim();

  if (!spokenText) {

    return;

  }

  const utterance = new SpeechSynthesisUtterance(spokenText);

  utterance.lang = "en-US";

  utterance.rate = 0.95;

  utterance.pitch = 1;

  utterance.volume = 1;

  utterance.onstart = () => {

    setVoiceStatus("GrowthLab AI is speaking...");

  };

  utterance.onend = () => {

    setVoiceStatus(

      "Voice ready. Click Mic, speak, then review and press Send."

    );

  };

  utterance.onerror = () => {

    setVoiceStatus("Voice playback stopped or unavailable.");

  };

  window.speechSynthesis.speak(utterance);

}

// --------------------------------------------------

// Send question to existing FastAPI endpoint

// --------------------------------------------------

async function sendQuestion(question) {

  const trimmedQuestion = String(question || "").trim();

  if (!trimmedQuestion || isSending) {

    return;

  }

  if (trimmedQuestion.length > 2000) {

    setVoiceStatus(

      "Questions must be 2,000 characters or less.",

      true

    );

    return;

  }

  // Do not send while microphone is recording.

  // Finish recognition first, then press Send.

  if (isListening || recognitionRequested) {

    setVoiceStatus(

      "Please click Stop and review your words before sending."

    );

    return;

  }

  stopSpeaking();

  isSending = true;

  sendButton.disabled = true;

  chatInput.disabled = true;

  updateVoiceControls();

  addMessage("user", trimmedQuestion);

  chatInput.value = "";

  resizeChatInput();

  const loadingMessage = addMessage("assistant", "Preparing response...");

  setVoiceStatus("GrowthLab AI is preparing an answer...");

  try {

    const response = await fetch(`${API_BASE}/ask/stream`, {

      method: "POST",

      headers: { "Content-Type": "application/json" },

      body: JSON.stringify({ question: trimmedQuestion, session_id: growthlabSessionId })

    });

    if (!response.ok || !response.body) throw new Error(`Request failed (${response.status})`);

    const bubble = loadingMessage.querySelector(".message-bubble");

    const reader = response.body.getReader();

    const decoder = new TextDecoder();

    let buffer = "";

    let generated = "";

    let completed = false;

    while (true) {

      const { value, done } = await reader.read();

      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");

      buffer = lines.pop();

      for (const line of lines) {

        if (!line.trim()) continue;

        const event = JSON.parse(line);

        if (event.type === "status" && !generated) {

          bubble.textContent = event.message;

        } else if (event.type === "token") {

          generated += event.text || "";

          renderClickableLinks(bubble, generated);

          scrollChatToBottom();

        } else if (event.type === "done") {

          completed = true;

          const answer = event.answer || generated || "No answer was returned.";

          renderClickableLinks(bubble, answer);

          addSources(loadingMessage, event.sources);

          if (event.session_id) {

            growthlabSessionId = event.session_id;

            sessionStorage.setItem("growthlab_ai_session_id", growthlabSessionId);

          }

          if (voiceRepliesEnabled) speakAnswer(answer);

          else setVoiceStatus("Answer ready.");

        } else if (event.type === "error") {

          throw new Error(event.message || "AI request failed");

        }

      }

    }

    if (!completed) throw new Error("The response ended before completion.");

  } catch (error) {

    const bubble = loadingMessage.querySelector(".message-bubble");

    bubble.textContent =

      `Unable to get an answer: ${error.message}`;

    setVoiceStatus(

      `AI request failed: ${error.message}`,

      true

    );

  } finally {

    isSending = false;

    sendButton.disabled = false;

    chatInput.disabled = false;

    updateVoiceControls();

    chatInput.focus();

    scrollChatToBottom();

  }

}

function resizeChatInput() {

  chatInput.style.height = "auto";

  chatInput.style.height =

    `${Math.min(chatInput.scrollHeight, 160)}px`;

}

chatInput.addEventListener("input", resizeChatInput);

chatInput.addEventListener("keydown", (event) => {

  if (event.key === "Enter" && !event.shiftKey) {

    event.preventDefault();

    if (!isSending) {

      chatForm.requestSubmit();

    }

  }

});

chatForm.addEventListener("submit", (event) => {

  event.preventDefault();

  sendQuestion(chatInput.value);

});

// --------------------------------------------------

// Improved voice recognition

// --------------------------------------------------

function updateTranscriptionPreview() {

  const speech = [finalTranscript, interimTranscript]

    .filter(Boolean)

    .join(" ")

    .trim();

  chatInput.value = [recognitionBaseText, speech]

    .filter(Boolean)

    .join(" ")

    .trim();

  resizeChatInput();

}

function finishVoiceInput() {

  isListening = false;

  recognitionRequested = false;

  updateVoiceControls();

  // Preserve final recognized speech in the input.

  interimTranscript = "";

  updateTranscriptionPreview();

  if (finalTranscript.trim()) {

    setVoiceStatus(

      "Speech captured. Review or edit your text, then click Send."

    );

  } else if (!speechFailed) {

    setVoiceStatus(

      "No final speech detected. Click Mic to try again."

    );

  }

  chatInput.focus();

}

if (speechSupported) {

  recognition = new SpeechRecognition();

  // Philippine English recognition.

  // Change to en-US later if your browser recognizes it better.

  recognition.lang = "en-PH";

  // Capture longer phrases when supported by the browser.

  recognition.continuous = true;

  recognition.interimResults = true;

  recognition.maxAlternatives = 1;

  recognition.onstart = () => {

    isListening = true;

    recognitionRequested = true;

    speechFailed = false;

    setVoiceStatus(

      "Listening... speak naturally. Click Stop when finished."

    );

    updateVoiceControls();

  };

  recognition.onresult = (event) => {

    let finalText = "";

    let interimText = "";

    for (let i = 0; i < event.results.length; i++) {

      const transcript = event.results[i][0].transcript.trim();

      if (event.results[i].isFinal) {

        finalText += `${transcript} `;

      } else {

        interimText += `${transcript} `;

      }

    }

    finalTranscript = finalText.trim();

    interimTranscript = interimText.trim();

    updateTranscriptionPreview();

    setVoiceStatus(

      finalTranscript

        ? "Speech captured. Continue speaking or click Stop."

        : "Listening and transcribing..."

    );

  };

  recognition.onerror = (event) => {

    // Stopping recognition deliberately is not a failure.

    if (event.error === "aborted") {

      return;

    }

    speechFailed = true;

    const messages = {

      "not-allowed":

        "Microphone permission denied. Allow microphone access in Chrome.",

      "audio-capture":

        "No working microphone detected.",

      "network":

        "Speech recognition service is unavailable. Check your connection.",

      "no-speech":

        "No speech detected. Click Mic to try again."

    };

    setVoiceStatus(

      messages[event.error] ||

        `Speech recognition error: ${event.error}`,

      true

    );

  };

  recognition.onend = () => {

    finishVoiceInput();

    // Important:

    // Do NOT call sendQuestion() here.

    // The user must press Send manually.

  };

}

voiceButton.addEventListener("click", () => {

  if (!recognition) {

    setVoiceStatus(

      "Speech recognition is not supported.",

      true

    );

    return;

  }

  if (isListening || recognitionRequested) {

    manuallyStopped = true;

    setVoiceStatus(

      "Finishing transcription... please wait."

    );

    try {

      recognition.stop();

    } catch (error) {

      setVoiceStatus(

        `Unable to stop microphone: ${error.message}`,

        true

      );

    }

    return;

  }

  if (isSending) {

    return;

  }

  stopSpeaking();

  // Keep any text already typed or previously recognized.

  recognitionBaseText = chatInput.value.trim();

  finalTranscript = "";

  interimTranscript = "";

  speechFailed = false;

  manuallyStopped = false;

  recognitionRequested = true;

  updateVoiceControls();

  try {

    recognition.start();

  } catch (error) {

    recognitionRequested = false;

    updateVoiceControls();

    setVoiceStatus(

      `Unable to start microphone: ${error.message}`,

      true

    );

  }

});

voiceToggle.addEventListener("click", () => {

  if (!voiceOutputSupported) {

    return;

  }

  voiceRepliesEnabled = !voiceRepliesEnabled;

  localStorage.setItem("growthlab_ai_voice_enabled", String(voiceRepliesEnabled));

  if (!voiceRepliesEnabled) {

    stopSpeaking();

  }

  updateVoiceControls();

  setVoiceStatus(

    voiceRepliesEnabled

      ? "Spoken AI replies enabled."

      : "Spoken AI replies disabled."

  );

});

// --------------------------------------------------

// Document library

// --------------------------------------------------

function formatFileSize(bytes) {

  if (!Number.isFinite(bytes) || bytes < 0) {

    return "Unknown size";

  }

  if (bytes < 1024) {

    return `${bytes} B`;

  }

  if (bytes < 1024 * 1024) {

    return `${(bytes / 1024).toFixed(1)} KB`;

  }

  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;

}

async function loadDocuments() {

  documentList.replaceChildren();

  const loading = document.createElement("p");

  loading.className = "empty-state";

  loading.textContent = "Loading documents...";

  documentList.appendChild(loading);

  try {

    const data = await apiRequest("/documents");

    const documents =

      Array.isArray(data.documents) ? data.documents : [];

    documentCount.textContent =

      `${documents.length} ${documents.length === 1 ? "file" : "files"}`;

    documentList.replaceChildren();

    if (documents.length === 0) {

      const empty = document.createElement("p");

      empty.className = "empty-state";

      empty.textContent = "No documents uploaded yet.";

      documentList.appendChild(empty);

      return;

    }

    documents.forEach((fileRecord) => {

      const item = document.createElement("div");

      item.className = "document-item";

      const icon = document.createElement("div");

      icon.className = "document-icon";

      const filename = fileRecord.filename || "Untitled";

      icon.textContent = filename.toLowerCase().endsWith(".pdf")

        ? "PDF"

        : "TXT";

      const details = document.createElement("div");

      details.className = "document-details";

      const name = document.createElement("div");

      name.className = "document-name";

      name.textContent = filename;

      const size = document.createElement("div");

      size.className = "document-size";

      size.textContent =

        formatFileSize(Number(fileRecord.size_bytes));

      details.append(name, size);

      const ready = document.createElement("span");

      ready.className = "document-ready";

      ready.textContent = "Available";

      item.append(icon, details, ready);

      documentList.appendChild(item);

    });

  } catch (error) {

    documentCount.textContent = "Unavailable";

    documentList.replaceChildren();

    const message = document.createElement("p");

    message.className = "empty-state";

    message.textContent =

      `Could not load documents: ${error.message}`;

    documentList.appendChild(message);

  }

}

uploadButton.addEventListener("click", () => {

  if (!isUploading) {

    fileInput.click();

  }

});

fileInput.addEventListener("change", async () => {

  const file = fileInput.files?.[0];

  if (!file || isUploading) {

    return;

  }

  const extension = file.name.split(".").pop()?.toLowerCase();

  if (!["pdf", "txt"].includes(extension)) {

    uploadStatus.textContent =

      "Only PDF and TXT files are supported.";

    fileInput.value = "";

    return;

  }

  if (file.size > 5 * 1024 * 1024) {

    uploadStatus.textContent =

      "File must be 5 MB or smaller.";

    fileInput.value = "";

    return;

  }

  isUploading = true;

  uploadButton.disabled = true;

  uploadButton.textContent = "Uploading...";

  uploadStatus.textContent = `Uploading ${file.name}...`;

  try {

    const formData = new FormData();

    formData.append("file", file);

    const data = await apiRequest("/documents/upload", {

      method: "POST",

      body: formData

    });

    uploadStatus.textContent =

      `Uploaded successfully: ${data.original_filename || file.name}`;

    await loadDocuments();

  } catch (error) {

    uploadStatus.textContent =

      `Upload failed: ${error.message}`;

  } finally {

    isUploading = false;

    uploadButton.disabled = false;

    uploadButton.textContent = "Choose File";

    fileInput.value = "";

  }

});

refreshDocuments.addEventListener("click", loadDocuments);

// --------------------------------------------------

// Initialization

// --------------------------------------------------

checkConnection();

// Start a fresh conversation. Existing backend history is left untouched.

const newChatButton = document.getElementById("newChatButton");

if (newChatButton) newChatButton.addEventListener("click", () => {

  if (isSending) return;

  stopSpeaking();

  growthlabSessionId = null;

  sessionStorage.removeItem("growthlab_ai_session_id");

  chatMessages.replaceChildren();

  const welcome = document.getElementById("welcomeTemplate");

  if (welcome) chatMessages.appendChild(welcome.content.cloneNode(true));

  chatInput.value = "";

  resizeChatInput();

  sidebar.classList.remove("open");

  chatInput.focus();

});

// Event delegation also supports welcome prompts inserted by New Chat.

chatMessages.addEventListener("click", (event) => {

  const suggestion = event.target.closest(".suggestion");

  if (suggestion) sendQuestion(suggestion.dataset.question || "");

});


// --------------------------------------------------
// GrowthLab AI local image generator
// --------------------------------------------------
const imageButton = document.createElement("button");
imageButton.type = "button";
imageButton.textContent = "🎨 Generate Image";
imageButton.title = "Generate an image using local ComfyUI";
applyVoiceButtonStyle(imageButton);
imageButton.style.borderColor = "#b91c1c";
voiceControls.appendChild(imageButton);

const imagePause = (ms) => new Promise(resolve => setTimeout(resolve, ms));
let isGeneratingImage = false;

imageButton.addEventListener("click", async () => {
  const prompt = chatInput.value.trim();
  if (!prompt) {
    setVoiceStatus("Enter an image description first.", true);
    chatInput.focus();
    return;
  }
  if (isSending || isGeneratingImage || isListening || recognitionRequested) {
    setVoiceStatus("Finish the current request or microphone recording first.", true);
    return;
  }
  isGeneratingImage = true;
  imageButton.disabled = true;
  imageButton.textContent = "🎨 Generating...";
  stopSpeaking();
  addMessage("user", "Generate image: " + prompt);
  const responseMessage = addMessage("assistant", "Submitting image to local ComfyUI...");
  const bubble = responseMessage.querySelector(".message-bubble");
  chatInput.value = "";
  resizeChatInput();
  try {
    const job = await apiRequest("/images/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt })
    });
    let completed = false;
    for (let attempt = 0; attempt < 120; attempt++) {
      await imagePause(2000);
      const status = await apiRequest(`/images/status/${encodeURIComponent(job.prompt_id)}`);
      if (status.status === "error") throw new Error(status.message || "Generation failed");
      if (status.status === "completed") {
        bubble.textContent = "Image generated successfully.";
        const img = document.createElement("img");
        img.src = status.image_url;
        img.alt = prompt;
        img.loading = "lazy";
        img.style.cssText = "display:block;max-width:100%;max-height:550px;border-radius:12px;margin-top:12px";
        bubble.appendChild(img);
        const link = document.createElement("a");
        link.href = status.image_url;
        link.download = "GrowthLab_AI_Image.png";
        link.textContent = "Download image";
        link.style.cssText = "display:inline-block;margin-top:10px;color:#ff6666;font-weight:600";
        bubble.appendChild(link);
        completed = true;
        setVoiceStatus("Image ready.");
        scrollChatToBottom();
        break;
      }
      bubble.textContent = `Generating image locally... ${Math.round((attempt + 1) * 2)} seconds elapsed`;
    }
    if (!completed) throw new Error("Generation is taking longer than four minutes. Check ComfyUI.");
  } catch (error) {
    bubble.textContent = `Image generation failed: ${error.message}`;
    setVoiceStatus(error.message, true);
  } finally {
    isGeneratingImage = false;
    imageButton.disabled = false;
    imageButton.textContent = "🎨 Generate Image";
    chatInput.focus();
  }
});


// --------------------------------------------------
// GrowthLab AI image viewer — click any generated image
// --------------------------------------------------
(function setupGrowthLabImageViewer() {
  const overlay = document.createElement("div");
  overlay.id = "growthlabImageViewer";
  overlay.setAttribute("role", "dialog");
  overlay.setAttribute("aria-modal", "true");
  overlay.setAttribute("aria-label", "Generated image preview");
  overlay.setAttribute("aria-hidden", "true");
  overlay.tabIndex = -1;
  overlay.style.cssText = "position:fixed;inset:0;z-index:99999;background:rgba(0,0,0,.94);display:none;flex-direction:column;align-items:center;justify-content:center;padding:16px;box-sizing:border-box;";

  const toolbar = document.createElement("div");
  toolbar.style.cssText = "display:flex;align-items:center;justify-content:flex-end;gap:8px;width:100%;max-width:1100px;margin-bottom:12px;flex-wrap:wrap;";
  const stage = document.createElement("div");
  stage.style.cssText = "width:100%;max-width:1100px;height:calc(100vh - 120px);display:flex;align-items:center;justify-content:center;overflow:auto;";
  const preview = document.createElement("img");
  preview.alt = "Generated image enlarged preview";
  preview.style.cssText = "display:block;max-width:100%;max-height:100%;object-fit:contain;transform-origin:center center;transition:transform .12s ease;";
  stage.appendChild(preview);
  overlay.append(toolbar, stage);
  document.body.appendChild(overlay);

  function makeButton(label, title, handler) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.title = title;
    button.setAttribute("aria-label", title);
    button.style.cssText = "border:1px solid #555;background:#222;color:white;border-radius:9px;padding:9px 13px;cursor:pointer;font:600 14px sans-serif;";
    button.addEventListener("click", handler);
    toolbar.appendChild(button);
    return button;
  }

  let zoom = 1;
  let previousFocus = null;
  function setZoom(value) {
    zoom = Math.min(4, Math.max(0.5, value));
    preview.style.transform = `scale(${zoom})`;
    zoomLabel.textContent = `${Math.round(zoom * 100)}%`;
  }
  function closeViewer() {
    if (overlay.style.display === "none") return;
    overlay.style.display = "none";
    overlay.setAttribute("aria-hidden", "true");
    document.body.style.overflow = overlay.dataset.previousOverflow || "";
    preview.removeAttribute("src");
    if (previousFocus && previousFocus.isConnected) previousFocus.focus();
  }

  makeButton("−", "Zoom out", () => setZoom(zoom - 0.25));
  const zoomLabel = document.createElement("span");
  zoomLabel.style.cssText = "color:white;min-width:52px;text-align:center;font:14px sans-serif;";
  toolbar.appendChild(zoomLabel);
  makeButton("+", "Zoom in", () => setZoom(zoom + 0.25));
  makeButton("Reset", "Reset zoom", () => setZoom(1));
  const download = document.createElement("a");
  download.textContent = "↓ Download";
  download.download = "GrowthLab_AI_Image.png";
  download.style.cssText = "border:1px solid #b91c1c;background:#991b1b;color:white;text-decoration:none;border-radius:9px;padding:9px 13px;font:600 14px sans-serif;";
  toolbar.appendChild(download);
  const closeButton = makeButton("✕", "Close image preview", closeViewer);

  chatMessages.addEventListener("click", (event) => {
    const image = event.target.closest(".message.assistant .message-bubble img");
    if (!image || !image.src) return;
    previousFocus = document.activeElement;
    preview.src = image.currentSrc || image.src;
    preview.alt = image.alt || "Generated image";
    download.href = image.src;
    setZoom(1);
    overlay.dataset.previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    overlay.style.display = "flex";
    overlay.setAttribute("aria-hidden", "false");
    closeButton.focus();
  });
  chatMessages.addEventListener("mouseover", (event) => {
    const image = event.target.closest(".message.assistant .message-bubble img");
    if (image) {
      image.style.cursor = "zoom-in";
      image.title = "Click to view full size";
    }
  });
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay || event.target === stage) closeViewer();
  });
  overlay.addEventListener("wheel", (event) => {
    if (!event.ctrlKey) return;
    event.preventDefault();
    setZoom(zoom + (event.deltaY < 0 ? 0.1 : -0.1));
  }, { passive: false });
  document.addEventListener("keydown", (event) => {
    if (overlay.style.display === "none") return;
    if (event.key === "Escape") closeViewer();
    if (event.key === "+" || event.key === "=") setZoom(zoom + 0.25);
    if (event.key === "-") setZoom(zoom - 0.25);
    if (event.key === "Tab") {
      const focusables = Array.from(toolbar.querySelectorAll("button, a"));
      const first = focusables[0], last = focusables[focusables.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    }
  });
})();
