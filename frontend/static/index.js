const chat = document.getElementById('chat');
const input = document.getElementById('question-input');
const sendBtn = document.getElementById('send-btn');
const micBtn = document.getElementById('mic-btn');
const inputHint = document.getElementById('input-hint');

// ── Voice Input (Web Speech API - Native Browser STT) ────────────────────────
let recognition = null;
let isListening = false;

const VOICE_HINT = {
  en: { idle: 'Press Enter to send · Shift+Enter for new line', listening: '🔴 Listening… click again to stop' },
  hi: { idle: 'Enter दबाएं भेजने के लिए · Shift+Enter नई लाइन के लिए', listening: '🔴 सुन रहा हूँ… रुकने के लिए क्लिक करें' },
  gu: { idle: 'Enter દબાવો મોકલવા માટે · Shift+Enter નવી લીટી માટે', listening: '🔴 સાંભળી રહ્યો છું… રોકવા માટે ફરીથી ક્લિક કરો' },
};

function toggleVoice() {
  if (isListening) {
    if (recognition) recognition.stop();
    return;
  }

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    alert("Your browser does not support Speech Recognition. Please use Chrome or Edge.");
    return;
  }

  try {
    recognition = new SpeechRecognition();
    const codes = { en: 'en-IN', hi: 'hi-IN', gu: 'gu-IN' };
    recognition.lang = codes[currentLang] || 'en-IN';
    recognition.interimResults = true;
    recognition.continuous = false;

    recognition.onstart = () => {
      isListening = true;
      micBtn.textContent = '⏹';
      micBtn.classList.add('listening');
      input.placeholder = VOICE_HINT[currentLang]?.listening || '🔴 Listening…';
    };

    recognition.onresult = (event) => {
      let currentTranscript = '';
      for (let i = event.resultIndex; i < event.results.length; ++i) {
        currentTranscript += event.results[i][0].transcript;
      }
      
      if (currentTranscript) {
        input.value = currentTranscript;
        autoResize(input);
      }
    };

    recognition.onend = () => {
      stopVoiceUI();
    };

    recognition.onerror = (event) => {
      console.error("Speech Recognition Error:", event.error);
      stopVoiceUI();
    };

    recognition.start();
  } catch (err) {
    console.error("Recognition Start Error:", err);
    stopVoiceUI();
  }
}

function stopVoiceUI() {
  isListening = false;
  micBtn.textContent = '🎙️';
  micBtn.classList.remove('listening');
  const L = LANG_UI[currentLang];
  if (L) input.placeholder = L.placeholder;
  recognition = null;
}

// ── Text-to-Speech ────────────────────────────────────────────────────────────
let currentAudio = null;
let autoReadEnabled = false;

function toggleAutoRead() {
  autoReadEnabled = !autoReadEnabled;
  document.getElementById('auto-read-btn').classList.toggle('on', autoReadEnabled);
  if (!autoReadEnabled && currentAudio) {
    currentAudio.pause();
    currentAudio = null;
  }
}

async function speakText(text, btn, lang) {
  // 1. Stop any current audio and clear ALL active highlights
  if (currentAudio) {
    currentAudio.pause();
    currentAudio = null;
  }

  // Clear highlights from ANY message
  document.querySelectorAll('.bubble.reading-active').forEach(b => b.classList.remove('reading-active'));
  document.querySelectorAll('.scheme-card.reading-active').forEach(c => c.classList.remove('reading-active'));

  // Reset all other speaker buttons
  document.querySelectorAll('.speak-btn').forEach(s => {
    if (s !== btn) {
      s.dataset.speaking = '0';
      s.textContent = '🔊 Listen';
    }
  });

  // If clicked button was already speaking, just stop and return
  if (btn && btn.dataset.speaking === '1') {
    btn.dataset.speaking = '0';
    btn.textContent = '🔊 Listen';
    return;
  }

  const targetLang = lang || currentLang;
  const url = `/tts?text=${encodeURIComponent(text)}&lang=${targetLang}`;
  const audio = new Audio(url);
  currentAudio = audio;

  // Ensure any existing reading state is cleared
  document.querySelectorAll('.reading-active').forEach(el => el.classList.remove('reading-active'));

  // Find the bubble or card associated with this button
  let container = null;
  if (btn) {
    // If it's a bubble, use it. If it's a card, use the WHOLE card
    container = btn.closest('.bubble') || btn.closest('.scheme-card');
  }

  if (container) {
    container.classList.add('reading-active');
  }

  if (btn) {
    btn.textContent = '⌛...';
    btn.dataset.speaking = '1';
    
    audio.onplay = () => { btn.textContent = '⏹ Stop'; };

    const cleanup = () => {
      if (btn) {
        btn.textContent = '🔊 Listen';
        btn.dataset.speaking = '0';
      }
      if (container) container.classList.remove('reading-active');
      document.querySelectorAll('.tts-word.active').forEach(w => w.classList.remove('active'));
      currentAudio = null;
    };

    audio.onended = cleanup;
    audio.onerror = cleanup;
    audio.onpause = cleanup;
  }

  try {
    await audio.play();
  } catch (err) {
    console.error("Playback failed:", err);
    if (btn) {
      btn.textContent = '🔊 Listen';
      btn.dataset.speaking = '0';
    }
  }
}

// ── UI / Sidebar / Auth ───────────────────────────────────────────────────────
function toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  const overlay = document.getElementById('sidebar-overlay');
  sidebar.classList.toggle('hidden');
  if (overlay) {
    overlay.classList.toggle('active', !sidebar.classList.contains('hidden'));
  }
}

function toggleAuthModal() {
  document.getElementById('auth-modal').classList.toggle('active');
}

function toggleDarkMode() {
  const isDark = document.body.classList.toggle('dark-mode');
  localStorage.setItem('theme', isDark ? 'dark' : 'light');
  updateThemeIcon(isDark);
}

function updateThemeIcon(isDark) {
  const icon = document.getElementById('theme-icon');
  if (icon) icon.textContent = isDark ? '☀️' : '🌙';
}

function initTheme() {
  const savedTheme = localStorage.getItem('theme');
  let isDark = false;
  
  if (savedTheme) {
    isDark = savedTheme === 'dark';
  } else {
    isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  }
  
  if (isDark) {
    document.body.classList.add('dark-mode');
  } else {
    document.body.classList.remove('dark-mode');
  }
  updateThemeIcon(isDark);
}

function closeAuthModal(e) {
  if (e.target.id === 'auth-modal') toggleAuthModal();
}

function switchAuthTab(tab) {
  const isLogin = tab === 'login';
  const isSignup = tab === 'signup';
  
  document.getElementById('login-form').classList.toggle('active', isLogin);
  document.getElementById('signup-form').classList.toggle('active', isSignup);
  
  // Hide forgot password forms when switching tabs
  document.getElementById('forgot-email-form').classList.remove('active');
  document.getElementById('forgot-otp-form').classList.remove('active');
  document.getElementById('forgot-reset-form').classList.remove('active');
  
  // Update Tab Buttons
  const tabBtns = document.querySelectorAll('.tab-btn');
  if (tabBtns.length >= 2) {
      tabBtns[0].classList.toggle('active', isLogin);
      tabBtns[1].classList.toggle('active', isSignup);
      // Show tabs if we are in login or signup
      document.querySelector('.modal-tabs').style.display = (isLogin || isSignup) ? 'flex' : 'none';
  }
}

function showForgotPassword(e) {
    if (e) e.preventDefault();
    document.getElementById('login-form').classList.remove('active');
    document.getElementById('signup-form').classList.remove('active');
    document.getElementById('forgot-email-form').classList.add('active');
    document.querySelector('.modal-tabs').style.display = 'none';
}

async function handleForgotPasswordSubmit(e) {
    if (e) e.preventDefault();
    const email = document.getElementById('forgot-email-input').value;
    if (!email) return alert("Please enter your email");

    // Clear any previous attempts from UI if needed
    
    try {
        const res = await fetch('/forgot_password', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email })
        });
        const data = await res.json();
        if (res.ok) {
            document.getElementById('forgot-email-form').classList.remove('active');
            document.getElementById('forgot-otp-form').classList.add('active');
        } else {
            alert(data.error);
        }
    } catch (err) {
        alert("Something went wrong. Please try again.");
        console.error("Forgot Password Error:", err);
    }
}

async function handleVerifyOTP() {
    const email = document.getElementById('forgot-email-input').value;
    const otp = document.getElementById('forgot-otp-input').value;
    if (!otp || otp.length < 6) return alert("Please enter the 6-digit OTP");

    try {
        const res = await fetch('/verify_otp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, otp })
        });
        const data = await res.json();
        if (res.ok) {
            document.getElementById('forgot-otp-form').classList.remove('active');
            document.getElementById('forgot-reset-form').classList.add('active');
        } else {
            alert(data.error);
        }
    } catch (err) {
        alert("Verification failed.");
    }
}

async function handleResetPassword() {
    const email = document.getElementById('forgot-email-input').value;
    const otp = document.getElementById('forgot-otp-input').value;
    const password = document.getElementById('forgot-new-password').value;
    const confirm = document.getElementById('forgot-confirm-password').value;

    if (!password) return alert("Please enter a new password");
    if (password !== confirm) return alert("Passwords do not match");

    try {
        const res = await fetch('/reset_password', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, otp, password })
        });
        const data = await res.json();
        if (res.ok) {
            alert("Password reset successfully! You can now login.");
            switchAuthTab('login');
        } else {
            alert(data.error);
        }
    } catch (err) {
        alert("Failed to reset password.");
    }
}

let currentUser = null;
let currentChatId = null;
let currentChatMessages = [];

async function handleLogin() {
  const email = document.getElementById('login-email').value;
  const password = document.getElementById('login-password').value;
  const res = await fetch('/login', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ email, password })});
  const data = await res.json();
  if (res.ok) { currentUser = data.user; updateUserUI(); toggleAuthModal(); loadHistory(); }
  else alert(data.error);
}

async function handleSignup() {
  const body = {
    full_name: document.getElementById('signup-name').value,
    email: document.getElementById('signup-email').value,
    password: document.getElementById('signup-password').value,
    age: document.getElementById('signup-age').value,
    gender: document.getElementById('signup-gender').value,
    income: document.getElementById('signup-income').value,
    category: document.getElementById('signup-category').value,
    residence: "Gujarat",
    occupation: document.getElementById('signup-occupation').value
  };
  const res = await fetch('/signup', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)});
  const data = await res.json();
  if (res.ok) { currentUser = data.user; updateUserUI(); toggleAuthModal(); loadHistory(); }
  else alert(data.error);
}

function updateUserUI() {
  const display = document.getElementById('username-display');
  const dropdown = document.getElementById('profile-dropdown');
  if (currentUser) {
    display.textContent = currentUser.name;
    // We now let handleProfileClick manage the click
  } else {
    display.textContent = 'Login / Sign Up';
    if (dropdown) dropdown.classList.remove('active');
  }
}

function handleProfileClick() {
    if (!currentUser) {
        toggleAuthModal();
    } else {
        const dropdown = document.getElementById('profile-dropdown');
        dropdown.classList.toggle('active');
    }
}

// Close dropdown when clicking outside
window.addEventListener('click', (e) => {
    const btn = document.getElementById('user-profile-btn');
    const dropdown = document.getElementById('profile-dropdown');
    if (btn && !btn.contains(e.target) && dropdown) {
        dropdown.classList.remove('active');
    }
});

function toggleProfileModal(event) {
  if (event) event.stopPropagation();
  const modal = document.getElementById('profile-modal');
  modal.classList.toggle('active');
  
  if (modal.classList.contains('active') && currentUser) {
    // Note: We need to fetch full profile data from /me to pre-fill
    fetch('/me').then(res => res.json()).then(data => {
        if (data.user) {
            document.getElementById('profile-name').value = data.user.name || "";
            document.getElementById('profile-age').value = data.user.age || "";
            document.getElementById('profile-income').value = data.user.income || "";
            document.getElementById('profile-category').value = data.user.category || "General";
            document.getElementById('profile-gender').value = data.user.gender || "Male";
            document.getElementById('profile-occupation').value = data.user.occupation || "";
            document.getElementById('profile-email-notif').checked = data.user.email_notifications;
        }
    });
  }
}

function closeProfileModal(e) {
  if (e.target.id === 'profile-modal') toggleProfileModal();
}

async function handleUpdateProfile() {
  const body = {
    full_name: document.getElementById('profile-name').value,
    age: document.getElementById('profile-age').value,
    income: document.getElementById('profile-income').value,
    category: document.getElementById('profile-category').value,
    occupation: document.getElementById('profile-occupation').value,
    gender: document.getElementById('profile-gender').value,
    email_notifications: document.getElementById('profile-email-notif').checked
  };

  const res = await fetch('/update_profile', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });

  const data = await res.json();
  if (res.ok) {
    currentUser = data.user;
    updateUserUI();
    toggleProfileModal();
    alert("Profile updated successfully!");
  } else {
    alert(data.error || "Failed to update profile");
  }
}

async function handleLogout(event) {
  if (event) event.stopPropagation();
  if (!confirm("Are you sure you want to log out?")) return;
  
  await fetch('/logout', { method: 'POST' });
  currentUser = null;
  currentChatId = null;
  location.reload();
}

window.onload = async () => {
  initTheme();
  setLang('en');
  updateUserUI();
  const res = await fetch('/me');
  const data = await res.json();
  if (data.user) {
    currentUser = data.user;
    updateUserUI();
    loadHistory();
  }
};

async function loadHistory() {
  if (!currentUser) return;
  const res = await fetch('/get_history');
  const items = await res.json();
  const list = document.getElementById('history-list');
  list.innerHTML = '';
  
  items.forEach(item => {
    const div = document.createElement('div');
    div.className = `history-item ${currentChatId == item.id ? 'active' : ''}`;
    div.onclick = (e) => {
        // Only switch if we didn't click the menu
        if (!e.target.closest('.history-menu-btn') && !e.target.closest('.history-dropdown')) {
            switchChat(item);
        }
    };

    div.innerHTML = `
      <span class="chat-title">${escapeHtml(item.title || 'New Chat')}</span>
      <div class="history-menu-btn" onclick="toggleHistoryMenu(event, ${item.id})">•••</div>
      <div id="dropdown-${item.id}" class="history-dropdown" onclick="event.stopPropagation()">
        <div class="history-dropdown-item" onclick="renameChat(event, ${item.id}, '${escapeJs(item.title || 'New Chat')}')">
          <span>✏️</span> Rename
        </div>
        <div class="history-dropdown-item delete" onclick="deleteChat(event, ${item.id})">
          <span>🗑️</span> Delete
        </div>
      </div>
    `;
    list.appendChild(div);
  });
}

function escapeJs(str) {
  return str.replace(/'/g, "\\'");
}

function toggleHistoryMenu(event, id) {
  event.stopPropagation();
  // Close others
  document.querySelectorAll('.history-dropdown').forEach(d => {
    if (d.id !== `dropdown-${id}`) d.classList.remove('active');
  });
  const d = document.getElementById(`dropdown-${id}`);
  d.classList.toggle('active');
}

// Close history menus when clicking elsewhere
window.addEventListener('click', (e) => {
  if (!e.target.closest('.history-menu-btn')) {
    document.querySelectorAll('.history-dropdown').forEach(d => d.classList.remove('active'));
  }
});

async function renameChat(event, id, oldTitle) {
  event.stopPropagation();
  const newTitle = prompt("Enter new chat name:", oldTitle);
  if (!newTitle || newTitle === oldTitle) return;

  const res = await fetch('/rename_chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ chat_id: id, title: newTitle })
  });

  if (res.ok) {
    loadHistory();
  } else {
    alert("Failed to rename chat");
  }
}

async function deleteChat(event, id) {
  event.stopPropagation();
  if (!confirm("Are you sure you want to delete this chat?")) return;

  const res = await fetch('/delete_chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ chat_id: id })
  });

  if (res.ok) {
    if (currentChatId === id) {
        clearChatUI();
    }
    loadHistory();
  } else {
    alert("Failed to delete chat");
  }
}

function clearChatUI() {
    currentChatId = null;
    currentChatMessages = [];
    chat.innerHTML = `<div class="welcome-card">
        <span class="wheel">☸️</span>
        <h2>New conversation started!</h2>
        <p>Ask me about any Gujarat government scheme.</p>
    </div>`;
}

function switchChat(chatData) {
  currentChatId = chatData.id;
  currentChatMessages = chatData.messages || [];
  chat.innerHTML = '';
  currentChatMessages.forEach(msg => msg.role === 'user' ? addUserMessage(msg.content) : renderResult(msg.result));
  loadHistory();
  if (window.innerWidth < 600) toggleSidebar();
}

async function saveChat(userMsg, aiResult) {
  if (!currentUser) return;
  currentChatMessages.push({ role: 'user', content: userMsg }, { role: 'assistant', result: aiResult });
  
  // Only suggest/send a title on the very first message save of a new chat
  const title = (currentChatId === null) ? currentChatMessages[0].content.substring(0, 30) : null;
  
  const res = await fetch('/save_chat', { 
    method: 'POST', 
    headers: { 'Content-Type': 'application/json' }, 
    body: JSON.stringify({ chat_id: currentChatId, title, messages: currentChatMessages })
  });
  
  const data = await res.json();
  if (res.ok) { currentChatId = data.chat_id; loadHistory(); }
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
}

function sendChip(el) {
  input.value = el.textContent.trim();
  sendMessage();
}

function scrollBottom() {
  setTimeout(() => chat.scrollTop = chat.scrollHeight, 50);
}

function addUserMessage(text) {
  const row = document.createElement('div');
  row.className = 'msg-row user';
  row.innerHTML = `
    <div class="avatar user">👤</div>
    <div class="bubble user">${escapeHtml(text)}</div>`;
  chat.appendChild(row);
  scrollBottom();
}

function addTyping() {
  const row = document.createElement('div');
  row.className = 'msg-row ai typing-row';
  row.id = 'typing';
  row.innerHTML = `
    <div class="avatar ai">🤖</div>
    <div class="typing-dots"><span></span><span></span><span></span></div>`;
  chat.appendChild(row);
  scrollBottom();
  return row;
}

function removeTyping() {
  const t = document.getElementById('typing');
  if (t) t.remove();
}

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>');
}

function renderResult(result) {
  const row = document.createElement('div');
  row.className = 'msg-row ai';

  const avatar = `<div class="avatar ai">🤖</div>`;
  let content = '';
  let attachSpeak = null;

  if (result.type === 'conversational') {
    const replyText = result.reply || '';
    const speakId = 'speak-' + Date.now();
    content = `<div class="bubble ai">
      ${escapeHtml(replyText)}
      <br><button class="speak-btn" id="${speakId}" data-speaking="0">🔊 Listen</button>
    </div>`;
    attachSpeak = { id: speakId, text: replyText };
  }
  else if (result.type === 'names_only') {
    content = `<div class="bubble ai">
      <strong style="color:var(--saffron);font-family:'Rajdhani',sans-serif;font-size:15px;">📋 Government Schemes Found</strong>
      <div style="margin-top:10px;line-height:2">${escapeHtml(result.reply)}</div>
    </div>`;
  }
  else if (result.type === 'specific_field') {
    const field = (result.field || '').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    content = `<div class="bubble ai">
      <strong style="color:var(--saffron);font-family:'Rajdhani',sans-serif;font-size:15px;">📌 ${field}</strong>
      <div style="margin-top:10px">${escapeHtml(result.reply)}</div>
    </div>`;
  }
  else if (result.type === 'full_detail') {
    const wrapper = document.createElement('div');
    wrapper.className = 'schemes-wrapper';
    (result.schemes || []).forEach((s, idx) => {
      const cardHtml = buildSchemeCard(s, idx + 1);
      const temp = document.createElement('div');
      temp.innerHTML = cardHtml;
      const card = temp.firstElementChild;
      
      // Manual binding for Listen button
      const btn = card.querySelector('.speak-btn');
      if (btn) {
        btn.onclick = () => speakText(btn.dataset.tts, btn);
      }
      wrapper.appendChild(card);
    });
    content = wrapper.outerHTML; // Wait, actually I should append wrapper directly
    row.innerHTML = avatar;
    row.appendChild(wrapper);
    chat.appendChild(row);
    scrollBottom();
    return; // handle manually
  }
  else if (result.type === 'eligibility_result') {
    const schemes = result.schemes || [];
    const profile = result.profile || {};
    const profileLines = Object.entries({
      'Age': profile.age, 'Income': profile.income, 'Occupation': profile.occupation,
      'State': profile.state, 'Gender': profile.gender, 'Caste/Category': profile.caste_category
    }).filter(([,v]) => v).map(([k,v]) => `<span style="margin-right:12px">• <b>${k}:</b> ${escapeHtml(String(v))}</span>`).join('');

    const schemeCards = schemes.map((s, i) => `
      <div style="background:var(--ai-bubble);border:1px solid var(--border);border-radius:10px;padding:12px 14px;margin-top:8px;">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
          <span style="background:var(--green);color:white;border-radius:50%;width:22px;height:22px;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;flex-shrink:0">${i+1}</span>
          <strong style="color:var(--saffron);font-size:14px">${escapeHtml(s.scheme_name || '')}</strong>
        </div>
        ${s.why_eligible ? `<div style="color:#138808;font-size:12px;margin-top:4px">✅ ${escapeHtml(s.why_eligible)}</div>` : ''}
        ${s.category ? `<div style="color:var(--muted);font-size:12px;margin-top:3px">📂 ${escapeHtml(s.category)}</div>` : ''}
        ${s.state ? `<div style="color:var(--muted);font-size:12px;margin-top:3px">📍 ${escapeHtml(s.state)}</div>` : ''}
        ${s.official_link && !['not available','n/a','none',''].includes((s.official_link||'').toLowerCase())
          ? `<div style="margin-top:6px"><a href="${escapeHtml(s.official_link)}" target="_blank" class="link-value">🔗 Visit Official Site ↗</a></div>` : ''}
      </div>`).join('');

    content = `<div class="bubble ai">
      <strong style="color:var(--saffron);font-family:'Rajdhani',sans-serif;font-size:15px;">🎯 Eligible Schemes Found (${schemes.length})</strong>
      ${profileLines ? `<div style="margin-top:8px;font-size:12px;color:var(--muted);line-height:1.8">${profileLines}</div>` : ''}
      ${schemeCards}
      ${schemes.length > 0 ? `<div style="margin-top:10px;font-size:12px;color:var(--muted)">💡 Ask for full details of any scheme above.</div>` : ''}
    </div>`;
  }
  else if (result.type === 'eligibility_for_shown') {
    const schemes = result.schemes || [];
    const profile = result.profile || {};
    const profileLines = Object.entries({
      'Age': profile.age, 'Income': profile.income, 'Occupation': profile.occupation,
      'State': profile.state, 'Gender': profile.gender, 'Caste/Category': profile.caste_category
    }).filter(([,v]) => v).map(([k,v]) => `<span style="margin-right:12px">• <b>${k}:</b> ${escapeHtml(String(v))}</span>`).join('');

    const eligible = schemes.filter(s => s.is_eligible);
    const notEligible = schemes.filter(s => !s.is_eligible);

    const schemeRows = schemes.map((s, i) => {
      const icon = s.is_eligible ? '✅' : '❌';
      const color = s.is_eligible ? '#138808' : '#c00';
      return `<div style="display:flex;gap:8px;padding:8px 0;border-bottom:1px solid var(--border);">
        <span style="font-size:15px;flex-shrink:0">${icon}</span>
        <div>
          <div style="font-weight:600;font-size:13px;color:var(--saffron)">${escapeHtml(s.scheme_name || '')}</div>
          <div style="font-size:12px;color:${color};margin-top:2px">${escapeHtml(s.reason || '')}</div>
          ${s.is_eligible && s.official_link && !['not available','n/a','none',''].includes((s.official_link||'').toLowerCase())
            ? `<a href="${escapeHtml(s.official_link)}" target="_blank" class="link-value" style="font-size:12px;margin-top:4px;display:inline-block">🔗 Apply here ↗</a>` : ''}
        </div>
      </div>`;
    }).join('');

    content = `<div class="bubble ai">
      <strong style="color:var(--saffron);font-family:'Rajdhani',sans-serif;font-size:15px;">🎯 Eligibility Check Results</strong>
      ${profileLines ? `<div style="margin-top:8px;font-size:12px;color:var(--muted);line-height:1.8">${profileLines}</div>` : ''}
      <div style="margin-top:10px">${schemeRows}</div>
      <div style="margin-top:10px;font-size:12px;color:var(--muted)">
        📊 <b>${eligible.length}</b> eligible &nbsp;|&nbsp; <b>${notEligible.length}</b> not eligible
      </div>
    </div>`;
  }
  else if (result.error) {
    content = `<div class="bubble ai" style="color:#c00;">⚠️ Error: ${escapeHtml(result.error)}</div>`;
  }
  else {
    // Fallback — should never happen but prevents blank messages
    content = `<div class="bubble ai" style="color:var(--muted);">Received an unexpected response. Please try again.</div>`;
  }

  row.innerHTML = avatar + content;
  chat.appendChild(row);
  scrollBottom();

  const responseLang = result.lang || currentLang;

  if (attachSpeak) {
    const btn = row.querySelector('#' + attachSpeak.id);
    if (btn) btn.onclick = () => speakText(attachSpeak.text, btn, responseLang);
  }

  // Auto-read: fires for ALL response types when toggle is ON
  if (autoReadEnabled) {
    let msgToSpeak = "";
    if (result.type === 'chunk' || result.type === 'conversational') {
      msgToSpeak = result.text || result.reply || '';

    } else if (result.type === 'names_only' && result.reply) {
      msgToSpeak = result.reply;

    } else if (result.type === 'specific_field' && result.reply) {
      const fieldLabel = (result.field || '').replace(/_/g, ' ');
      msgToSpeak = fieldLabel + '. ' + result.reply;

    } else if (result.type === 'full_detail' && result.schemes?.length) {
      msgToSpeak = result.schemes.map((s, i) =>
        `Scheme ${i+1}: ${s.scheme_name || ''}. ` +
        (s.description ? `${s.description}. ` : '') +
        (s.benefits ? `${s.benefits}. ` : '') +
        (s.eligibility ? `${s.eligibility}.` : '')
      ).join(' ');

    } else if (result.type === 'eligibility_result' && result.schemes?.length) {
      msgToSpeak = result.schemes.map((s, i) =>
        `${i+1}: ${s.scheme_name || ''}. ${s.why_eligible || ''}`
      ).join('. ');

    } else if (result.type === 'eligibility_for_shown' && result.schemes?.length) {
      msgToSpeak = result.schemes.map(s =>
        `${s.scheme_name || ''}: ${s.is_eligible ? 'Eligible. ' : 'Not eligible. '}${s.reason || ''}`
      ).join('. ');

    } else if (result.error) {
      msgToSpeak = 'Error: ' + result.error;
    }

    if (msgToSpeak) {
      // Small timeout to ensure DOM is fully ready
      setTimeout(() => {
        const btn = row.querySelector('.speak-btn');
        // Instantly start reading if autoReadEnabled is ON
        if (autoReadEnabled) {
           speakText(msgToSpeak, btn, responseLang);
        }
      }, 50);
    }
  }
}

// ── TTS word-wrap helpers (OBSOLETE with Audio TTS but kept for reference) ────
function wrapWordsInBubble(el) {
  if (!el || el.querySelector('.tts-word')) return; // Already wrapped or invalid
  const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const p = node.parentElement;
      if (!p) return NodeFilter.FILTER_REJECT;
      // Skip interactive or non-visible elements
      if (['BUTTON','A','SCRIPT','STYLE','SPAN'].includes(p.tagName)) {
         if (!p.classList.contains('field-value')) return NodeFilter.FILTER_REJECT;
      }
      if (p.classList.contains('tts-word') || p.classList.contains('step-marker') || p.classList.contains('bullet-marker')) return NodeFilter.FILTER_REJECT;
      if (node.textContent.trim() === '') return NodeFilter.FILTER_REJECT;
      return NodeFilter.FILTER_ACCEPT;
    }
  });
  const nodes = [];
  let current;
  while (current = walker.nextNode()) nodes.push(current);
  
  nodes.forEach(node => {
    const parent = node.parentNode;
    const text = node.textContent;
    // Split by whitespace but keep whitespace
    const parts = text.split(/(\s+)/);
    const fragment = document.createDocumentFragment();
    
    parts.forEach(part => {
      if (/^\s+$/.test(part)) {
        fragment.appendChild(document.createTextNode(part));
      } else if (part) {
        const span = document.createElement('span');
        span.className = 'tts-word';
        span.textContent = part;
        fragment.appendChild(span);
      }
    });
    parent.replaceChild(fragment, node);
  });
}

function formatStructuredText(text) {
  if (!text || ['not available','n/a','none',''].includes(String(text).toLowerCase())) {
    return `<span class="field-value" style="color:var(--muted)">Not available</span>`;
  }

  // Handle both literal \n and <br>
  const lines = String(text).split(/\r?\n|<br>/).filter(l => l.trim().length > 0);
  
  if (lines.length <= 1 && !/^[0-9]+[\.\)]|^\s*[\•\-\*]/.test(text)) {
    return `<span class="field-value">${escapeHtml(text)}</span>`;
  }

  let html = `<div class="structured-list">`;
  lines.forEach(line => {
    const trimmed = line.trim();
    // Match Step 1:, 1., (1)
    const stepMatch = trimmed.match(/^([0-9\u0AB0-\u0AB9]+[\.\)]|Step\s*[0-9]+[:\.]?|પગલું\s*[0-9]+[:\.]?|चरण\s*[0-9]+[:\.]?)\s*(.*)/i);
    // Match •, *, -
    const bulletMatch = trimmed.match(/^([\•\-\*])\s*(.*)/);

    if (stepMatch) {
      html += `<div class="list-item step"><span class="step-marker">${escapeHtml(stepMatch[1])}</span> ${escapeHtml(stepMatch[2])}</div>`;
    } else if (bulletMatch) {
      html += `<div class="list-item bullet"><span class="bullet-marker">●</span> ${escapeHtml(bulletMatch[2])}</div>`;
    } else {
      html += `<div class="list-item">${escapeHtml(trimmed)}</div>`;
    }
  });
  html += `</div>`;
  return html;
}

function unwrapWordsInBubble(el) {
  if (!el) return;
  el.querySelectorAll('.tts-word').forEach(span => {
    span.replaceWith(document.createTextNode(span.textContent));
  });
  el.normalize();
}

function buildSchemeCard(s, i) {
  const link = s.official_link && !['not available','n/a','none',''].includes(s.official_link.toLowerCase())
    ? `<a href="${escapeHtml(s.official_link)}" target="_blank" class="link-value">🔗 Visit Official Site ↗</a>`
    : `<span class="field-value" style="color:var(--muted)">Not available</span>`;

  // Construct text for TTS
  const fullText = `Scheme Name: ${s.scheme_name}. Description: ${s.description}. Benefits: ${s.benefits}. Eligibility: ${s.eligibility}. Documents required: ${s.documents_required}. Application process: ${s.application_process}.`;
  const speakId = `speak-card-${Date.now()}-${i}`;

  return `<div class="scheme-card">
    <div class="scheme-card-header">
      <div class="scheme-number">${i}</div>
      <div class="scheme-title">${escapeHtml(s.scheme_name || 'Unnamed Scheme')}</div>
      ${s.state && s.state.toLowerCase() !== 'not available' ? `<div class="scheme-state-badge">${escapeHtml(s.state)}</div>` : ''}
    </div>
    <div class="scheme-body">
      ${s.category && s.category.toLowerCase() !== 'not available' ? `
      <div class="scheme-field full">
        <span class="field-label">Category</span>
        <span class="category-badge">${escapeHtml(s.category)}</span>
      </div>` : ''}
      <div class="divider"></div>
      <div class="scheme-field full">
        <span class="field-label">Description</span>
        <span class="field-value">${escapeHtml(s.description || 'Not available')}</span>
      </div>
      <div class="scheme-field full">
        <span class="field-label">Benefits</span>
        ${formatStructuredText(s.benefits)}
      </div>
      <div class="scheme-field full">
        <span class="field-label">Eligibility</span>
        ${formatStructuredText(s.eligibility)}
      </div>
      <div class="scheme-field full">
        <span class="field-label">Documents Required</span>
        ${formatStructuredText(s.documents_required)}
      </div>
      <div class="scheme-field full">
        <span class="field-label">Application Process</span>
        ${formatStructuredText(s.application_process)}
      </div>
      <div class="divider"></div>
      <div class="scheme-field" style="grid-column: 1 / -1; display: flex; flex-direction: row; justify-content: space-between; align-items: center;">
        <button class="speak-btn" id="${speakId}" data-speaking="0" style="margin-top:0" 
                data-tts="${escapeHtml(fullText.replace(/\n/g, ' '))}">🔊 Listen</button>
        <div class="scheme-field">
          <span class="field-label">Official Link</span>
          ${link}
        </div>
      </div>
    </div>
  </div>`;
}

async function sendMessage() {
  const q = input.value.trim();
  if (!q) return;

  addUserMessage(q);
  input.value = '';
  input.style.height = 'auto';
  sendBtn.disabled = true;
  const typing = addTyping();

  try {
    const res = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, lang: currentLang })
    });
    
    if (!res.ok) {
      removeTyping();
      renderResult({ error: 'Network error. Please try again.' });
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = '';
    
    let streamingRow = null;
    let streamingBubble = null;
    let fullText = '';
    let schemesWrapper = null;
    let serverLang = currentLang;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let lines = buffer.split('\n');
      buffer = lines.pop();
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.substring(6));

            if (data.type === 'conversational_start') {
              removeTyping();
              if (data.lang) serverLang = data.lang;
              streamingRow = document.createElement('div');
              streamingRow.className = 'msg-row ai';
              streamingRow.innerHTML = `<div class="avatar ai">🤖</div><div class="bubble ai" id="streaming-bubble"></div>`;
              chat.appendChild(streamingRow);
              streamingBubble = streamingRow.querySelector('#streaming-bubble');
              fullText = '';
            } else if (data.type === 'chunk') {
              fullText += data.text;
              if (streamingBubble) { streamingBubble.innerHTML = escapeHtml(fullText); scrollBottom(); }
            } else if (data.type === 'conversational_end') {
              if (streamingRow) streamingRow.remove();
              const result = { type: 'conversational', reply: fullText, lang: serverLang };
              renderResult(result);
              saveChat(q, result);
            } else if (data.type === 'schemes_start') {
              removeTyping();
              const row = document.createElement('div');
              row.className = 'msg-row ai';
              row.innerHTML = `<div class="avatar ai">🤖</div><div class="schemes-wrapper" id="streaming-schemes"></div>`;
              chat.appendChild(row);
              schemesWrapper = row.querySelector('#streaming-schemes');
            } else if (data.type === 'scheme_card') {
              if (schemesWrapper) {
                const cardHtml = buildSchemeCard(data.scheme, data.index);
                const temp = document.createElement('div');
                temp.innerHTML = cardHtml;
                const card = temp.firstElementChild;
                
                // Manual binding for Listen button
                const btn = card.querySelector('.speak-btn');
                if (btn) {
                  btn.onclick = () => speakText(btn.dataset.tts, btn);
                }

                card.style.opacity = '0'; card.style.transform = 'translateY(12px)';
                card.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
                schemesWrapper.appendChild(card);
                requestAnimationFrame(() => { card.style.opacity = '1'; card.style.transform = 'translateY(0)'; });
                scrollBottom();
              }
            } else if (data.type === 'schemes_end') {
               // Full detail schemes are handled by individual cards usually, 
               // but for saving we might want the full list.
               // RAG agent usually sends individual cards or full detail if asked.
            } else if (data.type === 'convert_to_cards') {
              if (streamingRow) streamingRow.remove();
              data.type = 'full_detail'; data.lang = serverLang;
              renderResult(data);
              saveChat(q, data);
            } else {
              removeTyping();
              if (!data.lang) data.lang = serverLang;
              renderResult(data);
              saveChat(q, data);
            }
          } catch(e) {
             console.error("Error processing stream line:", e, line);
          }
        }
      }
    }
  } catch (err) {
    removeTyping();
    renderResult({ error: 'Network error. Please try again.' });
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

async function clearChat() {
  await fetch('/reset', { method: 'POST' });
  currentChatId = null;
  currentChatMessages = [];
  chat.innerHTML = `<div class="welcome-card">
    <span class="wheel">☸️</span>
    <h2>New conversation started!</h2>
    <p>Ask me about any Gujarat government scheme.</p>
  </div>`;
  loadHistory(); // Re-toggle active state in sidebar
}

// ── Language Switcher ────────────────────────────────────────────────────────
const LANG_UI = {
  en: {
    placeholder: "Ask about any government scheme…",
    hint: "Press Enter to send · Shift+Enter for new line",
    chips: [
      "Schemes for farmers 🌾", "Women welfare schemes", "Education scholarships",
      "Healthcare schemes", "housing scheme",
      "Startup schemes for youth", "Schemes in Gujarat", "Skill development programs"
    ]
  },
  hi: {
    placeholder: "सरकारी योजना के बारे में पूछें…",
    hint: "Enter दबाएं भेजने के लिए · Shift+Enter नई लाइन के लिए",
    chips: [
      "किसानों के लिए योजनाएं 🌾", "महिला कल्याण योजनाएं", "शिक्षा छात्रवृत्ति",
      "स्वास्थ्य योजनाएं", "आवास योजना",
      "युवाओं के लिए स्टार्टअप योजनाएं", "गुजरात में योजनाएं", "कૌશલ विकास कार्यक्रम"
    ]
  },
  gu: {
    placeholder: "સરકારી યોજના વિશે પૂછો…",
    hint: "Enter દબાવો મોકલવા માટે · Shift+Enter નવી લીટી માટે",
    chips: [
      "ખેડૂતો માટે યોજનાઓ 🌾", "મહિલા કલ્યાણ યોજનાઓ", "શિક્ષણ શિષ્યવૃત્તિ",
      "આરોગ્ય સેવા યોજનાઓ", "આવાસ યોજના",
      "યુવાનો માટે સ્ટાર્ટઅપ યોજનાઓ", "ગુજરાતમાં યોજનાઓ", "કૌશલ્ય વિકાસ કાર્યક્રમ"
    ]
  }
};

let currentLang = 'en';  // tracks which language button is active

function setLang(lang) {
  currentLang = lang;
  if (isListening && recognition) recognition.stop();
  // Update active button
  document.querySelectorAll('.lang-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.lang === lang)
  );
  const L = LANG_UI[lang];
  // Update input placeholder
  document.getElementById('question-input').placeholder = L.placeholder;
  // Update hint text
  document.getElementById('input-hint').textContent = L.hint;
  // Update suggestion chips
  const bar = document.getElementById('suggestions');
  bar.innerHTML = L.chips
    .map(c => `<div class="chip" onclick="sendChip(this)">${c}</div>`)
    .join('');
}

// ── Web Notifications ────────────────────────────────────────────────────────
function toggleNotifDropdown() {
  const dropdown = document.getElementById('notif-dropdown');
  dropdown.classList.toggle('active');
  if (dropdown.classList.contains('active')) {
    loadNotifications();
  }
}

async function loadNotifications() {
  if (!currentUser) return;
  const res = await fetch('/get_notifications');
  const data = await res.json();
  const list = document.getElementById('notif-list');
  const badge = document.getElementById('notif-badge');

  if (data.unread_count > 0) {
    badge.classList.add('active');
  } else {
    badge.classList.remove('active');
  }

  if (data.notifications && data.notifications.length > 0) {
    list.innerHTML = data.notifications.map(n => `
      <div class="notif-item">
        <div class="notif-content">
          <div class="notif-title">${escapeHtml(n.title)}</div>
          <div class="notif-text">${escapeHtml(n.message)}</div>
        </div>
        <span class="notif-delete" onclick="deleteNotification(event, ${n.id})" title="Delete notification">✕</span>
      </div>
    `).join('');
  } else {
    list.innerHTML = '<div class="notif-empty">No current notifications</div>';
  }
}

async function deleteNotification(e, id) {
  if (e) e.stopPropagation();
  await fetch('/delete_notification', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id: id })
  });
  loadNotifications();
}

async function markAllRead(e) {
  if (e) e.stopPropagation();
  await fetch('/mark_read', { method: 'POST' });
  const badge = document.getElementById('notif-badge');
  badge.classList.remove('active');
  loadNotifications();
}

// Close notifications when clicking outside
window.addEventListener('click', (e) => {
  const btn = document.getElementById('notif-btn');
  const dropdown = document.getElementById('notif-dropdown');
  if (btn && !btn.contains(e.target) && dropdown) {
    dropdown.classList.remove('active');
  }
});

// Update window.onload to also fetch notifications
const originalOnload = window.onload;
window.onload = async () => {
    if (originalOnload) await originalOnload();
    if (currentUser) {
        loadNotifications();
        // Check for new notifications every 5 minutes
        setInterval(loadNotifications, 5 * 60 * 1000);
    }
};
