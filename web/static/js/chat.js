/**
 * Eurus Chat WebSocket Client
 */

class EurusChat {
    constructor() {
        this.ws = null;
        this.messageId = 0;
        this.currentAssistantMessage = null;
        this.isConnected = false;
        this.keysConfigured = false;
        this.serverKeysPresent = { openai: false, arraylake: false };
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;

        this.messagesContainer = document.getElementById('chat-messages');
        this.messageInput = document.getElementById('message-input');
        this.chatForm = document.getElementById('chat-form');
        this.sendBtn = document.getElementById('send-btn');
        this.connectionStatus = document.getElementById('connection-status');
        this.clearBtn = document.getElementById('clear-btn');
        this.cacheBtn = document.getElementById('cache-btn');
        this.cacheModal = document.getElementById('cache-modal');
        this.apiKeysPanel = document.getElementById('api-keys-panel');
        this.saveKeysBtn = document.getElementById('save-keys-btn');
        this.openaiKeyInput = document.getElementById('openai-key');
        this.arraylakeKeyInput = document.getElementById('arraylake-key');

        marked.setOptions({
            highlight: (code, lang) => {
                if (lang && hljs.getLanguage(lang)) {
                    return hljs.highlight(code, { language: lang }).value;
                }
                return hljs.highlightAuto(code).value;
            },
            breaks: true,
            gfm: true
        });

        this.themeToggle = document.getElementById('theme-toggle');
        this.init();
    }

    async init() {
        await this.checkKeysStatus();
        this.connect();
        this.setupEventListeners();
        this.setupImageModal();
        this.setupTheme();
        this.setupKeysPanel();
    }

    async checkKeysStatus() {
        try {
            const resp = await fetch('/api/keys-status');
            const data = await resp.json();
            this.serverKeysPresent = data;

            if (data.openai) {
                // Keys pre-configured on server — hide the panel
                this.apiKeysPanel.style.display = 'none';
                this.keysConfigured = true;
                // Enable send if WS is already connected
                if (this.isConnected) {
                    this.sendBtn.disabled = false;
                }
            } else {
                // No server keys — show panel, user must enter keys each session
                this.apiKeysPanel.style.display = 'block';
                this.keysConfigured = false;
            }
        } catch (e) {
            // Can't reach server yet, show panel
            this.apiKeysPanel.style.display = 'block';
        }
    }

    setupKeysPanel() {
        this.saveKeysBtn.addEventListener('click', () => this.saveAndSendKeys());

        // Allow Enter in key fields to submit
        [this.openaiKeyInput, this.arraylakeKeyInput].forEach(input => {
            input.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    this.saveAndSendKeys();
                }
            });
        });

        // Restore keys from sessionStorage (survives refresh, cleared on browser close)
        this.restoreSessionKeys();
    }

    restoreSessionKeys() {
        const saved = sessionStorage.getItem('eurus-keys');
        if (!saved) return;
        try {
            const keys = JSON.parse(saved);
            if (keys.openai_api_key) this.openaiKeyInput.value = keys.openai_api_key;
            if (keys.arraylake_api_key) this.arraylakeKeyInput.value = keys.arraylake_api_key;
        } catch (e) {
            sessionStorage.removeItem('eurus-keys');
        }
    }

    autoSendSessionKeys() {
        // After WS (re)connects, resend saved keys so the new server-side session gets them.
        // Only skip if the server has pre-configured env keys (not user-provided).
        if (this.serverKeysPresent.openai) return;
        const saved = sessionStorage.getItem('eurus-keys');
        if (!saved) return;
        try {
            const keys = JSON.parse(saved);
            if (keys.openai_api_key && this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({
                    type: 'configure_keys',
                    openai_api_key: keys.openai_api_key,
                    arraylake_api_key: keys.arraylake_api_key || '',
                }));
            }
        } catch (e) {
            sessionStorage.removeItem('eurus-keys');
        }
    }

    saveAndSendKeys() {
        const openaiKey = this.openaiKeyInput.value.trim();
        const arraylakeKey = this.arraylakeKeyInput.value.trim();

        if (!openaiKey) {
            this.openaiKeyInput.focus();
            return;
        }

        // Save to sessionStorage (cleared when browser closes, survives refresh)
        const keysPayload = {
            openai_api_key: openaiKey,
            arraylake_api_key: arraylakeKey,
        };
        sessionStorage.setItem('eurus-keys', JSON.stringify(keysPayload));

        // Send keys via WebSocket
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.saveKeysBtn.disabled = true;
            this.saveKeysBtn.textContent = 'Connecting...';
            this.ws.send(JSON.stringify({
                type: 'configure_keys',
                ...keysPayload,
            }));
        }
    }

    setupTheme() {
        // Load saved theme or default to dark (neosynth)
        const savedTheme = localStorage.getItem('eurus-theme') || 'dark';
        document.documentElement.setAttribute('data-theme', savedTheme);
        this.updateThemeIcon(savedTheme);

        // Theme toggle click handler
        if (this.themeToggle) {
            this.themeToggle.addEventListener('click', () => {
                const currentTheme = document.documentElement.getAttribute('data-theme');
                const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
                document.documentElement.setAttribute('data-theme', newTheme);
                localStorage.setItem('eurus-theme', newTheme);
                this.updateThemeIcon(newTheme);
            });
        }
    }

    updateThemeIcon(theme) {
        if (this.themeToggle) {
            const icon = this.themeToggle.querySelector('.theme-icon');
            if (icon) {
                icon.textContent = theme === 'dark' ? '☀️' : '🌙';
            }
        }
    }

    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/chat`;

        this.updateConnectionStatus('connecting');

        try {
            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => {
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.updateConnectionStatus('connected');

                // Always resend keys on (re)connect — server session may have been
                // destroyed on disconnect, so keys stored in sessionStorage must be
                // re-sent even if keysConfigured is true from the previous session.
                this.autoSendSessionKeys();

                if (this.serverKeysPresent.openai || this.keysConfigured) {
                    this.sendBtn.disabled = false;
                }
            };

            this.ws.onclose = () => {
                this.isConnected = false;
                this.updateConnectionStatus('disconnected');
                this.sendBtn.disabled = true;
                this.attemptReconnect();
            };

            this.ws.onerror = () => {
                this.updateConnectionStatus('disconnected');
            };

            this.ws.onmessage = (event) => {
                this.handleMessage(JSON.parse(event.data));
            };

        } catch (error) {
            this.updateConnectionStatus('disconnected');
        }
    }

    attemptReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) return;

        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

        this.updateConnectionStatus('connecting');
        setTimeout(() => this.connect(), delay);
    }

    updateConnectionStatus(status) {
        this.connectionStatus.className = 'status-badge ' + status;
        const text = { connected: 'Connected', disconnected: 'Disconnected', connecting: 'Connecting...' };
        this.connectionStatus.textContent = text[status] || status;
    }

    setupEventListeners() {
        this.chatForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.sendMessage();
        });

        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        this.messageInput.addEventListener('input', () => {
            this.messageInput.style.height = 'auto';
            this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 150) + 'px';
        });

        this.clearBtn.addEventListener('click', (e) => {
            e.preventDefault();
            this.clearChat();
        });

        this.cacheBtn.addEventListener('click', (e) => {
            e.preventDefault();
            this.showCacheModal();
        });

        this.cacheModal.querySelector('.close-modal').addEventListener('click', () => {
            this.cacheModal.close();
        });
    }

    setupImageModal() {
        // Create modal for enlarged images
        const modal = document.createElement('div');
        modal.id = 'image-modal';
        modal.innerHTML = `
            <div class="image-modal-backdrop"></div>
            <div class="image-modal-content">
                <img alt="Enlarged plot">
                <div class="image-modal-actions">
                    <button class="download-btn">Download</button>
                    <button class="close-btn">Close</button>
                </div>
            </div>
        `;
        document.body.appendChild(modal);

        // Add modal styles
        const style = document.createElement('style');
        style.textContent = `
            #image-modal {
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                z-index: 1000;
            }
            #image-modal.active {
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .image-modal-backdrop {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0,0,0,0.8);
            }
            .image-modal-content {
                position: relative;
                max-width: 90%;
                max-height: 90%;
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .image-modal-content img {
                max-width: 100%;
                max-height: calc(90vh - 60px);
                border-radius: 4px;
            }
            .image-modal-actions {
                margin-top: 12px;
                display: flex;
                gap: 8px;
            }
            .image-modal-actions button {
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
            }
            .image-modal-actions .download-btn {
                background: #1976d2;
                color: white;
            }
            .image-modal-actions .close-btn {
                background: #757575;
                color: white;
            }
        `;
        document.head.appendChild(style);

        // Event listeners
        modal.querySelector('.image-modal-backdrop').addEventListener('click', () => {
            modal.classList.remove('active');
        });

        modal.querySelector('.close-btn').addEventListener('click', () => {
            modal.classList.remove('active');
        });

        modal.querySelector('.download-btn').addEventListener('click', () => {
            const img = modal.querySelector('img');
            const link = document.createElement('a');
            link.href = img.src;
            link.download = 'eurus_plot.png';
            link.click();
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && modal.classList.contains('active')) {
                modal.classList.remove('active');
            }
        });

        this.imageModal = modal;
    }

    showImageModal(src) {
        this.imageModal.querySelector('img').src = src;
        this.imageModal.classList.add('active');
    }

    sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || !this.isConnected) return;

        this.addUserMessage(message);
        this.ws.send(JSON.stringify({ message }));

        this.messageInput.value = '';
        this.messageInput.style.height = 'auto';
        this.sendBtn.disabled = true;
    }

    handleMessage(data) {
        switch (data.type) {
            case 'keys_configured':
                this.keysConfigured = data.ready;
                if (data.ready) {
                    this.apiKeysPanel.style.display = 'none';
                    this.sendBtn.disabled = false;
                } else {
                    this.saveKeysBtn.disabled = false;
                    this.saveKeysBtn.textContent = 'Connect';
                    this.showError('Failed to initialize agent. Check your API keys.');
                }
                break;

            case 'thinking':
                this.showThinkingIndicator();
                break;

            case 'status':
                this.updateStatusIndicator(data.content);
                break;


            case 'chunk':
                this.appendToAssistantMessage(data.content);
                break;

            case 'plot':
                this.addPlot(data.data, data.path, data.code || '');
                break;

            case 'video':
                console.log('[WS] Video message received:', data);
                this.addVideo(data.data, data.path, data.mimetype || 'video/mp4');
                break;

            case 'complete':
                this.finalizeAssistantMessage(data.content);
                this.sendBtn.disabled = false;
                break;

            case 'arraylake_snippet':
                this.addArraylakeSnippet(data.content);
                break;

            case 'error':
                this.showError(data.content);
                this.sendBtn.disabled = false;
                break;

            case 'clear':
                this.clearMessagesUI();
                break;
        }
    }

    addUserMessage(content) {
        const div = document.createElement('div');
        div.className = 'message user-message';
        div.innerHTML = `
            <div class="message-header">
                <span class="message-role">You</span>
            </div>
            <div class="message-content">${this.escapeHtml(content)}</div>
        `;
        this.messagesContainer.appendChild(div);
        this.scrollToBottom();
    }

    showThinkingIndicator() {
        this.removeThinkingIndicator();

        const div = document.createElement('div');
        div.className = 'message thinking-message';
        div.id = 'thinking-indicator';
        div.innerHTML = `
            <div class="typing-indicator">
                <span></span><span></span><span></span>
            </div>
        `;
        this.messagesContainer.appendChild(div);
        this.scrollToBottom();
    }

    removeThinkingIndicator() {
        const indicator = document.getElementById('thinking-indicator');
        if (indicator) indicator.remove();
    }

    updateStatusIndicator(statusText) {
        // Replace thinking dots with status message
        let indicator = document.getElementById('thinking-indicator');

        if (!indicator) {
            indicator = document.createElement('div');
            indicator.className = 'message thinking-message';
            indicator.id = 'thinking-indicator';
            this.messagesContainer.appendChild(indicator);
        }

        indicator.innerHTML = `
            <div class="status-indicator">
                <span class="status-spinner"></span>
                <span class="status-text">${this.escapeHtml(statusText)}</span>
            </div>
        `;
        this.scrollToBottom();
    }

    appendToAssistantMessage(content) {
        this.removeThinkingIndicator();

        if (!this.currentAssistantMessage) {
            this.currentAssistantMessage = document.createElement('div');
            this.currentAssistantMessage.className = 'message assistant-message';
            this.currentAssistantMessage.innerHTML = `
                <div class="message-header">
                    <img src="/static/favicon.jpeg" class="avatar-icon" alt="">
                    <span class="message-role">Eurus</span>
                </div>
                <div class="message-content markdown-content"></div>
                <div class="message-plots"></div>
            `;
            this.messagesContainer.appendChild(this.currentAssistantMessage);
        }

        const contentDiv = this.currentAssistantMessage.querySelector('.message-content');
        const raw = (contentDiv.getAttribute('data-raw') || '') + content;
        contentDiv.setAttribute('data-raw', raw);
        contentDiv.innerHTML = marked.parse(raw);

        contentDiv.querySelectorAll('pre code').forEach(block => hljs.highlightElement(block));
        this.scrollToBottom();
    }

    addPlot(base64Data, path, code = '') {
        this.removeThinkingIndicator();

        if (!this.currentAssistantMessage) {
            this.appendToAssistantMessage('');
        }

        const plotsDiv = this.currentAssistantMessage.querySelector('.message-plots');

        const figure = document.createElement('figure');
        figure.className = 'plot-figure';

        const imgSrc = `data:image/png;base64,${base64Data}`;
        const codeId = `code-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

        figure.innerHTML = `
            <img src="${imgSrc}" alt="Generated plot">
            <div class="plot-actions">
                <button class="enlarge-btn" title="Enlarge">Enlarge</button>
                <button class="download-btn" title="Download">Download</button>
                ${code && code.trim() ? `<button class="code-btn" title="Show Code">Show Code</button>` : ''}
            </div>
        `;

        // Add code block separately if code exists
        if (code && code.trim()) {
            const codeDiv = document.createElement('div');
            codeDiv.className = 'plot-code';
            codeDiv.style.display = 'none';

            const pre = document.createElement('pre');
            const codeEl = document.createElement('code');
            codeEl.className = 'language-python hljs';

            // Highlight immediately
            try {
                const highlighted = hljs.highlight(code, { language: 'python' });
                codeEl.innerHTML = highlighted.value;
            } catch (e) {
                console.error('Highlight error:', e);
                codeEl.textContent = code;
            }

            pre.appendChild(codeEl);
            codeDiv.appendChild(pre);
            figure.appendChild(codeDiv);
        }

        // Add enlarge action
        figure.querySelector('.enlarge-btn').addEventListener('click', () => {
            this.showImageModal(imgSrc);
        });

        // Add download action
        figure.querySelector('.download-btn').addEventListener('click', () => {
            const link = document.createElement('a');
            link.href = imgSrc;
            const filename = path ? path.split('/').pop() : 'eurus_plot.png';
            link.download = filename;
            link.click();
        });

        // Add show code toggle
        const codeBtn = figure.querySelector('.code-btn');
        if (codeBtn) {
            const codeDiv = figure.querySelector('.plot-code');

            codeBtn.addEventListener('click', () => {
                if (codeDiv.style.display === 'none') {
                    codeDiv.style.display = 'block';
                    codeBtn.textContent = 'Hide Code';
                } else {
                    codeDiv.style.display = 'none';
                    codeBtn.textContent = 'Show Code';
                }
            });
        }

        // Click on image to enlarge
        figure.querySelector('img').addEventListener('click', () => {
            this.showImageModal(imgSrc);
        });

        plotsDiv.appendChild(figure);
        this.scrollToBottom();
    }

    addVideo(base64Data, path, mimetype = 'video/mp4') {
        console.log('[VIDEO] addVideo called:', { path, mimetype, dataLength: base64Data?.length });
        this.removeThinkingIndicator();

        if (!this.currentAssistantMessage) {
            this.appendToAssistantMessage('');
        }

        const plotsDiv = this.currentAssistantMessage.querySelector('.message-plots');
        console.log('[VIDEO] plotsDiv found:', plotsDiv);

        const figure = document.createElement('figure');
        figure.className = 'plot-figure video-figure';

        // Handle different formats
        let videoSrc;
        if (mimetype === 'image/gif') {
            // GIFs display as img
            videoSrc = `data:image/gif;base64,${base64Data}`;
            figure.innerHTML = `
                <img src="${videoSrc}" alt="Generated animation" class="video-gif" style="max-width: 100%; border-radius: 8px;">
                <div class="plot-actions">
                    <button class="enlarge-btn" title="Enlarge">Enlarge</button>
                    <button class="download-btn" title="Download">Download</button>
                </div>
            `;

            // Enlarge for GIF
            figure.querySelector('.enlarge-btn').addEventListener('click', () => {
                this.showImageModal(videoSrc);
            });
            figure.querySelector('img').addEventListener('click', () => {
                this.showImageModal(videoSrc);
            });
        } else {
            // Video formats (webm, mp4)
            videoSrc = `data:${mimetype};base64,${base64Data}`;
            figure.innerHTML = `
                <video controls autoplay loop muted playsinline style="max-width: 100%; border-radius: 8px;">
                    <source src="${videoSrc}" type="${mimetype}">
                    Your browser does not support video playback.
                </video>
                <div class="plot-actions">
                    <button class="download-btn" title="Download">Download</button>
                </div>
            `;
        }

        // Download button
        figure.querySelector('.download-btn').addEventListener('click', () => {
            const link = document.createElement('a');
            link.href = videoSrc;
            const ext = mimetype.includes('gif') ? 'gif' : mimetype.includes('webm') ? 'webm' : 'mp4';
            const filename = path ? path.split('/').pop() : `eurus_animation.${ext}`;
            link.download = filename;
            link.click();
        });

        plotsDiv.appendChild(figure);
        this.scrollToBottom();
    }

    finalizeAssistantMessage(content) {
        this.removeThinkingIndicator();
        if (content && !this.currentAssistantMessage) {
            this.appendToAssistantMessage(content);
        }
        this.currentAssistantMessage = null;
    }

    addArraylakeSnippet(snippetText) {
        // Find the latest assistant message
        const messages = this.messagesContainer.querySelectorAll('.assistant-message');
        const targetMsg = messages.length > 0 ? messages[messages.length - 1] : null;
        if (!targetMsg) return;

        // Strip markdown fences for raw code display
        let cleanCode = snippetText
            .replace(/^\n?📦[^\n]*\n/, '')
            .replace(/^```python\n?/, '')
            .replace(/\n?```$/, '')
            .trim();

        // Find the last plot figure's action bar to add button inline
        const figures = targetMsg.querySelectorAll('.plot-figure');
        const lastFigure = figures.length > 0 ? figures[figures.length - 1] : null;
        const actionsDiv = lastFigure ? lastFigure.querySelector('.plot-actions') : null;

        if (actionsDiv) {
            // Check if an Arraylake button already exists on this figure
            const existingBtn = actionsDiv.querySelector('.arraylake-btn');
            if (existingBtn) {
                // Append new snippet to the existing code block
                const existingCodeDiv = lastFigure.querySelector('.arraylake-code-block');
                if (existingCodeDiv) {
                    const codeEl = existingCodeDiv.querySelector('code');
                    const prevRaw = existingCodeDiv.getAttribute('data-raw-code') || '';
                    const combined = prevRaw + '\n\n# ---\n\n' + cleanCode;
                    existingCodeDiv.setAttribute('data-raw-code', combined);
                    try {
                        codeEl.innerHTML = hljs.highlight(combined, { language: 'python' }).value;
                    } catch (e) {
                        codeEl.textContent = combined;
                    }
                    // Update copy button target
                    const copyBtn = existingCodeDiv.querySelector('.copy-snippet-btn');
                    if (copyBtn) {
                        copyBtn.onclick = () => {
                            navigator.clipboard.writeText(combined).then(() => {
                                copyBtn.textContent = '✓ Copied!';
                                setTimeout(() => copyBtn.textContent = 'Copy Code', 2000);
                            });
                        };
                    }
                }
                return;
            }

            // Add button inline with Enlarge/Download/Show Code
            const btn = document.createElement('button');
            btn.className = 'code-btn arraylake-btn';
            btn.title = 'Arraylake Code';
            btn.textContent = '📦 Arraylake Code';
            actionsDiv.appendChild(btn);

            // Add code block to figure (same pattern as Show Code)
            const codeDiv = document.createElement('div');
            codeDiv.className = 'plot-code arraylake-code-block';
            codeDiv.style.display = 'none';
            codeDiv.setAttribute('data-raw-code', cleanCode);

            const pre = document.createElement('pre');
            const codeEl = document.createElement('code');
            codeEl.className = 'language-python hljs';
            try {
                codeEl.innerHTML = hljs.highlight(cleanCode, { language: 'python' }).value;
            } catch (e) {
                codeEl.textContent = cleanCode;
            }
            pre.appendChild(codeEl);
            codeDiv.appendChild(pre);

            // Copy button inside code block
            const copyBtn = document.createElement('button');
            copyBtn.className = 'copy-snippet-btn';
            copyBtn.textContent = 'Copy Code';
            copyBtn.addEventListener('click', () => {
                const rawCode = codeDiv.getAttribute('data-raw-code') || cleanCode;
                navigator.clipboard.writeText(rawCode).then(() => {
                    copyBtn.textContent = '✓ Copied!';
                    setTimeout(() => copyBtn.textContent = 'Copy Code', 2000);
                });
            });
            codeDiv.appendChild(copyBtn);
            lastFigure.appendChild(codeDiv);

            // Toggle
            btn.addEventListener('click', () => {
                if (codeDiv.style.display === 'none') {
                    codeDiv.style.display = 'block';
                    btn.textContent = '📦 Hide Arraylake';
                } else {
                    codeDiv.style.display = 'none';
                    btn.textContent = '📦 Arraylake Code';
                }
            });
        } else {
            // No plot figure — check if standalone section already exists
            const existingWrapper = targetMsg.querySelector('.arraylake-snippet-section');
            if (existingWrapper) {
                // Append to existing standalone snippet
                const codeEl = existingWrapper.querySelector('code');
                const existingCodeDiv = existingWrapper.querySelector('.plot-code');
                const prevRaw = existingCodeDiv.getAttribute('data-raw-code') || '';
                const combined = prevRaw + '\n\n# ---\n\n' + cleanCode;
                existingCodeDiv.setAttribute('data-raw-code', combined);
                try {
                    codeEl.innerHTML = hljs.highlight(combined, { language: 'python' }).value;
                } catch (e) {
                    codeEl.textContent = combined;
                }
                return;
            }

            const wrapper = document.createElement('div');
            wrapper.className = 'arraylake-snippet-section';
            wrapper.innerHTML = `
                <div class="plot-actions">
                    <button class="code-btn arraylake-btn" title="Arraylake Code">📦 Arraylake Code</button>
                </div>
                <div class="plot-code" style="display: none;">
                    <pre><code class="language-python hljs"></code></pre>
                </div>
            `;
            const codeDivEl = wrapper.querySelector('.plot-code');
            codeDivEl.setAttribute('data-raw-code', cleanCode);
            const codeEl = wrapper.querySelector('code');
            try {
                codeEl.innerHTML = hljs.highlight(cleanCode, { language: 'python' }).value;
            } catch (e) {
                codeEl.textContent = cleanCode;
            }
            const btn = wrapper.querySelector('.code-btn');
            const codeDiv = wrapper.querySelector('.plot-code');
            btn.addEventListener('click', () => {
                if (codeDiv.style.display === 'none') {
                    codeDiv.style.display = 'block';
                    btn.textContent = '📦 Hide Arraylake';
                } else {
                    codeDiv.style.display = 'none';
                    btn.textContent = '📦 Arraylake Code';
                }
            });
            targetMsg.appendChild(wrapper);
        }
        this.scrollToBottom();
    }

    showError(message) {
        this.removeThinkingIndicator();

        const div = document.createElement('div');
        div.className = 'message error-message';
        div.innerHTML = `
            <div class="message-header">
                <span class="message-role">Error</span>
            </div>
            <div class="message-content">${this.escapeHtml(message)}</div>
        `;
        this.messagesContainer.appendChild(div);
        this.currentAssistantMessage = null;
        this.scrollToBottom();
    }

    async clearChat() {
        if (!confirm('Clear conversation?')) return;

        // Send clear command through WebSocket so the agent session memory is also cleared
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ message: '/clear' }));
        } else {
            // Fallback to REST if WS not available
            try {
                const response = await fetch('/api/conversation', { method: 'DELETE' });
                if (response.ok) this.clearMessagesUI();
            } catch (error) {
                console.error('Error clearing:', error);
            }
        }
    }

    clearMessagesUI() {
        const messages = this.messagesContainer.querySelectorAll('.message:not(.system-message)');
        messages.forEach(msg => msg.remove());
        this.currentAssistantMessage = null;
    }

    async showCacheModal() {
        this.cacheModal.showModal();
        const content = document.getElementById('cache-content');
        content.innerHTML = '<p>Loading...</p>';

        try {
            const response = await fetch('/api/cache');
            const data = await response.json();

            if (data.datasets && data.datasets.length > 0) {
                const formatSize = (bytes) => {
                    if (bytes < 1024) return bytes + ' B';
                    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
                    return (bytes / 1048576).toFixed(1) + ' MB';
                };

                let html = '<table><thead><tr><th>Variable</th><th>Period</th><th>Type</th><th>Size</th><th></th></tr></thead><tbody>';
                for (const ds of data.datasets) {
                    html += `<tr>
                        <td>${ds.variable}</td>
                        <td>${ds.start_date} to ${ds.end_date}</td>
                        <td>${ds.query_type}</td>
                        <td>${formatSize(ds.file_size_bytes)}</td>
                        <td><button class="cache-download-btn" data-path="${ds.path}" title="Download as ZIP">⬇</button></td>
                    </tr>`;
                }
                html += '</tbody></table>';
                html += `<p class="cache-total">Total: ${formatSize(data.total_size_bytes)} across ${data.datasets.length} dataset(s)</p>`;
                content.innerHTML = html;

                // Attach download handlers
                content.querySelectorAll('.cache-download-btn').forEach(btn => {
                    btn.addEventListener('click', async (e) => {
                        const path = e.target.dataset.path;
                        const origText = e.target.textContent;
                        e.target.textContent = '⏳';
                        e.target.disabled = true;
                        try {
                            const resp = await fetch(`/api/cache/download?path=${encodeURIComponent(path)}`);
                            if (!resp.ok) throw new Error('Download failed');
                            const blob = await resp.blob();
                            const url = URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = path.split('/').pop() + '.zip';
                            document.body.appendChild(a);
                            a.click();
                            a.remove();
                            URL.revokeObjectURL(url);
                            e.target.textContent = '✅';
                            setTimeout(() => { e.target.textContent = origText; e.target.disabled = false; }, 2000);
                        } catch (err) {
                            e.target.textContent = '❌';
                            setTimeout(() => { e.target.textContent = origText; e.target.disabled = false; }, 2000);
                        }
                    });
                });
            } else {
                content.innerHTML = '<p>No cached datasets.</p>';
            }
        } catch (error) {
            content.innerHTML = `<p>Error: ${error.message}</p>`;
        }
    }

    scrollToBottom() {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.eurusChat = new EurusChat();
});
