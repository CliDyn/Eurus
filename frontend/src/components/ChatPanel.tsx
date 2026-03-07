import { useCallback, useEffect, useRef, useState, ReactNode } from 'react';
import { Send, Wifi, WifiOff, Loader2, Trash2 } from 'lucide-react';
import { useWebSocket, WSEvent } from '../hooks/useWebSocket';
import MessageBubble, { ChatMessage, MediaItem } from './MessageBubble';
import ApiKeysPanel from './ApiKeysPanel';
import './ChatPanel.css';

interface ChatPanelProps {
    cacheToggle?: ReactNode;
}

let msgCounter = 0;
const uid = () => `msg-${++msgCounter}-${Date.now()}`;

export default function ChatPanel({ cacheToggle }: ChatPanelProps) {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [input, setInput] = useState('');
    const [isThinking, setIsThinking] = useState(false);
    const [statusMsg, setStatusMsg] = useState('');
    const [needKeys, setNeedKeys] = useState<boolean | null>(null); // null = don't know yet
    const bottomRef = useRef<HTMLDivElement>(null);
    const streamBuf = useRef('');
    const streamMedia = useRef<MediaItem[]>([]);
    const streamSnippets = useRef<string[]>([]);
    const streamId = useRef<string | null>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);


    /* ── event handler ── */
    const handleEvent = useCallback((ev: WSEvent) => {
        switch (ev.type) {
            case 'thinking':
                setIsThinking(true);
                setStatusMsg('');
                streamBuf.current = '';
                streamMedia.current = [];
                streamSnippets.current = [];
                streamId.current = uid();
                break;

            case 'status':
                setStatusMsg(ev.content ?? '');
                break;

            case 'tool_start':
                setMessages(prev => {
                    const id = streamId.current ?? uid();
                    streamId.current = id;
                    const exists = prev.find(m => m.id === id);
                    if (exists) {
                        return prev.map(m =>
                            m.id === id ? { ...m, toolLabel: ev.content ?? '' } : m
                        );
                    }
                    return [...prev, { id, role: 'assistant', content: '', toolLabel: ev.content ?? '', isStreaming: true }];
                });
                break;

            case 'stream': {
                setIsThinking(false);
                setStatusMsg('');
                const chunk = ev.content ?? '';
                streamBuf.current += chunk;
                const id = streamId.current ?? uid();
                streamId.current = id;
                setMessages(prev => {
                    const exists = prev.find(m => m.id === id);
                    if (exists) {
                        return prev.map(m =>
                            m.id === id ? { ...m, content: streamBuf.current, isStreaming: true } : m
                        );
                    }
                    return [...prev, { id, role: 'assistant', content: streamBuf.current, isStreaming: true }];
                });
                break;
            }

            case 'plot': {
                const id = streamId.current ?? uid();
                streamId.current = id;
                if (ev.data) {
                    streamMedia.current.push({
                        type: 'plot',
                        base64: ev.data as string,
                        path: ev.path as string | undefined,
                        code: ev.code as string | undefined,
                    });
                }
                setMessages(prev => {
                    const exists = prev.find(m => m.id === id);
                    if (exists) {
                        return prev.map(m =>
                            m.id === id ? { ...m, media: [...streamMedia.current] } : m
                        );
                    }
                    return [...prev, { id, role: 'assistant', content: streamBuf.current, media: [...streamMedia.current], isStreaming: true }];
                });
                break;
            }

            case 'video': {
                const id = streamId.current ?? uid();
                streamId.current = id;
                if (ev.data) {
                    streamMedia.current.push({
                        type: 'video',
                        base64: ev.data as string,
                        path: ev.path as string | undefined,
                        mimetype: ev.mimetype as string | undefined,
                    });
                }
                setMessages(prev => {
                    const exists = prev.find(m => m.id === id);
                    if (exists) {
                        return prev.map(m =>
                            m.id === id ? { ...m, media: [...streamMedia.current] } : m
                        );
                    }
                    return [...prev, { id, role: 'assistant', content: streamBuf.current, media: [...streamMedia.current], isStreaming: true }];
                });
                break;
            }

            case 'arraylake_snippet': {
                const id = streamId.current;
                if (ev.content && id) {
                    streamSnippets.current.push(ev.content);
                    setMessages(prev =>
                        prev.map(m =>
                            m.id === id ? { ...m, arraylakeSnippets: [...streamSnippets.current] } : m
                        )
                    );
                }
                break;
            }

            case 'complete':
                setIsThinking(false);
                setStatusMsg('');
                if (streamId.current) {
                    const finalContent = ev.content ?? streamBuf.current;
                    setMessages(prev => {
                        const exists = prev.find(m => m.id === streamId.current);
                        if (exists) {
                            return prev.map(m =>
                                m.id === streamId.current
                                    ? { ...m, content: finalContent, media: [...streamMedia.current], arraylakeSnippets: [...streamSnippets.current], isStreaming: false, toolLabel: undefined, statusText: undefined }
                                    : m
                            );
                        }
                        return [...prev, { id: streamId.current!, role: 'assistant', content: finalContent, media: [...streamMedia.current], arraylakeSnippets: [...streamSnippets.current] }];
                    });
                } else {
                    setMessages(prev => [...prev, { id: uid(), role: 'assistant', content: ev.content ?? '' }]);
                }
                streamBuf.current = '';
                streamMedia.current = [];
                streamSnippets.current = [];
                streamId.current = null;
                break;

            case 'error':
                setIsThinking(false);
                setStatusMsg('');
                setMessages(prev => [...prev, { id: uid(), role: 'system', content: `⚠ ${ev.content ?? 'Unknown error'}` }]);
                streamId.current = null;
                break;

            case 'keys_configured':
                if (ev.ready) {
                    setNeedKeys(false);
                }
                break;

            case 'request_keys':
                // Server lost keys — resend from sessionStorage
                setNeedKeys(true);
                break;

            case 'clear':
                setMessages([]);
                streamBuf.current = '';
                streamMedia.current = [];
                streamSnippets.current = [];
                streamId.current = null;
                break;

            default:
                break;
        }
    }, []);

    const { status, sendMessage, configureKeys } = useWebSocket(handleEvent);

    /* ── check if server has keys ── */
    useEffect(() => {
        if (status !== 'connected') return; // only check when connected
        fetch('/api/keys-status')
            .then(r => r.json())
            .then(data => {
                setNeedKeys(!data.openai);
            })
            .catch(() => setNeedKeys(false)); // fallback: assume keys in .env
    }, [status]);

    /* ── auto-scroll ── */
    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, isThinking, statusMsg]);

    /* ── send ── */
    const handleSend = () => {
        const text = input.trim();
        if (!text || status !== 'connected') return;
        setMessages(prev => [...prev, { id: uid(), role: 'user', content: text }]);
        sendMessage(text);
        setInput('');
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
        }
    };

    /* ── clear conversation ── */
    const handleClear = async () => {
        try {
            await fetch('/api/conversation', { method: 'DELETE' });
            setMessages([]);
        } catch { /* ignore */ }
    };

    /* ── auto-resize textarea ── */
    const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        setInput(e.target.value);
        const ta = e.target;
        ta.style.height = 'auto';
        ta.style.height = Math.min(ta.scrollHeight, 160) + 'px';
    };

    /* ── keys handler ── */
    const handleSaveKeys = (keys: { openai_api_key: string; arraylake_api_key: string }) => {
        configureKeys(keys);
    };

    const statusColor = status === 'connected' ? '#34d399' : status === 'connecting' ? '#fbbf24' : '#f87171';
    const StatusIcon = status === 'connected' ? Wifi : WifiOff;
    const statusClass = `status-badge ${status === 'disconnected' ? 'disconnected' : ''}`;

    const canSend = status === 'connected' && needKeys !== true;

    return (
        <div className="chat-panel">
            {/* header */}
            <header className="chat-header">
                <div className="chat-title">
                    <div className="chat-logo">🌊</div>
                    <h1>Eurus Climate Agent</h1>
                </div>
                <div className="chat-header-actions">
                    {cacheToggle}
                    <button className="icon-btn" onClick={handleClear} title="Clear conversation">
                        <Trash2 size={16} />
                    </button>
                    <div className={statusClass} style={{ color: statusColor }}>
                        <StatusIcon size={12} />
                        <span>{status}</span>
                    </div>
                </div>
            </header>

            {/* API keys panel */}
            <ApiKeysPanel visible={needKeys === true} onSave={handleSaveKeys} />

            {/* messages */}
            <div className="messages-container">
                {messages.length === 0 && (
                    <div className="empty-state">
                        <div className="empty-icon">🌍</div>
                        <h2>Welcome to Eurus</h2>
                        <p>Ask about ERA5 climate data — SST, wind, precipitation, temperature and more.</p>
                        <p className="empty-warning">
                            ⚠️ <strong>Experimental</strong> — research prototype. Avoid very large datasets. Use 📦 Arraylake Code for heavy workloads.
                        </p>
                        <div className="example-queries">
                            <button onClick={() => { setInput('Show SST for California coast, Jan 2024'); }}>
                                🌡 SST — California coast
                            </button>
                            <button onClick={() => { setInput('Compare wind speed Berlin vs Tokyo, March 2023'); }}>
                                💨 Wind — Berlin vs Tokyo
                            </button>
                            <button onClick={() => { setInput('Precipitation anomalies over Amazon, 2023'); }}>
                                🌧 Rain — Amazon basin
                            </button>
                        </div>
                    </div>
                )}
                {messages.map((m) => <MessageBubble key={m.id} msg={m} />)}
                {(isThinking || statusMsg) && (
                    <div className="thinking-indicator">
                        <Loader2 className="spin" size={16} />
                        <span>{statusMsg || 'Analyzing...'}</span>
                    </div>
                )}
                <div ref={bottomRef} />
            </div>

            {/* input */}
            <div className="input-bar">
                <textarea
                    ref={textareaRef}
                    value={input}
                    onChange={handleInputChange}
                    onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
                    placeholder={canSend ? 'Ask about climate data…' : needKeys ? 'Enter API keys above…' : 'Connecting…'}
                    disabled={!canSend}
                    rows={1}
                />
                <button
                    className="send-btn"
                    onClick={handleSend}
                    disabled={!input.trim() || !canSend}
                >
                    <Send size={18} />
                </button>
            </div>
        </div>
    );
}
