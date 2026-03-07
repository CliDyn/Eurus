import { useState, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import { Maximize2, Download, Code, X } from 'lucide-react';
import './MessageBubble.css';

export type MessageRole = 'user' | 'assistant' | 'system';

export interface MediaItem {
    type: 'plot' | 'video';
    base64: string;
    path?: string;
    code?: string;
    mimetype?: string;
}

export interface ChatMessage {
    id: string;
    role: MessageRole;
    content: string;
    plots?: string[];          // base64 PNG strings (legacy)
    media?: MediaItem[];       // full media items (plot + video)
    toolLabel?: string;        // e.g. "Fetching ERA5..."
    statusText?: string;       // e.g. "🔍 Analyzing..."
    arraylakeSnippets?: string[];
    isStreaming?: boolean;
}

/* ── Image Modal ── */
function ImageModal({ src, onClose }: { src: string; onClose: () => void }) {
    return (
        <div className="image-modal-overlay" onClick={onClose}>
            <div className="image-modal-box" onClick={e => e.stopPropagation()}>
                <img src={src} alt="Enlarged plot" />
                <div className="image-modal-actions">
                    <button className="modal-btn" onClick={() => {
                        const a = document.createElement('a');
                        a.href = src;
                        a.download = 'eurus_plot.png';
                        a.click();
                    }}>
                        <Download size={14} /> Download
                    </button>
                    <button className="modal-btn modal-close" onClick={onClose}>
                        <X size={14} /> Close
                    </button>
                </div>
            </div>
        </div>
    );
}

/* ── Plot Figure ── */
function PlotFigure({ item, onEnlarge }: { item: MediaItem; onEnlarge: (src: string) => void }) {
    const [showCode, setShowCode] = useState(false);
    const src = item.base64.startsWith('data:') ? item.base64 : `data:image/png;base64,${item.base64}`;

    return (
        <figure className="plot-figure">
            <img
                src={src}
                alt="Generated plot"
                className="plot-img"
                onClick={() => onEnlarge(src)}
                style={{ cursor: 'pointer' }}
            />
            <div className="plot-actions">
                <button className="plot-action-btn" onClick={() => onEnlarge(src)} title="Enlarge">
                    <Maximize2 size={13} /> Enlarge
                </button>
                <button className="plot-action-btn" onClick={() => {
                    const a = document.createElement('a');
                    a.href = src;
                    a.download = item.path ? item.path.split('/').pop()! : 'eurus_plot.png';
                    a.click();
                }} title="Download">
                    <Download size={13} /> Download
                </button>
                {item.code && item.code.trim() && (
                    <button className="plot-action-btn" onClick={() => setShowCode(v => !v)} title="Toggle code">
                        <Code size={13} /> {showCode ? 'Hide Code' : 'Show Code'}
                    </button>
                )}
            </div>
            {showCode && item.code && (
                <div className="plot-code-block">
                    <pre><code>{item.code}</code></pre>
                    <button className="copy-btn" onClick={() => {
                        navigator.clipboard.writeText(item.code!);
                    }}>Copy</button>
                </div>
            )}
        </figure>
    );
}

/* ── Video Figure ── */
function VideoFigure({ item, onEnlarge }: { item: MediaItem; onEnlarge: (src: string) => void }) {
    const isGif = item.mimetype === 'image/gif';
    const src = item.base64.startsWith('data:') ? item.base64 : `data:${item.mimetype || 'video/mp4'};base64,${item.base64}`;

    const handleDownload = () => {
        const a = document.createElement('a');
        a.href = src;
        const ext = isGif ? 'gif' : item.mimetype?.includes('webm') ? 'webm' : 'mp4';
        a.download = item.path ? item.path.split('/').pop()! : `eurus_animation.${ext}`;
        a.click();
    };

    if (isGif) {
        return (
            <figure className="plot-figure">
                <img src={src} alt="Animation" className="plot-img" onClick={() => onEnlarge(src)} style={{ cursor: 'pointer' }} />
                <div className="plot-actions">
                    <button className="plot-action-btn" onClick={() => onEnlarge(src)}><Maximize2 size={13} /> Enlarge</button>
                    <button className="plot-action-btn" onClick={handleDownload}><Download size={13} /> Download</button>
                </div>
            </figure>
        );
    }

    return (
        <figure className="plot-figure">
            <video controls autoPlay loop muted playsInline style={{ maxWidth: '100%', borderRadius: '8px' }}>
                <source src={src} type={item.mimetype || 'video/mp4'} />
            </video>
            <div className="plot-actions">
                <button className="plot-action-btn" onClick={handleDownload}><Download size={13} /> Download</button>
            </div>
        </figure>
    );
}

/* ── Arraylake Snippet ── */
function ArraylakeSnippet({ code }: { code: string }) {
    const [open, setOpen] = useState(false);
    // Strip markdown fences
    const clean = code
        .replace(/^\n?📦[^\n]*\n/, '')
        .replace(/^```python\n?/, '')
        .replace(/\n?```$/, '')
        .trim();

    return (
        <div className="arraylake-section">
            <button className="plot-action-btn arraylake-btn" onClick={() => setOpen(v => !v)}>
                📦 {open ? 'Hide Arraylake' : 'Arraylake Code'}
            </button>
            {open && (
                <div className="plot-code-block">
                    <pre><code>{clean}</code></pre>
                    <button className="copy-btn" onClick={() => navigator.clipboard.writeText(clean)}>Copy</button>
                </div>
            )}
        </div>
    );
}

/* ── Legacy plot ── */
function LegacyPlotImage({ base64, onEnlarge }: { base64: string; onEnlarge: (src: string) => void }) {
    const src = base64.startsWith('data:') ? base64 : `data:image/png;base64,${base64}`;
    return (
        <figure className="plot-figure">
            <img src={src} alt="Generated plot" className="plot-img" onClick={() => onEnlarge(src)} style={{ cursor: 'pointer' }} />
            <div className="plot-actions">
                <button className="plot-action-btn" onClick={() => onEnlarge(src)}><Maximize2 size={13} /> Enlarge</button>
                <button className="plot-action-btn" onClick={() => {
                    const a = document.createElement('a'); a.href = src; a.download = 'eurus_plot.png'; a.click();
                }}><Download size={13} /> Download</button>
            </div>
        </figure>
    );
}

/* ── Main Bubble ── */
export default function MessageBubble({ msg }: { msg: ChatMessage }) {
    const isUser = msg.role === 'user';
    const [modalSrc, setModalSrc] = useState<string | null>(null);

    const handleEnlarge = useCallback((src: string) => setModalSrc(src), []);

    return (
        <>
            <div className={`bubble-row ${isUser ? 'user' : 'assistant'}`}>
                <div className={`bubble ${isUser ? 'bubble-user' : 'bubble-assistant'}`}>
                    {msg.toolLabel && (
                        <div className="tool-label">⚙ {msg.toolLabel}</div>
                    )}

                    {msg.statusText && (
                        <div className="status-text">{msg.statusText}</div>
                    )}

                    <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
                        {msg.content}
                    </ReactMarkdown>

                    {/* Legacy plots */}
                    {msg.plots?.map((b64, i) => <LegacyPlotImage key={`p-${i}`} base64={b64} onEnlarge={handleEnlarge} />)}

                    {/* Rich media (plots + videos) */}
                    {msg.media?.map((item, i) =>
                        item.type === 'video'
                            ? <VideoFigure key={`v-${i}`} item={item} onEnlarge={handleEnlarge} />
                            : <PlotFigure key={`m-${i}`} item={item} onEnlarge={handleEnlarge} />
                    )}

                    {/* Arraylake snippets */}
                    {msg.arraylakeSnippets?.map((s, i) => <ArraylakeSnippet key={`al-${i}`} code={s} />)}

                    {msg.isStreaming && !msg.media?.length && !msg.arraylakeSnippets?.length && <span className="cursor-blink">▍</span>}
                </div>
            </div>

            {modalSrc && <ImageModal src={modalSrc} onClose={() => setModalSrc(null)} />}
        </>
    );
}
