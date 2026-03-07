import { useState, useEffect, useRef } from 'react';
import { ChevronDown } from 'lucide-react';
import './ModelSelector.css';

interface ModelOption {
    id: string;
    label: string;
    provider: string;
}

interface ModelSelectorProps {
    send: (payload: Record<string, unknown>) => void;
}

export default function ModelSelector({ send }: ModelSelectorProps) {
    const [current, setCurrent] = useState('gpt-5.2');
    const [open, setOpen] = useState(false);
    const ref = useRef<HTMLDivElement>(null);

    const models: ModelOption[] = [
        { id: 'gpt-5.2', label: 'GPT-5.2', provider: 'openai' },
        { id: 'gpt-4.1', label: 'GPT-4.1', provider: 'openai' },
        { id: 'o3', label: 'o3', provider: 'openai' },
        { id: 'gemini-3.1-pro-preview', label: 'Gemini 3.1 Pro', provider: 'google' },
    ];

    // Close on click outside
    useEffect(() => {
        if (!open) return;
        const handleClick = (e: MouseEvent) => {
            if (ref.current && !ref.current.contains(e.target as Node)) {
                setOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClick);
        return () => document.removeEventListener('mousedown', handleClick);
    }, [open]);

    const handleSelect = (modelId: string) => {
        setCurrent(modelId);
        setOpen(false);
        send({ type: 'set_provider', model: modelId });
    };

    const currentLabel = models.find(m => m.id === current)?.label ?? current;

    return (
        <div className="model-selector" ref={ref}>
            <button
                className="model-selector-btn"
                onClick={() => setOpen(v => !v)}
                title="Switch AI model"
            >
                <span className="model-label">{currentLabel}</span>
                <ChevronDown size={13} className={`chevron ${open ? 'open' : ''}`} />
            </button>
            {open && (
                <div className="model-dropdown">
                    {models.map(m => (
                        <button
                            key={m.id}
                            className={`model-option ${m.id === current ? 'active' : ''}`}
                            onClick={() => handleSelect(m.id)}
                        >
                            <span>{m.label}</span>
                            {m.id === current && <span className="check">✓</span>}
                        </button>
                    ))}
                </div>
            )}
        </div>
    );
}
