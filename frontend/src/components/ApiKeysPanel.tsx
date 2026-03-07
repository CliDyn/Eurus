import { useState, useEffect } from 'react';
import { Key } from 'lucide-react';
import './ApiKeysPanel.css';

interface ApiKeysPanelProps {
    visible: boolean;
    onSave: (keys: { openai_api_key: string; arraylake_api_key: string }) => void;
}

export default function ApiKeysPanel({ visible, onSave }: ApiKeysPanelProps) {
    const [openaiKey, setOpenaiKey] = useState('');
    const [arraylakeKey, setArraylakeKey] = useState('');
    const [saving, setSaving] = useState(false);

    // Restore from sessionStorage
    useEffect(() => {
        const saved = sessionStorage.getItem('eurus-keys');
        if (saved) {
            try {
                const k = JSON.parse(saved);
                if (k.openai_api_key) setOpenaiKey(k.openai_api_key);
                if (k.arraylake_api_key) setArraylakeKey(k.arraylake_api_key);
            } catch { /* ignore */ }
        }
    }, []);

    if (!visible) return null;

    const handleSubmit = () => {
        if (!openaiKey.trim()) return;
        setSaving(true);
        onSave({ openai_api_key: openaiKey.trim(), arraylake_api_key: arraylakeKey.trim() });
    };

    return (
        <div className="keys-panel">
            <div className="keys-header">
                <Key size={16} />
                <span>API Keys Required</span>
            </div>
            <p className="keys-note">
                Enter your API keys to use Eurus. Keys are kept in your browser session only — cleared when you close the browser.
            </p>
            <div className="keys-field">
                <label>OpenAI API Key <span className="required">*</span></label>
                <input
                    type="password"
                    value={openaiKey}
                    onChange={e => setOpenaiKey(e.target.value)}
                    placeholder="sk-..."
                    autoComplete="off"
                    onKeyDown={e => { if (e.key === 'Enter') handleSubmit(); }}
                />
            </div>
            <div className="keys-field">
                <label>Arraylake API Key</label>
                <input
                    type="password"
                    value={arraylakeKey}
                    onChange={e => setArraylakeKey(e.target.value)}
                    placeholder="ema_..."
                    autoComplete="off"
                    onKeyDown={e => { if (e.key === 'Enter') handleSubmit(); }}
                />
            </div>
            <button className="keys-submit" onClick={handleSubmit} disabled={saving || !openaiKey.trim()}>
                {saving ? 'Connecting...' : 'Connect'}
            </button>
        </div>
    );
}
