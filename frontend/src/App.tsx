import { useState } from 'react';
import { Database } from 'lucide-react';
import ChatPanel from './components/ChatPanel';
import CachePanel from './components/CachePanel';
import './App.css';

export default function App() {
    const [showCache, setShowCache] = useState(false);

    return (
        <div className="app-layout">
            <ChatPanel
                cacheToggle={
                    <button
                        className="icon-btn"
                        onClick={() => setShowCache(v => !v)}
                        title={showCache ? 'Hide datasets' : 'Show cached datasets'}
                        style={showCache ? { background: 'rgba(109,92,255,0.12)', borderColor: 'rgba(109,92,255,0.25)', color: '#a78bfa' } : undefined}
                    >
                        <Database size={16} />
                    </button>
                }
            />
            {showCache && <CachePanel />}
        </div>
    );
}
