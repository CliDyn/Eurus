import { useEffect, useState } from 'react';
import { Database, HardDrive, RefreshCw, Download } from 'lucide-react';
import './CachePanel.css';

interface Dataset {
    variable: string;
    query_type: string;
    start_date: string;
    end_date: string;
    lat_bounds: [number, number];
    lon_bounds: [number, number];
    file_size_bytes: number;
    path: string;
}

interface CacheData {
    datasets: Dataset[];
    total_size_bytes: number;
}

function formatBytes(bytes: number): string {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
}

async function downloadDataset(path: string) {
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
    } catch (err) {
        console.error('Download error:', err);
    }
}

export default function CachePanel() {
    const [cache, setCache] = useState<CacheData | null>(null);
    const [loading, setLoading] = useState(false);

    const fetchCache = async () => {
        setLoading(true);
        try {
            const res = await fetch('/api/cache');
            if (res.ok) setCache(await res.json());
        } catch { /* ignore */ }
        setLoading(false);
    };

    useEffect(() => { fetchCache(); }, []);

    return (
        <div className="cache-panel">
            <div className="cache-header">
                <div className="cache-title">
                    <Database size={16} />
                    <span>Cached Datasets</span>
                </div>
                <button className="cache-refresh" onClick={fetchCache} disabled={loading} title="Refresh">
                    <RefreshCw size={14} className={loading ? 'spin' : ''} />
                </button>
            </div>

            {cache && cache.datasets.length > 0 ? (
                <>
                    <div className="cache-summary">
                        <HardDrive size={13} />
                        <span>{cache.datasets.length} datasets · {formatBytes(cache.total_size_bytes)}</span>
                    </div>
                    <ul className="cache-list">
                        {cache.datasets.map((ds, i) => (
                            <li key={i} className="cache-item">
                                <div className="cache-item-row">
                                    <div>
                                        <div className="cache-var">{ds.variable}</div>
                                        <div className="cache-meta">
                                            {ds.start_date} → {ds.end_date} · {formatBytes(ds.file_size_bytes)}
                                        </div>
                                    </div>
                                    <button
                                        className="cache-dl-btn"
                                        onClick={() => downloadDataset(ds.path)}
                                        title="Download as ZIP"
                                    >
                                        <Download size={13} />
                                    </button>
                                </div>
                            </li>
                        ))}
                    </ul>
                </>
            ) : (
                <div className="cache-empty">No cached datasets yet</div>
            )}
        </div>
    );
}
