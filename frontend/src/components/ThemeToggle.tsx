import { useState, useEffect } from 'react';
import { Sun, Moon } from 'lucide-react';
import './ThemeToggle.css';

export default function ThemeToggle() {
    const [light, setLight] = useState(() => {
        return localStorage.getItem('eurus-theme') === 'light';
    });

    useEffect(() => {
        const root = document.documentElement;
        if (light) {
            root.classList.add('light-theme');
        } else {
            root.classList.remove('light-theme');
        }
        localStorage.setItem('eurus-theme', light ? 'light' : 'dark');
    }, [light]);

    return (
        <button
            className="theme-toggle-btn"
            onClick={() => setLight(prev => !prev)}
            title={light ? 'Switch to dark mode' : 'Switch to light mode'}
            aria-label="Toggle theme"
        >
            <div className={`toggle-track ${light ? 'light' : 'dark'}`}>
                <div className="toggle-thumb">
                    {light ? <Sun size={12} /> : <Moon size={12} />}
                </div>
            </div>
        </button>
    );
}
