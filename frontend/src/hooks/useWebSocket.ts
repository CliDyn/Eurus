import { useCallback, useEffect, useRef, useState } from 'react';

/* ───────────────── Types ───────────────── */
export type WSEventType =
    | 'thinking'
    | 'status'
    | 'tool_start'
    | 'stream'
    | 'chunk'        // old backend sends 'chunk', we normalize to 'stream'
    | 'plot'
    | 'video'
    | 'complete'
    | 'error'
    | 'clear'
    | 'keys_configured'
    | 'request_keys'
    | 'arraylake_snippet';

export interface WSEvent {
    type: WSEventType;
    content?: string;
    ready?: boolean;
    reason?: string;
    data?: string;        // base64 payload for plot/video
    path?: string;        // file path for plot/video
    code?: string;        // code that generated the plot
    mimetype?: string;    // video mimetype
    [key: string]: unknown;
}

export type WSStatus = 'connecting' | 'connected' | 'disconnected';

interface UseWebSocketReturn {
    status: WSStatus;
    send: (payload: Record<string, unknown>) => void;
    sendMessage: (message: string) => void;
    configureKeys: (keys: { openai_api_key?: string; arraylake_api_key?: string; hf_token?: string }) => void;
    lastEvent: WSEvent | null;
}

/* ─────────────── Hook ─────────────── */
export function useWebSocket(onEvent?: (event: WSEvent) => void): UseWebSocketReturn {
    const wsRef = useRef<WebSocket | null>(null);
    const [status, setStatus] = useState<WSStatus>('disconnected');
    const [lastEvent, setLastEvent] = useState<WSEvent | null>(null);
    const reconnectTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
    const onEventRef = useRef(onEvent);
    const connectingRef = useRef(false);   // guard against double-connect
    onEventRef.current = onEvent;

    const connect = useCallback(() => {
        // Guard: don't open a second WS if one is OPEN or CONNECTING
        if (wsRef.current) {
            const rs = wsRef.current.readyState;
            if (rs === WebSocket.OPEN || rs === WebSocket.CONNECTING) return;
        }
        if (connectingRef.current) return;
        connectingRef.current = true;

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const ws = new WebSocket(`${protocol}//${window.location.host}/ws/chat`);
        wsRef.current = ws;
        setStatus('connecting');

        ws.onopen = () => {
            connectingRef.current = false;
            setStatus('connected');
            // Auto-resend keys from sessionStorage on reconnect
            const saved = sessionStorage.getItem('eurus-keys');
            if (saved) {
                try {
                    const keys = JSON.parse(saved);
                    if (keys.openai_api_key) {
                        ws.send(JSON.stringify({ type: 'configure_keys', ...keys }));
                    }
                } catch { /* ignore */ }
            }
        };

        ws.onmessage = (e) => {
            try {
                const event: WSEvent = JSON.parse(e.data);
                // Normalize 'chunk' → 'stream' for unified handling
                if (event.type === 'chunk') {
                    event.type = 'stream';
                }
                setLastEvent(event);
                onEventRef.current?.(event);
            } catch { /* ignore non-json */ }
        };

        ws.onclose = () => {
            connectingRef.current = false;
            setStatus('disconnected');
            wsRef.current = null;
            // auto-reconnect after 2 s
            reconnectTimer.current = setTimeout(connect, 2000);
        };

        ws.onerror = () => {
            connectingRef.current = false;
            ws.close();
        };
    }, []);

    useEffect(() => {
        connect();
        return () => {
            clearTimeout(reconnectTimer.current);
            connectingRef.current = false;
            if (wsRef.current) {
                wsRef.current.onclose = null;  // prevent reconnect on cleanup close
                wsRef.current.close();
                wsRef.current = null;
            }
        };
    }, [connect]);

    const send = useCallback((payload: Record<string, unknown>) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(payload));
        }
    }, []);

    const sendMessage = useCallback((message: string) => send({ message }), [send]);

    const configureKeys = useCallback(
        (keys: { openai_api_key?: string; arraylake_api_key?: string; hf_token?: string }) => {
            // Save to sessionStorage
            sessionStorage.setItem('eurus-keys', JSON.stringify(keys));
            send({ type: 'configure_keys', ...keys });
        },
        [send],
    );

    return { status, send, sendMessage, configureKeys, lastEvent };
}
