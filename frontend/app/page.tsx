"use client";

import { useEffect, useRef, useState } from "react";

export const metadata = undefined; // metadata exported from client components is not supported; set in layout.tsx

export default function Home() {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const iframe = iframeRef.current;
    if (!iframe) return;
    const onLoad = () => setLoading(false);
    iframe.addEventListener("load", onLoad);
    return () => iframe.removeEventListener("load", onLoad);
  }, []);

  return (
    <main
      style={{
        margin: 0,
        padding: 0,
        width: "100vw",
        height: "100vh",
        overflow: "hidden",
        background: "#080b10",
        fontFamily: "'Outfit', sans-serif",
        position: "relative",
      }}
    >
      {/* Loading overlay */}
      {loading && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            zIndex: 10,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            background: "#080b10",
            gap: "20px",
          }}
        >
          {/* Animated rings */}
          <div style={{ position: "relative", width: 72, height: 72 }}>
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                style={{
                  position: "absolute",
                  inset: i * 12,
                  borderRadius: "50%",
                  border: "2px solid transparent",
                  borderTopColor: ["#3b82f6", "#6366f1", "#10b981"][i],
                  animation: `spin ${0.8 + i * 0.3}s linear infinite`,
                }}
              />
            ))}
          </div>
          <p
            style={{
              margin: 0,
              fontSize: "0.9rem",
              fontWeight: 500,
              color: "#64748b",
              letterSpacing: "0.06em",
              textTransform: "uppercase",
            }}
          >
            Loading Bluru AQI Dashboard…
          </p>
          <style>{`
            @keyframes spin { to { transform: rotate(360deg); } }
          `}</style>
        </div>
      )}

      {/* Full-viewport iframe embedding the FastAPI-served frontend */}
      <iframe
        ref={iframeRef}
        src="http://localhost:8000"
        title="Bluru AQI & Traffic Dashboard"
        style={{
          width: "100%",
          height: "100%",
          border: "none",
          display: "block",
          opacity: loading ? 0 : 1,
          transition: "opacity 0.5s ease",
        }}
        allow="geolocation"
      />
    </main>
  );
}
