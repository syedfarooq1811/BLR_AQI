"use client";

import { useEffect, useState } from "react";
import { useRef } from "react";

export default function Home() {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const [iframeSrc, setIframeSrc] = useState("http://127.0.0.1:8001");

  useEffect(() => {
    const host = window.location.hostname || "127.0.0.1";
    const protocol = window.location.protocol || "http:";
    setIframeSrc(`${protocol}//${host}:8001`);
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
      }}
    >
      <iframe
        ref={iframeRef}
        src={iframeSrc}
        title="Bluru AQI & Traffic Dashboard"
        style={{
          width: "100%",
          height: "100%",
          border: "none",
          display: "block",
        }}
        allow="geolocation"
      />
    </main>
  );
}
