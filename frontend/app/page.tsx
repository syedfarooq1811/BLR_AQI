"use client";

import { useRef } from "react";

export const metadata = undefined;

export default function Home() {
  const iframeRef = useRef<HTMLIFrameElement>(null);

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
        src="http://localhost:8000"
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
