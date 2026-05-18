"use client";
import dynamic from "next/dynamic";
// Plotly is heavy — load only on the client.
export const Plot = dynamic(() => import("react-plotly.js"), { ssr: false }) as any;
