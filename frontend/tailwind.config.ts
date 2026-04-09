import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        ink: "#070b10",
        panel: "#0c131b",
        steel: "#101926",
        line: "rgba(255,255,255,0.08)",
        mist: "#eaf1ff",
        ember: "#ff6b35",
        gold: "#f6c56d",
        cyan: "#63d4ff",
        pine: "#8fe3b0",
      },
      boxShadow: {
        halo: "0 0 0 1px rgba(255,255,255,0.08), 0 24px 80px rgba(0,0,0,0.45)",
        glow: "0 20px 60px rgba(99,212,255,0.18)",
      },
      backgroundImage: {
        grid:
          "linear-gradient(to right, rgba(255,255,255,0.03) 1px, transparent 1px), linear-gradient(to bottom, rgba(255,255,255,0.03) 1px, transparent 1px)",
      },
    },
  },
  plugins: [],
};

export default config;
