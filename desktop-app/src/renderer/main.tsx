import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./App";
import { createDemoFervisClient } from "./demoClient";
import "./styles/app.css";

const root = document.getElementById("root");

if (root === null) {
  throw new Error("Fervis desktop app root element was not found");
}

createRoot(root).render(
  <StrictMode>
    <App initialClient={demoModeEnabled() ? createDemoFervisClient() : null} />
  </StrictMode>
);

function demoModeEnabled(): boolean {
  return new URLSearchParams(window.location.search).get("demo") === "1";
}
