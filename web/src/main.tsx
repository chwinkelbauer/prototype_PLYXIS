import React from "react";
import ReactDOM from "react-dom/client";
import "./theme.css";
import "maplibre-gl/dist/maplibre-gl.css";
import App from "./App";
import { connect } from "./store";

connect();
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
