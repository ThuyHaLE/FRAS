// dashboard/src/main.jsx
// This file is the entry point for the React application. 
// It renders the FireRiskDashboard component into the root element of the HTML page.

import React from "react";
import ReactDOM from "react-dom/client";
import FireRiskDashboard from "./FireRiskDashboard";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <FireRiskDashboard />
  </React.StrictMode>
);