import { useState } from "react";
import LandingPage from "./LandingPage";

function App() {
  const [entered, setEntered] = useState(false);

  if (entered) {
    return (
      <div
        style={{
          width: "100vw",
          height: "100vh",
          display: "grid",
          placeItems: "center",
          background: "#050a0a",
          color: "white",
          fontFamily: "Inter, sans-serif",
        }}
      >
        <div style={{ textAlign: "center" }}>
          <p style={{ color: "#79dbae", letterSpacing: "4px" }}>
            SYSTEM INITIALIZED
          </p>

          <h1>SEM Analysis Dashboard</h1>

          <p style={{ color: "#718079" }}>
            Dashboard construction is next.
          </p>
        </div>
      </div>
    );
  }

  return <LandingPage onEnter={() => setEntered(true)} />;
}

export default App;