import "./LandingPage.css";

function LandingPage({ onEnter }) {
  return (
    <main className="landing">
      {/* Animated background */}
      <div className="microscope-bg">
        <div className="fibre fibre-1"></div>
        <div className="fibre fibre-2"></div>
        <div className="fibre fibre-3"></div>
        <div className="fibre fibre-4"></div>
        <div className="fibre fibre-5"></div>

        <div className="particle p1"></div>
        <div className="particle p2"></div>
        <div className="particle p3"></div>
        <div className="particle p4"></div>
        <div className="particle p5"></div>
        <div className="particle p6"></div>

        <div className="scan-line"></div>
        <div className="noise"></div>
      </div>

      {/* Top navigation */}
      <nav className="topbar">
        <div className="brand">
          <span className="brand-mark"></span>

          <div>
            <strong>SEM ANALYSIS</strong>
            <small>MICROSTRUCTURAL INTELLIGENCE</small>
          </div>
        </div>

        <div className="system-status">
          <span className="status-dot"></span>
          SYSTEM READY
        </div>
      </nav>

      {/* Hero */}
      <section className="hero">
        <div className="eyebrow">
          AI-POWERED COMPOSITE MICROSTRUCTURE ANALYSIS
        </div>

        <h1>
          See beyond the
          <span> microstructure.</span>
        </h1>

        <p className="hero-description">
          Deep-learning based SEM analysis of woven GFRP/PP composites
          for automated glass-fibre segmentation and quantitative
          microstructural assessment.
        </p>

        <div className="tech-row">
          <span>U-NET</span>
          <i></i>
          <span>SVM</span>
          <i></i>
          <span>SEM</span>
          <i></i>
          <span>GFRP / PP</span>
        </div>

        <button className="enter-button" onClick={onEnter}>
          <span>ENTER ANALYSIS SYSTEM</span>
          <b>→</b>
        </button>
      </section>

      {/* Performance panel */}
      <section className="performance-panel">
        <div className="metric">
          <span>DATASET</span>
          <strong>48</strong>
          <small>Original specimens</small>
        </div>

        <div className="divider"></div>

        <div className="metric">
          <span>U-NET TEST DICE</span>
          <strong>93.54<sup>%</sup></strong>
          <small>Untouched test set</small>
        </div>

        <div className="divider"></div>

        <div className="metric">
          <span>TEST PATCHES</span>
          <strong>179</strong>
          <small>Independent evaluation</small>
        </div>

        <div className="divider"></div>

        <div className="metric">
          <span>MODELS</span>
          <strong>02</strong>
          <small>U-Net + SVM</small>
        </div>
      </section>

      {/* Decorative microscopy HUD */}
      <div className="hud hud-left">
        <span>SEM</span>
        <div></div>
        <small>256 × 256 PX</small>
      </div>

      <div className="hud hud-right">
        <small>SEGMENTATION ENGINE</small>
        <span>ONLINE</span>
      </div>

      <div className="corner corner-tl"></div>
      <div className="corner corner-tr"></div>
      <div className="corner corner-bl"></div>
      <div className="corner corner-br"></div>

      <footer>
        FINAL YEAR MAJOR PROJECT
        <span>•</span>
        COMPUTER SCIENCE & ENGINEERING
      </footer>
    </main>
  );
}

export default LandingPage;