<div align="center">
  <h1>🎭 Real-Time Emotion Recognition AI</h1>
  <p><i>Developed with passion by <b>Pritam Kumar</b></i></p>
  
  <a href="https://emotion-detection-ai-nwfybvjapxbs5jhbtpd4cy.streamlit.app">
    <img src="https://static.streamlit.io/badges/streamlit_badge_svg.svg" alt="Streamlit App">
  </a>
  <br><br>
  <strong>A high-fidelity Deep Learning application optimized for real-time and group facial analysis.</strong>
</div>

<hr>

<div align="center">
  <a href="https://emotion-detection-ai-nwfybvjapxbs5jhbtpd4cy.streamlit.app">🚀 Live Demo</a> •
  <a href="#key-features">🧠 Features</a> •
  <a href="#technical-stack">🛠️ Tech Stack</a> •
  <a href="#contact">📬 Contact</a>
</div>

<hr>

<h2 id="showcase">📸 Project Showcase</h2>
<p align="center">
  <table width="100%">
    <tr>
      <td width="33%" align="center">
        <b>Real-Time Live Feed</b><br>
        <img src="https://via.placeholder.com/400x250?text=Live+Result+Screenshot" width="100%" alt="Live Result">
      </td>
      <td width="33%" align="center">
        <b>Group Photo Analysis</b><br>
        <img src="https://via.placeholder.com/400x250?text=Group+Result+Screenshot" width="100%" alt="Group Result">
      </td>
      <td width="33%" align="center">
        <b>High-Res Uploads</b><br>
        <img src="https://via.placeholder.com/400x250?text=Upload+Result+Screenshot" width="100%" alt="Upload Result">
      </td>
    </tr>
  </table>
</p>

<blockquote>
  <b>💡 Note:</b> To showcase your real work, replace the placeholder links above with your actual result images stored in your repository.
</blockquote>

<hr>

<h2 id="key-features">🧠 Key Engineering Features</h2>
<ul>
  <li><b>Adaptive Rescaling:</b> Automatically handles high-resolution images to prevent OpenCV memory crashes (Fixed <code>scaleIdx</code> assertion errors).</li>
  <li><b>Contrast Enhancement (CLAHE):</b> Integrated Contrast Limited Adaptive Histogram Equalization for accurate detection in low-light and complex group settings.</li>
  <li><b>Temporal Stability:</b> Optimized for low-latency inference on both Mobile and Desktop browsers via Streamlit-WebRTC.</li>
  <li><b>Multi-Face Support:</b> Specifically fine-tuned to detect up to 15 faces in a single frame.</li>
</ul>

<hr>

<h2 id="technical-stack">🛠️ Technical Stack</h2>
<table>
  <tr>
    <td><b>Deep Learning</b></td>
    <td>TensorFlow, Keras (CNN Architecture), FER2013 Dataset</td>
  </tr>
  <tr>
    <td><b>Computer Vision</b></td>
    <td>OpenCV (Haar-Cascade Classifiers)</td>
  </tr>
  <tr>
    <td><b>Web Framework</b></td>
    <td>Streamlit, Streamlit-WebRTC</td>
  </tr>
  <tr>
    <td><b>Language</b></td>
    <td>Python 3.10+</td>
  </tr>
</table>

<hr>

<h2>⚙️ Setup Instructions</h2>

```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/emotion-detection-ai.git](https://github.com/YOUR_USERNAME/emotion-detection-ai.git)

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
