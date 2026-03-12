<div align="center">
  <h1>🎭 Real-Time Emotion Recognition AI</h1>
  <p><i>Developed with passion by <b>Pritam Kumar</b></i></p>
  
  <br>
  <a href="https://emotion-detection-ai-nwfybvjapxbs5jhbtpd4cy.streamlit.app" style="text-decoration: none;">
    <div style="
      display: inline-block;
      padding: 15px 40px;
      font-size: 20px;
      font-family: 'Courier New', Courier, monospace;
      font-weight: bold;
      color: #ffffff;
      background-color: #1e2130;
      border: 2px solid #ff4b4b;
      border-radius: 10px;
      box-shadow: 0 8px #ff4b4b;
      cursor: pointer;
      transition: all 0.2s ease;
      text-transform: uppercase;
      letter-spacing: 2px;">
      [ CLICK TO PREVIEW THE SYSTEM ]
    </div>
  </a>
  <p style="margin-top: 20px; color: #888;"><i>Best experienced on Desktop & Mobile</i></p>
  <br>
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
  <table width="100%" style="border-collapse: collapse;">
    <tr>
      <td width="33%" align="center" style="padding: 10px;">
        <div style="border: 1px solid #333; border-radius: 10px; padding: 10px; background: #161b22;">
          <b>Real-Time Live Feed</b><br><br>
          <img src="https://via.placeholder.com/400x250?text=Live+Result+Screenshot" width="100%" style="border-radius: 5px;" alt="Live Result">
        </div>
      </td>
      <td width="33%" align="center" style="padding: 10px;">
        <div style="border: 1px solid #333; border-radius: 10px; padding: 10px; background: #161b22;">
          <b>Group Photo Analysis</b><br><br>
          <img src="https://via.placeholder.com/400x250?text=Group+Result+Screenshot" width="100%" style="border-radius: 5px;" alt="Group Result">
        </div>
      </td>
      <td width="33%" align="center" style="padding: 10px;">
        <div style="border: 1px solid #333; border-radius: 10px; padding: 10px; background: #161b22;">
          <b>High-Res Uploads</b><br><br>
          <img src="https://via.placeholder.com/400x250?text=Upload+Result+Screenshot" width="100%" style="border-radius: 5px;" alt="Upload Result">
        </div>
      </td>
    </tr>
  </table>
</p>

<blockquote>
  <b>💡 Note:</b> Replace the placeholder images above with real screenshots from your app to showcase your actual engineering work.
</blockquote>

<hr>

<h2 id="key-features">🧠 Key Engineering Features</h2>
<ul>
  <li><b>Adaptive Rescaling:</b> Automatically handles high-resolution images to prevent OpenCV memory crashes (Fixed <code>scaleIdx</code> assertion errors).</li>
  <li><b>Contrast Enhancement (CLAHE):</b> Integrated Contrast Limited Adaptive Histogram Equalization for accurate detection in low-light and complex group settings.</li>
  <li><b>Temporal Stability:</b> Optimized for low-latency inference via Streamlit-WebRTC.</li>
  <li><b>Multi-Face Support:</b> Fine-tuned to detect up to 15 faces in a single frame.</li>
</ul>

<hr>

<h2 id="technical-stack">🛠️ Technical Stack</h2>


[Image of Convolutional Neural Network architecture]

<table width="100%">
  <tr>
    <td width="30%"><b>Deep Learning</b></td>
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
