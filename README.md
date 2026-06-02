# ⚡ NEXUS Energy Intelligence Dashboard

> A dark-themed, multi-page global energy analytics dashboard powered by OWID data and Streamlit.

![Python](https://img.shields.io/badge/Python-3.9+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red) ![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## 📸 Features

| Page | Description |
|------|-------------|
| 🏠 Mission Control | Hero landing page with boot sequence & KPI cards |
| 📊 Executive Overview | Global energy mix trends & top country rankings |
| ⚡ Energy Consumption | Primary energy trends, per capita analysis |
| 🌱 Renewables Intelligence | Solar, wind, hydro growth tracking |
| 🔥 Fossil Fuel Analytics | Coal, oil, gas dependency tracking |
| 💨 CO₂ & Emissions | Carbon intensity & emissions bubbles |
| 🔮 Predictive Analytics | ML forecasting with confidence intervals |
| 🌍 Country Deep Dive | Full single-country energy profile |
| 📡 Real-Time Monitor | Live global leaderboards & rankings |

---

## 🛠️ Tech Stack

- **Streamlit** — Web app framework
- **Plotly** — Interactive charts
- **Pandas / NumPy** — Data processing
- **Scikit-learn** — ML predictions (Polynomial Regression)
- **Our World in Data** — Energy dataset (auto-downloaded)

---

## 📦 Installation & Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/nexus-energy-dashboard.git
cd nexus-energy-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"**
4. Select your repo → `app.py`
5. Click **Deploy** — done! 🎉

---

## 📊 Data Source

Data from [Our World in Data — Energy](https://ourworldindata.org/energy)  
Auto-fetched from: `https://owid-public.owid.io/data/energy/owid-energy-data.csv`

Covers: 200+ countries, 1965–present, 100+ energy metrics.

---

## 📁 Project Structure

```
nexus-energy-dashboard/
│
├── app.py              ← Main Streamlit application
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## 👤 Author

Built with ❤️ using Streamlit + OWID Energy Data

---

## 📄 License

MIT License — free to use and modify.
