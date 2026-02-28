# Urban Environmental Intelligence 2025

An interactive **Streamlit dashboard** and accompanying analysis pipeline
that explores a multi‑gigabyte air‑quality dataset collected from 100
worldwide sensor nodes during the year 2025.  The system is designed as
part of a Smart City initiative where the lead data architect must
identify anomalies and maximise visual integrity.

## 📂 Repository Contents

```
├── data/                      # raw CSV export from OpenAQ
│   └── openaq_2025.csv        # ~small-size placeholder for assignment
├── scripts/                   # stand‑alone analysis modules
│   ├── pca_analysis.py        # Task 1: dimensionality challenge
│   ├── temporal_analysis.py   # Task 2: high‑density temporal plots
│   ├── distribution_analysis.py # Task 3: tail‑aware distributions
│   └── visual_audit.py        # Task 4: audit / small‑multiples
├── utils.py                   # shared data loader / preprocessing
├── dashboard.py               # Streamlit application (entry point)
├── requirements.txt           # Python dependencies
└── README.md                  # you are reading it 😉
```

## 🎯 Objective

Build a diagnostic engine that:

1. Reduces six correlated environmental measurements to two dimensions
   (PM2.5, PM10, NO2, Ozone, Temperature, Humidity) with PCA and shows
   how industrial vs residential zones separate. Loadings expose the
   pollutants driving the variance.
2. Provides a compact, high‑density temporal view of 100 hourly time
   series and reveals daily/seasonal pollution signatures.
3. Models distributions in an industrial zone, highlighting peaks and
   the long tail; computes the 99ᵗʰ percentile and the probability of
   extreme (PM2.5 > 200 µg/m³) events.
4. Audits a proposed 3‑D bar chart, rejects it via lie‑factor/data‑ink
   analysis, and replaces it with a small‑multiples/bivariate mapping
   solution using sequential colour scales.

All visualisations avoid 3‑D effects and unnecessary decorations. The
entire pipeline is modular Python – **no notebooks** – ensuring
reproducibility.

## 🚀 Getting Started

1. **Clone the repository** (or copy your local workspace into it):

   ```bash
   git clone https://github.com/AbdurRafayBaig/Urban-Environmental-Intelligence-2025.git
   cd Urban-Environmental-Intelligence-2025
   ```

2. **Install dependencies** (preferably in a virtual environment):

   ```bash
   python -m venv .venv          # optional but recommended
   source .venv/Scripts/activate # Windows
   pip install -r requirements.txt
   ```

3. **Run the dashboard**:

   ```bash
   python -m streamlit run dashboard.py
   ```

   The application will open at `http://localhost:8501`. Use the sidebar
to filter by zone, pollutant and date range; explore each analysis tab.

4. **Run scripts** (optional) for offline reports or development:

   ```bash
   python scripts/pca_analysis.py
   python scripts/temporal_analysis.py
   python scripts/distribution_analysis.py
   python scripts/visual_audit.py
   ```

## 📦 Publishing to GitHub

If you haven't yet initialised a Git repository locally, do so and set
the remote using the provided link:

```bash
cd D:/6\ Semester\ Subjects/Data\ Science/Assignment_02
git init
git add .
git commit -m "Initial commit: complete dashboard and analysis scripts"
git branch -M main
git remote add origin https://github.com/AbdurRafayBaig/Urban-Environmental-Intelligence-2025.git
git push -u origin main
```

Replace the remote URL with the one you create if it differs. After
pushing, future changes can be committed and pushed normally.

## 📝 License

This project is released under the MIT License – feel free to reuse or
adapt the code for educational purposes.

---

*Created for the Urban Environmental Intelligence Challenge, 2025.*

