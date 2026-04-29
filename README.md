# Bangalore AQI Forecasting & Routing System

## Overview
A production-grade, journal-ready AQI forecasting & health-aware routing system for Bangalore. Uses a 2-year CPCB dataset (12 stations) to provide highly accurate predictions and hyper-local routing based on user health profiles.

## Architecture
- **Model**: ST-MHGTD (Spatio-Temporal Multi-Horizon Graph Transformer with Adaptive Diffusion & Physics-Regularization).
- **Frontend**: Next.js 14, TailwindCSS, Framer Motion.
- **Backend**: FastAPI, Redis, NetworkX for routing.

## Run Instructions

1. **Setup Environment**:
   ```bash
   make setup
   ```

2. **Run ETL Pipeline**:
   ```bash
   make etl
   ```

3. **Train Model**:
   ```bash
   make train
   ```

4. **Start API**:
   ```bash
   make run-api
   ```

5. **Start Frontend (Dev)**:
   ```bash
   make dev
   ```

6. **Docker Deploy**:
   ```bash
   make docker-up
   ```

## Journal Checklist
- [x] Zero Synthetic Data
- [x] Automated FE & Training
- [x] Config-Driven
- [x] Shape & Metric Assertions
- [x] Type Safety
- [x] Deterministic & Reproducible
- [x] Academic Rigor
