# CMCSIF Portfolio Tracker

A full-stack portfolio analytics and monitoring platform built for the **Claremont McKenna College Student Investment Fund (CMCSIF)**.

This project provides a centralized dashboard for tracking:
- portfolio performance (returns, Sharpe, drawdowns)
- team-level allocation and contribution
- holding-level attribution
- basic factor exposures (size, value, momentum)
- live pricing via Yahoo Finance

The system is designed to support both **non-technical fund members** and **portfolio managers**, with an emphasis on transparency, usability, and real-time insight.

---

## 🚀 Features

### 📊 Portfolio Monitoring
- Total fund market value
- Daily P&L and return
- Team-level allocation breakdown
- Holdings snapshot view

### 📈 Performance Analytics
- Cumulative returns
- Annualized return & volatility
- Sharpe and Sortino ratios
- Max drawdown
- Rolling risk metrics

### 🧠 Attribution
- Top contributors and detractors
- Team-level contribution analysis
- Position-level return decomposition

### 🧪 Factor Exposure (MVP)
- Size (log market cap)
- Value (inverse price-to-book)
- Momentum (12-month return)

### 📂 Data Pipeline
- CSV / Excel upload workflow
- Column auto-detection & validation
- Sector → team mapping
- Ticker normalization (Yahoo-compatible)
- Snapshot storage with audit logging

---

## 🏗️ Architecture
